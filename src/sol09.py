"""SOL 0.9 deployments (src/models/<model>_sol/lib_{cpu,gpu}, installed by scripts/sol_v09/install.sh).

Each library folder holds the SOL output (<model>.so, <model>.sol_weights, SOL runtime libs) and vaccel/ with the
generic wrapper (libsol_<model>_wrapper.so, scripts/sol_v09/wrapper.c) and manifest.json (resources, tensor
shapes). Two ways to run one:

  VaccelRunner  through vAccel exec (vaccel-local-sol / vaccel-remote-sol): LIB resource = the wrapper, DATA
                resource = model library + weights + SOL runtime libs (one folder, also on a remote agent); the
                wrapper opens the model library privately (RTLD_LOCAL) in sol_init_unpack, then one
                sol_predict_unpack call per inference
  DirectRunner  the SOL C API through ctypes in this process (backend "sol")

Both take a float32 host input and fill a float32 host output buffer (the same scope as the AOTInductor path, input
on host -> output on host). On GPU they copy H2D into a persistent device buffer, run SOL on device pointers on their own CUDA stream and copy
D2H into the output buffer; after 3 calls SOL's inference is captured into a CUDA graph and replayed (faster than
SOL's own offload path, identical results; SOL_CUDA_GRAPH=0 disables the graph). The GPU libraries must be deployed
with Conv_NoTextureMemory (scripts/sol_v09/deploy.py).

vaccel-remote-sol uses the zero-copy vAccel RPC client by default (VACCEL_RPC_CLIENT, see load_rpc_client).

GPU libraries run with cuDNN 9.20 only: src/models/runtime/cudnn-9.20, put first on the library path by
run_agent_sol.sh and evaluate_models_ros.sh for the SOL backends; a process with an older cuDNN (the client image's
torch alone loads 9.10) gets a clear error from the runners.
"""
import ctypes
import json
import os
from pathlib import Path

import numpy as np

MANIFEST = "manifest.json"


def manifest_dir(lib_path) -> Path:
    """<lib_{cpu,gpu}>/vaccel for a library folder or its vaccel/ subfolder."""
    p = Path(lib_path)
    return p if p.name == "vaccel" else p / "vaccel"


def is_sol09(lib_path) -> bool:
    f = manifest_dir(lib_path) / MANIFEST
    return f.is_file() and json.loads(f.read_text()).get("format") == "sol-0.9"


def read_manifest(lib_path) -> dict:
    d = manifest_dir(lib_path)
    m = json.loads((d / MANIFEST).read_text())
    m["dir"] = d
    m["input_shape"], m["output_shape"] = tuple(m["input_shape"]), tuple(m["output_shape"])
    return m


MIN_CUDNN = 92000   # the GPU libraries are built against cuDNN 9.20


def check_local_cudnn() -> None:
    """GPU library run in this process: fail clearly if the cuDNN loaded here (e.g. torch's) is too old, instead of
    SOL aborting inside cuDNN (CUDNN_STATUS_BAD_PARAM)."""
    try:
        v = ctypes.CDLL("libcudnn.so.9").cudnnGetVersion()
    except OSError:
        return
    if v < MIN_CUDNN:
        raise RuntimeError(f"SOL 0.9 GPU libraries need cuDNN >= 9.20 but this process uses {v}; put "
                           f"src/models/runtime/cudnn-9.20 first on LD_LIBRARY_PATH (as evaluate_models_ros.sh / run_agent_sol.sh do)")


# Zero-copy vAccel RPC client (scripts/vaccel_rpc): same wire protocol as the image's client, without its redundant
# copies of the request/response buffers. It replaces the RPC plugin's client functions when it is loaded
# RTLD_GLOBAL before the first remote session of the process (the plugin binds them lazily, at session creation).
# VACCEL_RPC_CLIENT=zero-copy (default) | stock.
RPC_CLIENT_LIB = Path(__file__).resolve().parent / "models" / "runtime" / "vaccel-rpc" / "libvaccel-rpc-client-zc.so"
_rpc_client = None   # (name, ctypes library or None), chosen once per process


def load_rpc_client():
    global _rpc_client
    if _rpc_client is None:
        want = os.environ.get("VACCEL_RPC_CLIENT", "zero-copy")
        if want == "stock":
            _rpc_client = ("stock", None)
        elif want == "zero-copy":
            if not RPC_CLIENT_LIB.is_file():
                raise RuntimeError(f"{RPC_CLIENT_LIB} missing: build it with scripts/vaccel_rpc/build.sh (and copy it to "
                                   f"this host), or set VACCEL_RPC_CLIENT=stock")
            lib = ctypes.CDLL(str(RPC_CLIENT_LIB), mode=os.RTLD_GLOBAL | os.RTLD_NOW)
            lib.vaccel_rpc_client_fastpath_calls.restype = ctypes.c_uint64
            _rpc_client = ("zero-copy", lib)
        else:
            raise ValueError(f"VACCEL_RPC_CLIENT must be 'zero-copy' or 'stock', not {want!r}")
    return _rpc_client


class VaccelRunner:
    def __init__(self, lib_path, use_remote: bool = False):
        import vaccel
        self.m = m = read_manifest(lib_path)
        if m["device"] == "gpu" and not use_remote:
            check_local_cudnn()
        self.rpc_client = None   # vaccel-remote only: "zero-copy" | "stock" (recorded with the results)
        if use_remote:
            name, zc = load_rpc_client()
            calls0 = zc.vaccel_rpc_client_fastpath_calls() if zc else 0
        d = m["dir"]
        plugin = vaccel.PluginType.GENERIC | (vaccel.PluginType.REMOTE if use_remote else 0)
        self._vaccel = vaccel
        self.session = vaccel.Session(plugin)
        files = lambda key: [str((d / f).resolve()) for f in m[key]]
        self.libs = vaccel.Resource(files("libs"), vaccel.ResourceType.LIB)    # the wrapper
        self.libs.register(self.session)
        self.data = vaccel.Resource(files("data"), vaccel.ResourceType.DATA)   # model library, weights, runtime libs
        self.data.register(self.session)
        did = self.data.remote_id if self.session.is_remote else self.data.id
        self.session.exec_with_resource(self.libs, "sol_init_unpack", [vaccel.Arg(int(did), vaccel.ArgType.INT64)])
        if use_remote:   # sol_init_unpack went through the RPC client: check which one served it
            if zc and zc.vaccel_rpc_client_fastpath_calls() == calls0:
                raise RuntimeError("zero-copy vAccel RPC client loaded but not in use (a remote vAccel session was "
                                   "created before it was loaded?); set VACCEL_RPC_CLIENT=stock or load SOL first")
            self.rpc_client = name

    def predict(self, x: np.ndarray, out: np.ndarray) -> np.ndarray:
        v = self._vaccel
        self.session.exec_with_resource(self.libs, "sol_predict_unpack",
                                        [v.Arg(x, v.ArgType.BUFFER)], [v.Arg(out, v.ArgType.BUFFER)])
        return out


class DirectRunner:
    def __init__(self, lib_path):
        self.m = m = read_manifest(lib_path)
        lib_dir = m["dir"].parent
        self.gpu = m["device"] == "gpu"
        if self.gpu:
            check_local_cudnn()
        # the model library finds its SOL runtime libraries itself (RPATH $ORIGIN)
        self.lib = lib = ctypes.CDLL(str((lib_dir / f"{m['model']}.so").resolve()))
        lib.sol_predict.restype = lib.sol_predict_create.restype = ctypes.c_void_p
        if self.gpu:   # as the vAccel wrapper (scripts/sol_v09/wrapper.c): own stream, device input buffer, offload=0,
            # CUDA graph after GRAPH_WARMUP plain calls (SOL_CUDA_GRAPH=0 disables it)
            vp, sz = ctypes.c_void_p, ctypes.c_size_t
            lib.sol_predict_create.argtypes = [ctypes.c_int, vp]
            lib.sol_predict.argtypes = [vp, vp, ctypes.c_int, vp]
            lib.sol_predict_free.argtypes = [vp, ctypes.c_int, vp]
            self.cudart = cr = ctypes.CDLL("libcudart.so.12")
            cr.cudaMalloc.argtypes = [ctypes.POINTER(vp), sz]
            cr.cudaMemcpyAsync.argtypes = [vp, vp, sz, ctypes.c_int, vp]
            cr.cudaStreamCreate.argtypes = [ctypes.POINTER(vp)]
            cr.cudaStreamSynchronize.argtypes = [vp]
            cr.cudaStreamBeginCapture.argtypes = [vp, ctypes.c_int]
            cr.cudaStreamEndCapture.argtypes = [vp, ctypes.POINTER(vp)]
            cr.cudaGraphInstantiateWithFlags.argtypes = [ctypes.POINTER(vp), vp, ctypes.c_ulonglong]
            cr.cudaGraphLaunch.argtypes = [vp, vp]
            self.stream, self.d_in = vp(), vp()
            n_in = int(np.prod(m["input_shape"])) * 4
            if cr.cudaStreamCreate(ctypes.byref(self.stream)) != 0 or cr.cudaMalloc(ctypes.byref(self.d_in), n_in) != 0:
                raise RuntimeError("CUDA stream/buffer allocation failed")
            self.h = lib.sol_predict_create(0, self.stream)
            self.graph_state = -1 if os.environ.get("SOL_CUDA_GRAPH") == "0" else 0
            self.graph_exec, self.graph_out = vp(), None
        else:
            lib.sol_predict_create.argtypes = []
            lib.sol_predict.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            lib.sol_predict_free.argtypes = [ctypes.c_void_p]
            self.h = lib.sol_predict_create()
        if not self.h:
            raise RuntimeError(f"sol_predict_create failed for {lib_dir}")

    GRAPH_WARMUP = 3

    def _gpu_plain(self, x, out):
        cr, st = self.cudart, self.stream
        if cr.cudaMemcpyAsync(self.d_in, x.ctypes.data, x.nbytes, 1, st) != 0:
            raise RuntimeError("cudaMemcpyAsync H2D failed")
        p = self.lib.sol_predict(self.h, self.d_in, 0, st)
        if not p:
            raise RuntimeError("sol_predict failed")
        ok = cr.cudaMemcpyAsync(out.ctypes.data, p, out.nbytes, 2, st) == 0 and cr.cudaStreamSynchronize(st) == 0
        self.lib.sol_predict_free(p, 0, st)
        if not ok:
            raise RuntimeError("cudaMemcpyAsync D2H failed")

    def _gpu_capture(self) -> bool:
        cr, st, graph = self.cudart, self.stream, ctypes.c_void_p()
        if cr.cudaStreamBeginCapture(st, 1) != 0:           # thread-local capture mode
            return False
        self.graph_out = self.lib.sol_predict(self.h, self.d_in, 0, st)
        if cr.cudaStreamEndCapture(st, ctypes.byref(graph)) != 0 or not graph.value or not self.graph_out:
            return False
        return cr.cudaGraphInstantiateWithFlags(ctypes.byref(self.graph_exec), graph, 1) == 0   # auto-free on launch

    def predict(self, x: np.ndarray, out: np.ndarray) -> np.ndarray:
        if self.gpu:   # H2D, SOL (plain or CUDA graph), D2H into `out`, synchronised
            if self.graph_state == self.GRAPH_WARMUP:
                self.graph_state = 99 if self._gpu_capture() else -1
            if self.graph_state == 99:
                cr, st = self.cudart, self.stream
                if (cr.cudaMemcpyAsync(self.d_in, x.ctypes.data, x.nbytes, 1, st) | cr.cudaGraphLaunch(self.graph_exec, st) |
                        cr.cudaMemcpyAsync(out.ctypes.data, self.graph_out, out.nbytes, 2, st) | cr.cudaStreamSynchronize(st)) != 0:
                    raise RuntimeError("CUDA graph inference failed")
            else:
                self._gpu_plain(x, out)
                if 0 <= self.graph_state < self.GRAPH_WARMUP:
                    self.graph_state += 1
            return out
        p = self.lib.sol_predict(self.h, x.ctypes.data)
        if not p:
            raise RuntimeError("sol_predict failed")
        ctypes.memmove(out.ctypes.data, p, out.nbytes)
        self.lib.sol_predict_free(p)
        return out
