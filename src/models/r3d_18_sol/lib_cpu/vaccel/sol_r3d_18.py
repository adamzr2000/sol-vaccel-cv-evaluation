import inspect
import logging
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import vaccel


class sol_r3d_18:
    def __init__(
        self,
        path=".",
        *,
        use_remote=False,
        enable_profiler=False,
        logger=None,
    ):
        self._path = Path(path)
        self._use_remote = use_remote
        self._enable_profiler = enable_profiler
        self._logger = logger or logging.getLogger(__name__)
        plugin_type = vaccel.PluginType.GENERIC
        if use_remote:
            plugin_type = plugin_type | vaccel.PluginType.REMOTE
        self._session = vaccel.Session(plugin_type)
        self._resource = None
        self._arg_resources = []
        self._libs = None
        self._parse_lib_manifest()

    def _parse_lib_manifest(self):
        manifest_path = self._path / "dlopen_manifest.txt"
        with manifest_path.open() as f:
            libs = [self._path.joinpath(line.strip()).resolve() for line in f]

        self._libs = libs

    @property
    def use_remote(self):
        return self._use_remote

    @contextmanager
    def profiler(self, name="block"):
        if not self._enable_profiler:
            with nullcontext():
                yield
                return

        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000
            class_name = self.__class__.__name__
            frame = inspect.currentframe().f_back.f_back
            caller_name = frame.f_code.co_name
            self._logger.info(
                "[%s.profiler] %s > %s: %.4f ms",
                class_name,
                caller_name,
                name,
                elapsed,
            )

    def open(self):
        pass

    def close(self):
        pass

    def call(self, *args):
        self.__call__(args)

    def init(self):
        resource = vaccel.Resource(self._libs, vaccel.ResourceType.LIB)
        resource.register(self._session)
        self._resource = resource

        self._session.exec_with_resource(
            resource,
            "sol_r3d_18_init_unpack",
        )

    def free(self):
        self._session.exec_with_resource(
            self._resource,
            "sol_r3d_18_free_unpack",
        )

    def sync(self):
        self._session.exec_with_resource(
            self._resource,
            "sol_r3d_18_sync_unpack",
        )

    def __call__(self, *args):
        return self.run(*args)

    def set_seed(self, seed):
        self._session.exec_with_resource(
            self._resource,
            "sol_r3d_18_set_seed_unpack",
            [vaccel.Arg(seed, vaccel.ArgType.INT64)],
        )

    def optimize(self, level):
        self._session.exec_with_resource(
            self._resource,
            "sol_r3d_18_optimize_unpack",
            [vaccel.Arg(level, vaccel.ArgType.INT32)],
        )

    def _prepare_run_args(self, in__input_0, out__0=None, vdims=None):
        if vdims is None:
            vdims_0 = in__input_0.shape[0]
            vdims = np.array(
                [
                    vdims_0,
                ],
                dtype=np.int64,
            )
        else:
            vdims_0 = vdims[0]
        if out__0 is None:
            out__0 = np.zeros((vdims_0, 400), dtype=np.float32)

        return (in__input_0, out__0, vdims)

    def set_IO(self, args):
        (in__input_0, out__0, vdims) = self._prepare_run_args(*args)

        with self.profiler("init resources"):
            session = self._session
            in_resource = vaccel.Resource.from_numpy(in__input_0)
            out_resource = vaccel.Resource.from_numpy(out__0)
            vdims_resource = vaccel.Resource.from_numpy(vdims)

        with self.profiler("register resources"):
            in_resource.register(session)
            out_resource.register(session)
            vdims_resource.register(session)

        with self.profiler("init args"):
            session_id = session.remote_id if session.is_remote else session.id
            in_resource_id = (
                in_resource.remote_id if session.is_remote else in_resource.id
            )
            out_resource_id = (
                out_resource.remote_id if session.is_remote else out_resource.id
            )
            vdims_resource_id = (
                vdims_resource.remote_id if session.is_remote else vdims_resource.id
            )
            in_args = [
                vaccel.Arg(session_id, vaccel.ArgType.INT64),
                vaccel.Arg(in_resource_id, vaccel.ArgType.INT64),
                vaccel.Arg(out_resource_id, vaccel.ArgType.INT64),
                vaccel.Arg(vdims_resource_id, vaccel.ArgType.INT64),
            ]

        with self.profiler("exec"):
            self._session.exec_with_resource(
                self._resource,
                "sol_r3d_18_set_IO_unpack",
                in_args,
            )

        self._arg_resources = [in_resource, out_resource, vdims_resource]

    def predict(self, in__input_0=None, out__0=None, vdims=None):
        if self._resource is None:
            self.init()

        (in__input_0, out__0, vdims) = self._prepare_run_args(
            in__input_0,
            out__0,
            vdims,
        )

        with self.profiler("init args"):
            in_args = [
                vaccel.Arg(in__input_0, vaccel.ArgType.BUFFER),
                vaccel.Arg(vdims, vaccel.ArgType.BUFFER),
            ]
            out_args = [vaccel.Arg(out__0, vaccel.ArgType.BUFFER)]

        with self.profiler("exec"):
            out = self._session.exec_with_resource(
                self._resource,
                "sol_predict_unpack",
                in_args,
                out_args,
            )
        return out[0]

    def run(self, in__input_0=None, out__0=None, vdims=None):
        if in__input_0 is not None:
            return self.predict(in__input_0, out__0, vdims)

        if self._resource is None:
            self.init()

        self._session.exec_with_resource(
            self._resource,
            "sol_r3d_18_run_unpack",
        )
        return None

    def get_output(self):
        out_resource = self._arg_resources[1]
        if out_resource is not None:
            with self.profiler("exec"):
                self._session.exec_with_resource(
                    self._resource,
                    "sol_r3d_18_get_output_unpack",
                )

            with self.profiler("sync resource"):
                out_resource.sync(self._session)


def run_example():
    ####### Option 1: Call the deployed lib directly #######
    vdims_0 = 1
    in__input_0 = np.random.rand(vdims_0, 3, 16, 112, 112).astype(np.float32)
    lib = sol_r3d_18()
    (out__0,) = lib(
        in__input_0,
    )
    print(f"Max_V: {np.max(out__0, axis=1)}\nMax_I: {np.argmax(out__0, axis=1)}")

    vdims = np.array(
        [
            vdims_0,
        ],
        dtype=np.int64,
    )

    in__input_0 = np.random.rand(vdims_0, 3, 16, 112, 112).astype(np.float32)
    out__0 = np.zeros((vdims_0, 400), dtype=np.float32)
    dp_args = [
        in__input_0,
        out__0,
        vdims,
    ]  # Inputs, Outputs, VDims must be in this exact order!

    # Call Function and evaluate output---------------------------------------------
    lib = sol_r3d_18()
    lib.init()  # optional, loads parameters on host
    lib.set_seed(271828)  # optional

    ####### Option 2: Run the underlying lib with predefined buffers #######
    lib.run(*dp_args)
    print(f"Max_V: {np.max(out__0, axis=1)}\nMax_I: {np.argmax(out__0, axis=1)}")

    ####### Option 3: Run after setting in- and outputs #######
    lib.set_IO(dp_args)
    lib.optimize(level=2)
    lib.run()  # (async)
    lib.get_output()  # syncs
    print(f"Max_V: {np.max(out__0, axis=1)}\nMax_I: {np.argmax(out__0, axis=1)}")
    # Free used data and close lib---------------------------------------------
    lib.free()
    lib.close()


if __name__ == "__main__":
    run_example()
