# SOL 0.9 models

The 12 paper models are deployed with SOL 0.9.1rc2 (NEC image `sol-deploy:rc2`) to match the AOTInductor `.pt2`
models used by vAccel + PyTorch:
- weights: the project's `<model>_state_dict.pt`
- output: the main output only (no segmentation auxiliary head)
- input: batch 1, fixed shape
- precision: TF32 convolutions and FP32 matrix multiplies (`deploy.py`)

## Files
| File | Role |
|---|---|
| `deploy.py` | Runs `sol.deploy` for one model and device. |
| `wrapper.c` | Generic vAccel exec wrapper, compiled once per model and device. |
| `install.sh` | Deploys (or reuses a build with `--from-build`) and installs into `src/models/<model>_sol/lib_{cpu,gpu}`. |
| `../../src/sol09.py` | Runtime: `VaccelRunner` (vaccel-local/remote-sol) and `DirectRunner` (backend `sol`), used by `model_adapter.SolAdapter` / `VaccelSolAdapter` whenever `lib_*/vaccel/manifest.json` exists. Models without one keep the old path. |

```bash
scripts/sol_v09/install.sh all resnet50 swin_t ...                    # deploy + install
scripts/sol_v09/install.sh --from-build DIR all resnet50 swin_t ...   # install existing builds (DIR/<nvidia|x86>/<model>)
```

## Installed layout (`lib_cpu` / `lib_gpu`)
- `<model>.so`, `<model>.sol_weights` and the SOL runtime libraries (`libsol-x86-dnn-*`, `libsleef*`). These are gitignored and must be copied to every host that runs the client (for example the robot).
- `vaccel/`:
  - `libsol_<model>_wrapper.so` (gitignored)
  - `sol_wrapper.c` and `manifest.json` (tracked)

## How a vAccel call works
1. `VaccelRunner` registers two resources. The LIB resource is the wrapper only. The DATA resource is the model library, the weights and the runtime libraries.
2. `sol_init_unpack(DATA id)` opens the model library with `RTLD_LOCAL` from the DATA folder and creates the SOL handle.
3. Each `sol_predict_unpack(in, out)` is one call of the SOL C API:
   - It covers host input to host output. On GPU that includes the copies and the synchronisation.
   - Its vAccel profiling region is `sol_predict`, which `agent_server.py` / `check_agent_profile.py` read.

**CPU worker thread** (`SOL_WORKER_THREAD=1`, set by `run_agent_sol.sh cpu`): in the agent, every CPU inference runs on one persistent worker thread of the wrapper. The agent serves RPCs on several threads, and OpenMP keeps one thread team per calling thread (the agent reached 47 threads on 8 cores). The worker keeps a single, warm team, and it blocks on a condition variable while idle (no spinning). `run_agent_sol.sh` (CPU and GPU) also sets `GLIBC_TUNABLES` so glibc keeps freed memory in the process. SOL (CPU) allocates its output and intermediate buffers per inference, and the vAccel RPC layer allocates the multi-MB request and response buffers per call; with glibc's defaults they are returned to the kernel and page-faulted in again on every call. CPU: ResNet-50 14.0 → 12.0 ms, FCN-R50 68.6 → 56.7, R(2+1)D-18 110.9 → 92.1. GPU: a 4.2 MB segmentation result, seen from the robot, 63.4 → 58.1 ms per call.

**GPU copies and CUDA graph:** on GPU the wrapper copies H2D into a persistent device buffer, runs on device pointers and copies D2H straight into the output buffer, all on its own CUDA stream.
- After 3 plain calls it captures one `sol_predict` into a CUDA graph; a single SOL inference launches up to about 4,000 GPU operations (Swin-T).
- Every later call is H2D copy, graph launch, D2H copy. Outputs are identical, and inference is 5–26% faster (more for Swin).
- It needs GPU libraries deployed with `Conv_NoTextureMemory` (`deploy.py`), the option SOL itself uses for CUDA graphs.
- `SOL_CUDA_GRAPH=0` disables it. If capture fails, the wrapper stays on the plain path.

**vAccel RPC (vaccel-remote-sol):** the zero-copy vAccel RPC client and agent are used by default (`scripts/vaccel_rpc/README.md`); `VACCEL_RPC_CLIENT=stock` / `run_agent_sol.sh --stock-rpc` select the image's.

**Why the model library is not in the LIB resource:** every SOL 0.9 library exports the same C API names. A model loaded globally would take over the calls of the next model served by the same agent. A LIB resource also cannot carry the weights file.

## GPU
- **cuDNN:** the SOL GPU libraries run with cuDNN 9.20 only (9.10 and 9.28 fail with `CUDNN_STATUS_BAD_PARAM`).
  - `install.sh` provides `src/models/runtime/cudnn-9.20` (from `sol-deploy:rc2`) and `src/models/runtime/cudnn-9.28` (pip wheel), both gitignored.
  - SOL (`run_agent_sol.sh gpu`, sol backends in `src/evaluate_models_ros.sh`) uses 9.20; the PyTorch/AOTInductor agent (`run_agent_ptc.sh gpu`) and the other GPU backends use 9.28.
  - A process that still has an older cuDNN gets a clear error from `src/sol09.py`.
- **GELU:** SOL fuses Linear+GELU with the tanh approximation on GPU (Swin models only; about 2e-3 at the output, same top-1). This is a known SOL 0.9.1rc2 deviation.

## Old libraries
The SOL rc4/rc5 libraries are backed up on edge-asus in `/home/nextnet/adam/backup/sol-models-rc5-2026-11-25`.
