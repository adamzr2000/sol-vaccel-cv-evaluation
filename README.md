# SOL-vAccel CV Evaluation

## Build

```bash
# Non-ROS
sudo docker build -f dockerfile-torchvision-cpu . -t torchvision-app:cpu
sudo docker build -f dockerfile-torchvision-gpu . -t torchvision-app:gpu

# vAccel remote agent (GPU host)
sudo docker build -f dockerfile-torchvision-gpu-agent . -t torchvision-app:agent

# ROS
sudo docker build -f dockerfile-torchvision-ros-cpu . -t torchvision-ros-app:cpu
sudo docker build -f dockerfile-torchvision-ros-gpu . -t torchvision-ros-app:gpu
```

### Generate vAccel SOL wrappers

```bash
./run.sh gpu
python3 /scripts/sol_vaccel_wrappers/gen_sol_vaccel_wrappers.py models
```

---

## ROS Benchmark

All commands below run **inside the container** from `/src`. Start the container with `./run_ros.sh` before running experiments.

### Setup

#### Monitoring services (on each host, before experiments that capture resource stats)

```bash
./run_monitoring.sh edge    # docker-stats-collector + system-stats-collector on edge
./run_monitoring.sh robot   # same on robot
```

See [monitoring/](./monitoring) for details.

#### Benchmark container

```bash
# Edge
./run_ros.sh cpu --iface enp130s0
./run_ros.sh gpu --iface enp130s0

# Robot (always CPU)
./run_ros.sh cpu --iface ue0
```

For `vaccel-remote-*` backends, also start the vAccel agent on the edge host and point the robot container at it:

```bash
# On edge host
./run_agent.sh 9125

# On robot (--remote defaults to 10.5.1.20:9125 = edge-asus)
./run_ros.sh cpu --iface ue0                              # → edge-asus
./run_ros.sh cpu --remote 10.5.1.21:9125 --iface ue0      # → edge-xtreme
```

---

### Phase 1 — Edge isolation (run on each edge host)

**Goal:** measure pure inference latency per backend on edge hardware, without RPC overhead. Monitoring is disabled; only `benchmark_summary.json` is saved for later latency breakdown analysis.

Run on **edge-asus** (repeat with `--host edge-xtreme` on that machine):

```bash
./evaluate_models_ros.sh --host edge-asus --device cpu --backend vaccel-local-sol --run-tag iso --no-monitoring
./evaluate_models_ros.sh --host edge-asus --device gpu --backend vaccel-local-sol --run-tag iso --no-monitoring
./evaluate_models_ros.sh --host edge-asus --device cpu --backend vaccel-local-ptc --run-tag iso --no-monitoring
./evaluate_models_ros.sh --host edge-asus --device gpu --backend vaccel-local-ptc --run-tag iso --no-monitoring
```

---

### Phase 2 — Robot local (run on robot)

**Goal:** measure robot-side inference latency for vaccel-local backends (CPU only). Captures robot resource usage.

```bash
./evaluate_models_ros.sh --host robot --device cpu --backend vaccel-local-sol --run-tag run1
./evaluate_models_ros.sh --host robot --device cpu --backend vaccel-local-ptc --run-tag run1
```

---

### Phase 3 — Robot remote (run on robot)

**Goal:** measure end-to-end latency (robot preprocessing + RPC transport + edge inference + return) with full resource monitoring on both robot and edge simultaneously.

Requires: monitoring running on both hosts, agent running on edge, robot container started with correct `--remote` address.

```bash
# → edge-asus GPU
./evaluate_models_ros.sh --host robot --device gpu --backend vaccel-remote-sol --run-tag run1 --remote-host edge-asus
./evaluate_models_ros.sh --host robot --device gpu --backend vaccel-remote-ptc --run-tag run1 --remote-host edge-asus

# → edge-xtreme GPU  (container started with --remote 10.5.1.21:9125)
./evaluate_models_ros.sh --host robot --device gpu --backend vaccel-remote-sol --run-tag run1 --remote-host edge-xtreme
./evaluate_models_ros.sh --host robot --device gpu --backend vaccel-remote-ptc --run-tag run1 --remote-host edge-xtreme
```

---

### Results layout

```
results/experiments/
  model-stats/{host}/{run_id}/benchmark_summary.json   ← latency + FPS stats
  docker-stats/{host}/torchvision-app_{run_id}.csv     ← container CPU/mem
  system-stats/{host}/{run_id}.csv                     ← host CPU or GPU stats
  system-stats/{host}/{run_id}_net.csv                 ← network usage
```

For remote runs (Phase 3), edge-side stats are captured under `docker-stats/{remote-host}/` and `system-stats/{remote-host}/` simultaneously.

`run_id` format:
- Local: `{tag}_{model}_{backend}_{device}` — e.g. `iso_resnet50_vaccel-local-sol_gpu`
- Remote (robot side): `{tag}_{model}_{backend}_{local-device}_target-{remote-host}-{device}` — e.g. `run1_resnet50_vaccel-remote-sol_cpu_target-edge-asus-gpu`
- Remote (edge side): `{tag}_{model}_{backend}_{device}` — e.g. `run1_resnet50_vaccel-remote-sol_gpu`

**Other options:**
```bash
--model resnet50,swin_t    # run a subset of models
--no-export                # disable both export and monitoring (dry run)
--sleep N                  # seconds between model runs (default: 20)
```

---

## Backend reference

| Backend | Adapter | Compilation | Inference path |
|---|---|---|---|
| `stock` | `PyTorchBaselineAdapter` | None — eager PyTorch | Direct Python call |
| `ptc` | `PyTorchBaselineAdapter` | JIT at runtime via `torch.compile` (Inductor) | Direct Python call to compiled graph |
| `sol` | `SolAdapter` | Offline (SOL compiler) | SOL C library via dlopen |
| `vaccel-local-torch` | `VaccelPyTorchAdapter` | None — loads TorchScript `.torchscript` | vAccel TORCH plugin (local) |
| `vaccel-local-ptc` | `VaccelPyTorchAdapter` | Offline AOTI — loads `.pt2` | vAccel TORCH plugin (local) |
| `vaccel-remote-torch` | `VaccelPyTorchAdapter` | None — loads TorchScript `.torchscript` | vAccel TORCH + RPC → remote agent |
| `vaccel-remote-ptc` | `VaccelPyTorchAdapter` | Offline AOTI — loads `.pt2` | vAccel TORCH + RPC → remote agent |
| `vaccel-local-sol` | `VaccelSolAdapter` | Offline (SOL compiler) | vAccel EXEC plugin (local) |
| `vaccel-remote-sol` | `VaccelSolAdapter` | Offline (SOL compiler) | vAccel EXEC + RPC → remote agent |

**Notes:**
- `ptc` and `vaccel-local/remote-ptc` both use Inductor under the hood, but differ in when compilation happens (JIT vs offline AOT) and the dispatch path. `ptc` sets `cudnn.benchmark=True` and `float32_matmul_precision=high`; `vaccel-ptc` does not (optimizations are baked into the `.pt2` at compile time). Each `vaccel-*` inference call also pays a `Tensor.from_torch` / `as_torch` conversion cost that bare `ptc` does not.
- `stock` vs `ptc` isolates the cost of Inductor compilation/optimization.
- `vaccel-local-*` vs `vaccel-remote-*` isolates RPC transport overhead.
- `ptc` vs `vaccel-local-ptc` is **not a fair apples-to-apples comparison** due to the above differences; prefer comparing within the same dispatch family.

---

## Model I/O reference

Input/output shapes and sizes (float32 = 4 B, uint8 = 1 B). See [model_adapter.py](./src/model_adapter.py).

### Object Detection — `yolov5s`

| | Shape | Size |
|---|---|---|
| Input | `(1, 3, 640, 640)` float32 | ≈ 4.69 MiB |
| Raw output | `(1, 25200, 85)` float32 | ≈ 8.17 MiB |
| Postprocessed | `N` objects: boxes + scores + classes | ≈ 28 B/object |

### Segmentation — `deeplabv3_resnet50/101`, `fcn_resnet50/101`

| | Shape | Size |
|---|---|---|
| Input | `(1, 3, 224, 224)` float32 | ≈ 588 KiB |
| Raw output | `(1, 21, 224, 224)` float32 | ≈ 4.02 MiB |
| SOL aux buffer | `(1, 21, 224, 224)` float32 | ≈ 4.02 MiB |
| Postprocessed | `(224, 224)` uint8 class IDs | ≈ 49 KiB |

### Image Classification — `resnet50`, `swin_t`, `swin_s`, `swin_v2_b`

| | Shape | Size |
|---|---|---|
| Input | `(1, 3, 224, 224)` float32 | ≈ 602 KB |
| Raw output | `(1, 1000)` float32 | ≈ 4 KB |
| Postprocessed | top class (int64) + prob (float32) | ≈ 12 B |

### Video Classification — `mc3_18`, `r3d_18`, `r2plus1d_18`, `swin3d_s/b`

| | Shape | Size |
|---|---|---|
| Input | `(1, 3, 16, 112, 112)` float32 | ≈ 2.40 MB |
| Raw output | `(1, 400)` float32 | ≈ 1.6 KB |
| Postprocessed | top class (int64) + prob (float32) | ≈ 12 B |

---

## E2E inference workflow

### `vaccel-local-*` (in-process)

```
Python benchmark (ROS node)
  └─ model_adapter.py
       ├─ load:  vaccel.Resource(model_path) → session.torch_model_load()
       │          reads .pt2 / SOL .so from /src/models/ and loads into local plugin
       └─ infer: session.torch_model_run([input_tensor])
                  └─ libvaccel-exec.so → vaccel-torch / SOL plugin → result
```

Inference runs in the **same process** as the ROS node. OMP workers compete with ROS/DDS threads at parallel-region barriers → variance for AOTI (vaccel-local-ptc); use OMP=1 for stable latency.

### `vaccel-remote-*` (RPC to agent)

```
Python benchmark (ROS node)              vAccel agent container
  └─ model_adapter.py                       (harbor.nbfc.io/desire6g/torchvision-vaccel)
       ├─ load:  session.torch_model_load()  ──► agent receives model bytes over RPC,
       │          serialises .pt2 / SOL .so       loads into vaccel-torch / SOL plugin
       │          and ships over TCP                (no volume mount needed)
       └─ infer: session.torch_model_run()  ──► agent runs inference, returns output
                  └─ libvaccel-rpc.so ──TCP──►  libvaccel-exec.so → plugin → result
```

Model bytes are transferred **once per session** at load time (`VACCEL_RPC_SEND_WRITE_ENABLED=0` uses the standard genop transfer path). Per-inference cost is input tensor serialisation + network RTT + output deserialisation. The agent process has no competing ROS threads → OMP workers run uncontended → stable latency at full thread count.

**Loopback** (`--host edge-asus --backend vaccel-remote-*`): agent runs on the same machine; TCP loopback RTT is negligible but process isolation eliminates the OMP barrier-stall problem of the local variant.

---

## Known issue: `sol_mobilenet_v3_large` + cuDNN mismatch

Segmentation models ship `libsol-dnn-cudnn-deployment-0.8.0rc6-9.10.2.so` (needs cuDNN **9.10.x**); classification models ship `rc5-9.1` (needs cuDNN **9.1.x**, but 9.10.x is backward-compatible). Use cuDNN 9.10.x (the default from `nvidia-cudnn-cu12` without a version pin) for both the GPU container and the agent image.
