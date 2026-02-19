# SOL-vAccel integration and evaluation in 5TONIC - Publication

## Build docker images
Build the container image according to the target device:

```bash
sudo docker build -f dockerfile-torchvision-cpu . -t torchvision-app:cpu
sudo docker build -f dockerfile-torchvision-gpu . -t torchvision-app:gpu

# vAccel agent on the GPU host
sudo docker build -f dockerfile-torchvision-gpu-agent . -t torchvision-app:agent
```

## Download models

Models are provided as `models.tar.gz`, which extracts into a `models/` directory.

1. Download `models.tar.gz` from the link below.
2. Copy `models.tar.gz` into `src/`.
3. Replace `src/models/` with the extracted directory:
   ```bash
   cd src
   rm -rf models
   tar -xzf models.tar.gz
   ```
   
- [Download link](https://drive.google.com/file/d/1pte0m_6HKMS40C-olHXxqNc0_EAQssZU/view?usp=sharing)
- [Download link (rc4)](https://drive.google.com/file/d/1z1Av5hBn2I1S4mdNuMO88p3VpbLRTtrR/view?usp=sharing)

### Generate vAccel wrappers

To generate vAccel wrappers for the supported models from a directory of stock
SOL models, launch the gpu container:

```bash
./run.sh gpu
```

and run the provided script:

```bash
python3 /scripts/sol_vaccel_wrappers/gen_sol_vaccel_wrappers.py models
```

## Run model benchmark

### 1. Start the container

Start the container in CPU or GPU mode:

```bash
./run.sh cpu
# or
./run.sh gpu
```

#### vAccel remote setup

To use the vAccel remote backend, first start the agent container on the remote
GPU host:

```bash
./run_agent.sh 9125
# or
./run_agent.sh 9125 --debug
```

where `9125` is the port to use. If the port is omitted, `9125` will be used by
default.

Consequently, specify the remote agent's address on the (local) host when
starting the main container:

```bash
./run.sh cpu 10.5.1.20:9125
# or
./run.sh gpu 10.5.1.20:9125
```

### 2. Run model benchmark script

Execute `model_benchmark.py` with the following environment variables.

- **DEVICE**: execution device  
  - `cpu` or `gpu`
  - If using `cpu` on ASUS G815 laptop, set `OMP_NUM_THREADS=10`

- **HOST**: execution host identifier  
  - Default: `edge-asus`

- **MODEL**: neural network model to benchmark  
  - **Segmentation**:  
    `deeplabv3_resnet50`, `deeplabv3_resnet101`, `fcn_resnet50`, `fcn_resnet101`,
    `deeplabv3_mobilenet_v3_large`
  - **Image classification**:  
    `resnet50`, `swin_t`, `swin_s`, `swin_v2_b`, `swin_s`,
    `mobilenet_v3_large`
  - **Video classification**:  
    `swin3d_t`, `swin3d_s`, `swin3d_b`, `mc3_18`, `r3d_18`, `r2dplus1d_18` 
  - **Object detection**:  
    `yolov5s`

- **BACKEND**: inference backend  
  - `stock` (default), `ptc`, `sol`, `vaccel-local-sol`, `vaccel-remote-sol`

- **ENABLE_VACCEL_PROFILER**: enable vAccel execution profiling *(vAccel SOL models only)*  
  - `true` or `false` (default: `false`)

- **SOL_RUN_MODE**: SOL execution mode *(SOL models only)*  
  - `2` → **Option 2 (default)**: explicit buffers per call  
    - Uses `run(*args)` (explicit input/output buffers each iteration)  
    - By default, no `set_IO` is required  
    - *(GPU only)* may still apply `set_IO(...)` + `optimize(2)` for supported models to improve performance  
  - `3` → **Option 3**: bound buffers + optimized run  
    - Calls `set_IO(...)` once  
    - Uses `run()` (no args) + `get_output()` / `sync()`  
    - Enables GPU optimizations (`optimize(2)` on GPU)  
  - Default: `2`

- **NUM_IMAGES**: number of images to benchmark  
  - Uses the first *N* images (sorted) from `data/images`  
  - Default: `64`

- **NUM_VIDEOS**: number of videos to benchmark (video classification models only)  
  - Uses the first *N* videos (sorted) from `data/videos`  
  - Default: `10`

- **EXPORT_RESULTS**: save benchmark results  
  - Outputs `benchmark_data.csv` and `benchmark_summary.json`  
  - `true` or `false` (default: `false`)

- **RUN_TAG**: experiment run prefix identifier *(optional)*  
  - Used to tag the output directory under  
    `/results/experiments/model-stats`  
  - The final directory name is always auto-generated as:  
    `<host>/<RUN_TAG>_<model>_<backend>_<host>_<device>`
  - If not set, the current `<timestamp>` is used as the prefix.

- **EXPORT_OUTPUT_IMAGES**: save output images  
  - Requires `EXPORT_RESULTS=true`  
  - `true` or `false` (default: `false`)

Examples:

```shell
# Pytorch image classification (CPU)
DEVICE=cpu BACKEND=stock MODEL=resnet50 OMP_NUM_THREADS=10 python3 model_benchmark.py
# Pytorch image classification (GPU)
DEVICE=gpu BACKEND=stock MODEL=resnet50 python3 model_benchmark.py
# SOL image classification (GPU)
DEVICE=gpu BACKEND=sol MODEL=resnet50 python3 model_benchmark.py
# vAccel remote + SOL image classification (GPU)
DEVICE=gpu BACKEND=vaccel-remote-sol MODEL=resnet50 python3 model_benchmark.py
```

> Note: Results and images are not saved by default to avoid unnecessary disk usage. Set `EXPORT_RESULTS=1` to save benchmark metrics, and also set `EXPORT_OUTPUT_IMAGES=1` to store output images in the [results/experiments/model-stats](./results/experiments/model-stats/) directory.

---

## Run model benchmark with resource consumption

### 1. Start the monitoring service

See [monitoring](./monitoring)

Start the `docker-stats-collector` and `system-stats-collector` containers:

```bash
./run_monitoring.sh edge
# or
./run_monitoring.sh robot
```

### 2. Start the container

Start the container in CPU or GPU mode:

```bash
./run.sh cpu
# or
./run.sh gpu
```

### 2. Run benchmark script

Execute `model_benchmark.py` with the following environment variables.

Examples:
```shell
RESOURCE_MONITORING=1 EXPORT_RESULTS=1 NUM_IMAGES=32 BACKEND=stock HOST=robot DEVICE=cpu MODEL=resnet50 RUN_TAG=run1 DOCKER_STATS_ENDPOINT=http://192.168.2.2:6000 SYSTEM_STATS_ENDPOINT=http://192.168.2.2:6001 python3 model_benchmark_resources.py

RESOURCE_MONITORING=1 EXPORT_RESULTS=1 NUM_IMAGES=32 BACKEND=vaccel-remote-sol HOST=robot DEVICE=gpu MODEL=resnet50 RUN_TAG=run1 DOCKER_STATS_ENDPOINT=http://192.168.2.2:6000 SYSTEM_STATS_ENDPOINT=http://192.168.2.2:6001 DOCKER_STATS_REMOTE_ENDPOINT=http://10.5.1.20:6000 SYSTEM_STATS_REMOTE_ENDPOINT=http://10.5.1.20:6001 python3 model_benchmark_resources.py
```

Auto:
```shell
./evaluate_models.sh --backend stock --host edge-asus --device cpu --run-tag run1 --sleep 10
./evaluate_models.sh --backend vaccel-local-sol --host edge-asus --device cpu --run-tag run1 --sleep 10
```

### GPU laptop specs
```shell
nextnet@asus-g815:~$ nvidia-smi
Mon Feb 16 17:48:56 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.95.05              Driver Version: 580.95.05      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5080 ...    Off |   00000000:02:00.0 Off |                  N/A |
| N/A   41C    P8              9W /   80W |     181MiB /  16303MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            2433      G   /usr/lib/xorg/Xorg                      145MiB |
|    0   N/A  N/A            2765      G   /usr/bin/gnome-shell                     14MiB |
+-----------------------------------------------------------------------------------------+
nextnet@asus-g815:~$ uname -a
Linux asus-g815 6.14.0-37-generic #37~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Thu Nov 20 10:25:38 UTC 2 x86_64 x86_64 x86_64 GNU/Linux
```

### Model I/O quick reference

Quick reminder of **input/output shapes and their sizes** (based on [model_adapter.py](./src/model_adapter.py), assuming **float32 = 4 B**, **uint8 = 1 B**).

#### Object Detection — `yolov5s`

* **Input to model**
  Shape: `(1, 3, 640, 640)` (float32)
  **Size:** **≈ 4.69 MiB**

* **Raw model output (predictions)** 
  Shape: `(1, 25200, 85)` (float32)
  **Size:** **≈ 8.17 MiB**

* **Postprocessed output (returned by app)**
  Dictionary containing `N` detected objects: `boxes` (float32), `scores` (float32), and `classes` (int64).
  **Size:** **≈ 28 B per detected object (negligible)**

---

#### Segmentation (2D) — `deeplabv3_resnet50`, `fcn_resnet50`

* **Input to model**
  Shape: `(1, 3, 224, 224)` (float32)
  **Size:** **≈ 588 KiB**

* **Raw model output (logits)**
  Shape: `(1, 21, 224, 224)` (float32)
  **Size:** **≈ 4.02 MiB**

* **SOL extra buffer (aux output, segmentation only)**
  Shape: `(1, 21, 224, 224)` (float32)
  **Size:** **≈ 4.02 MiB**

* **Postprocessed output (returned by app)**
  Shape: `(224, 224)` class IDs (uint8)
  **Size:** **≈ 49 KiB**

---

#### Image classification — `resnet50`, `mobilenet_v3_large`, `swin_t`

* **Input to model**
  Shape: `(1, 3, 224, 224)` (float32)
  **Size:** **≈ 602 KB**

* **Raw model output (logits)**
  Shape: `(1, 1000)` (float32)
  **Size:** **≈ 4 KB**

* **Postprocessed output (returned by app)**
  `top_class` (int64) + `top_prob` (float32)
  **Size:** **≈ 12 B (negligible)**

---

#### Video classification — `mc3_18`, `r3d_18`

* **Input to model**
  Shape: `(1, 3, 16, 112, 112)` (float32)
  **Size:** **≈ 2.40 MB**

* **Raw model output (logits)**
  Shape: `(1, 400)` (float32)
  **Size:** **≈ 1.6 KB**

* **Postprocessed output (returned by app)**
  `top_class` (int64) + `top_prob` (float32)
  **Size:** **≈ 12 B (negligible)**

---

## Required fix for `sol_mobilenet_v3_large` (SOL rc4 + cuDNN Graph)

The GPU deployment of **`sol_mobilenet_v3_large`** included in this repository
(`libsol-dnn-cudnn-deployment-0.8.0rc4-9.1.so`) is built against **cuDNN Graph 9.1.x**.

If the environment installs **PyTorch nightly / cu128**, it pulls **cuDNN 9.10.x** by default.
This causes `sol_mobilenet_v3_large.py` to fail at runtime with:

`CUDNN_STATUS_BAD_PARAM` (from `api_v9_graph.cpp`)

To run `sol_mobilenet_v3_large` correctly, **cuDNN must be downgraded to 9.1.1.17**:

```shell
# On torchvision-app:gpu container
python3 -m pip install --no-cache-dir --force-reinstall \
  "nvidia-cudnn-cu12==9.1.1.17" --no-deps
```

---

## ROS Integration

### Build docker images
Build the container image according to the target device:

```bash
sudo docker build -f dockerfile-torchvision-ros-cpu . -t torchvision-ros-app:cpu
sudo docker build -f dockerfile-torchvision-ros-gpu . -t torchvision-ros-app:gpu
```

### Start the container in CPU or GPU mode:

```bash
./run_ros.sh cpu
# or
./run_ros.sh gpu
```

```bash
EXPERIMENT_DURATION_SEC=30 HOST=edge-asus BACKEND=stock DEVICE=cpu MODEL=resnet50 INPUT_TOPIC=/camera/color/image_raw python3 model_benchmark_ros.py
```