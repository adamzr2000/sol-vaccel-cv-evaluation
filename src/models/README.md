## Setup

Run inside the container (`./run_ros.sh gpu` or `./run.sh gpu`) — all scripts below assume `/src/models` as CWD.

---

## 1. Download models

Downloads pretrained weights as `{model}/{model}_state_dict.pt` + TorchScript `.torchscript.pt` where supported.

```bash
for m in resnet50 swin_t swin_s swin_v2_b \
          swin3d_s swin3d_b r3d_18 r2plus1d_18 \
          deeplabv3_resnet50 fcn_resnet50 deeplabv3_resnet101 fcn_resnet101; do
  python3 download_models.py --model "$m"
done
```

---

## 2. Convert to AOTI (.pt2)

Required for `vaccel-local-ptc` and `vaccel-remote-ptc` backends.
Reads `{model}_state_dict.pt`, writes `{model}_cuda.pt2` — a packaged AOTInductor archive with a compiled native `.so` that vAccel's torch plugin loads directly.

Requires `nvcc` (available in the GPU container; the Dockerfile installs `cuda-nvcc-12-8`).

```bash
# All models
python3 convert_aoti_models.py --device cuda
python3 convert_aoti_models.py --device cpu

# Specific model(s)
python3 convert_aoti_models.py --device cuda --model resnet50,swin_t
```
