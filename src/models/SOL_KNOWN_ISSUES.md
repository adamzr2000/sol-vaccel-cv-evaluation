# SOL models: known issues (checked 2026-11-20)

The SOL libraries in `*_sol/lib_{cpu,gpu}` are used as they are; none were changed.
Reference for each check: eager PyTorch with the project's `*_state_dict.pt` (the same weights as the
AOTInductor `.pt2` used by vAccel + PyTorch). AOTInductor matched that reference for all 12 paper models.

## Problems

| Model(s) | Problem | Effect |
|---|---|---|
| swin_t, swin_s, swin_v2_b, swin3d_s, swin3d_b | **Wrong outputs on GPU.** 0% top-1 on ImageNet samples. swin_t is also non-deterministic (same input, different output). | SOL Swin GPU results are not valid. |
| swin_t, swin_s, swin3d_s, swin3d_b | **Wrong outputs on CPU** (swin_v2_b on CPU is correct). | SOL Swin CPU results are not valid. |
| resnet50 | SOL has the **ImageNet V1** weights; PyTorch/AOTInductor uses **V2**. | Same architecture, so timing is comparable, but they are not the same trained model (top-1 ~71% vs ~74% on our 100 samples). |
| resnet50 | SOL (rc5 build) runs **strict FP32** on GPU; AOTInductor uses **TF32** convolutions (libtorch default). | Disadvantages SOL on GPU. |
| fcn_*, deeplabv3_* | SOL still **computes the auxiliary head** (the output is just not sent back); the AOTInductor export returns only `out`, so it skips it. | SOL does extra work: at most ~0.8 ms on GPU and 1–3 ms on CPU (deeplabv3_resnet50). |

**Correct with SOL:** r3d_18, r2plus1d_18, mc3_18 and the 4 segmentation models (same precision level as AOTInductor).

## What a rebuild can and cannot fix (tested in a scratch folder only)
- **Swin CPU:** correct when deployed **without** `torch.jit.trace`, but slower (swin_t 21 vs 14.6 ms).
- **Swin GPU:** wrong in every variant tried (rc5, rc6, with/without trace, with/without autotune, and SOL's own JIT `sol.optimize`). This is a SOL NVIDIA-backend bug, so it needs a fix from NEC.
- **resnet50 rc6 + V2 weights:** correct and faster when called directly (−19% GPU, −8% CPU), but it **crashes inside the vAccel agent** at `optimize(2)` (`CUDNN_STATUS_INTERNAL_ERROR_TEXTURE_CREATION_FAILED`). Not usable yet.

## How these were checked
Scripts in `scripts/` (hard-coded paths, run on edge-asus):
- `parity_all.py`: SOL GPU and AOTInductor vs eager PyTorch
- `parity_cpu.py`: SOL CPU vs eager PyTorch
- `acc_container.py`: ImageNet top-1
- `prec2.py`: FP32 vs TF32 error level
