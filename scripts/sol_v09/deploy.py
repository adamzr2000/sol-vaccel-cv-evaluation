#!/usr/bin/env python3
"""Redeploy one paper model with SOL 0.9.1rc2 so that it matches the AOTInductor export as closely as possible.

Same as src/models/convert_aoti_models.py (the .pt2 used by vAccel + PyTorch):
  * weights: the project's {model}_state_dict.pt (not torchvision's download, not random)
  * graph:   the model wrapped to return only the main output (segmentation: out["out"], so the auxiliary head
             is dead code, as in the AOTI export)
  * input:   batch 1, (1,3,224,224) images / (1,3,16,112,112) video clips, float32
  * precision: the torch settings AOTInductor runs with (libtorch defaults): cuDNN convolutions may use TF32,
             FP32 matmuls stay IEEE FP32 (float32_matmul_precision "highest"); cudnn.benchmark off.
             SOL 0.9 reads these flags while parsing the model (sol/pytorch/parser_utils.py), so they are set
             before sol.deploy, as NEC advised.
  * GEMM_FP16 / GEMM_BF16 off (see main), Swin masks rewritten without slice assignment (patch_swin_masks).
  * GPU: Conv_NoTextureMemory (needed by the CUDA-graph replay in wrapper.c)
Plus NEC's workaround from deploy_swin_s_unpatched.py (drop CONSTANT_TENSOR input specs from the exported graph).

Usage (inside sol-deploy:rc2, src/models mounted at /models; see install.sh):
    python3 deploy.py <model> <nvidia|x86> <out_dir>
Writes <out_dir>/<model>.{so,sol_weights,h,...}. Correctness is checked by check.py.
"""
import sys
from pathlib import Path

import torch
import torchvision

MODELS = Path("/models")   # sol-vaccel-cv-evaluation/src/models, mounted read-only
VIDEO = {"r3d_18", "r2plus1d_18", "swin3d_s", "swin3d_b"}
SEG = ("fcn_", "deeplabv3_")


class OutOnly(torch.nn.Module):
    """As model_adapter._OutOnly / the AOTI export: return the main output tensor only."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        out = self.m(x)
        if isinstance(out, dict):
            return out["out"]
        if isinstance(out, (list, tuple)):
            return out[0]
        return out


def _region_fill(t: torch.Tensor, slices, value) -> torch.Tensor:
    """Functional `t[s0, s1, ...] = value` (python slice semantics, one (start, stop) pair per leading axis)."""
    cond = None
    for axis, (a, b) in enumerate(slices):
        n = t.shape[axis]
        start, stop, _ = slice(a, b).indices(n)
        idx = torch.arange(n, device=t.device)
        shape = [1] * t.dim()
        shape[axis] = n
        c = ((idx >= start) & (idx < stop)).view(shape)
        cond = c if cond is None else cond & c
    return torch.where(cond, torch.full_like(t, value), t)


def patch_swin_masks() -> None:
    """SOL 0.9.1rc2 drops in-place writes into tensor slices (`a[i:j, k:l] = v` leaves `a` unchanged), so the
    shifted-window attention mask of torchvision's Swin / SwinV2 / Swin3D comes out all zero. Rewrite only that
    assignment as the equivalent out-of-place _region_fill (same mask, same weights, same everything else)."""
    import inspect
    import textwrap
    import torchvision.models.swin_transformer as s2
    import torchvision.models.video.swin_transformer as s3
    for mod, fn, old, new in [
        (s2, "shifted_window_attention", "attn_mask[h[0] : h[1], w[0] : w[1]] = count",
         "attn_mask = _region_fill(attn_mask, (h, w), count)"),
        (s3, "_compute_attention_mask_3d", "attn_mask[d[0] : d[1], h[0] : h[1], w[0] : w[1]] = count",
         "attn_mask = _region_fill(attn_mask, (d, h, w), count)"),
    ]:
        src = textwrap.dedent(inspect.getsource(getattr(mod, fn)))
        assert src.count(old) == 1, f"{fn}: torchvision source changed"
        src = "\n".join(l for l in src.replace(old, new).splitlines() if not l.startswith("@"))
        mod.__dict__["_region_fill"] = _region_fill
        exec(compile(src, f"<patched {fn}>", "exec"), mod.__dict__)


def build(name: str) -> torch.nn.Module:
    sd = torch.load(MODELS / name / f"{name}_state_dict.pt", map_location="cpu")
    kw = {"weights": None}
    if name.startswith(SEG):
        kw.update(weights_backbone=None, aux_loss=any(k.startswith("aux_classifier.") for k in sd))
    m = torchvision.models.get_model(name, **kw)
    m.load_state_dict(sd, strict=True)
    return OutOnly(m).eval()


def main():
    name, device, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)

    # precision = what AOTInductor runs with (libtorch defaults), read by SOL at parse time
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False   # (True was tested: no gain)
    torch.backends.cudnn.deterministic = False
    # PyTorch's defaults (True) only concern FP16/BF16 GEMMs, so they never touch AOTI's FP32 GEMMs; SOL turns
    # them into GEMM_FP16 / GEMM_BF16 permissions for the whole model, so switch them off to stay at FP32.
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    if device == "nvidia":
        # without an existing CUDA context, SOL 0.9.1rc2 fails with CUDA_ERROR_INVALID_CONTEXT once the reduced-
        # precision GEMMs above are disabled
        torch.zeros(1, device="cuda")
        torch.cuda.synchronize()

    import sol   # SOL options at their defaults (autotuning on; a 7x larger autotuning budget was tested: no gain)
    import sol.pytorch.parser_fx as pf
    from torch.export.graph_signature import InputKind, InputSpec

    # NEC workaround (deploy_swin_s_unpatched.py): skip constant-tensor inputs of the exported graph
    _orig = pf._exported_specs

    def _exported_specs(gs, cs, lst):
        keep = [(s, a) for s, a in zip(gs, lst)
                if not (isinstance(s, InputSpec) and s.kind == InputKind.CONSTANT_TENSOR)]
        return _orig([s for s, _ in keep], cs, [a for _, a in keep])
    pf._exported_specs = _exported_specs

    if name.startswith("swin"):
        patch_swin_masks()
    sol.set_seed(0)
    model = build(name)
    shape = (1, 3, 16, 112, 112) if name in VIDEO else (1, 3, 224, 224)
    x = torch.zeros(shape, dtype=torch.float32)
    with torch.no_grad():
        det = None
        if device == "nvidia":
            # CUDA-graph compatible convolutions (no texture memory), the option SOL itself adds for CUDA graphs: the
            # vAccel wrapper replays each inference as a CUDA graph (wrapper.c); same results
            from sol.pytorch.parser_fx import Determinism
            det = sol.pytorch.determinism() | {Determinism.Conv_NoTextureMemory}
        sol.deploy(model, [x], device=device, name=name, path=str(out), determinism=det)
    print(f"[deploy] {name} {device} -> {out}")


if __name__ == "__main__":
    main()
