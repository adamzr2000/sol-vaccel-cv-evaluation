#!/usr/bin/env python3
"""
convert_aoti_models.py

Converts downloaded state-dict models to AOTI (.pt2) format required by
the vaccel-local-ptc and vaccel-remote-ptc backends.

Uses torch._inductor.aot_compile() to produce a true AOTInductor .pt2 archive
(compiled native .so packaged at data/aotinductor/), which is the format
vAccel's torch plugin expects. torch.export.save() produces a different
ExportedProgram format and must NOT be used here.

Reads:  {model_dir}/{model_name}_state_dict.pt
Writes: {model_dir}/{model_name}_{device}.pt2

Must be run inside the GPU container (torchvision-ros-app:gpu or torchvision-app:gpu)
so that CUDA is available for cuda exports.

Usage:
  python3 convert_aoti_models.py --device cuda
  python3 convert_aoti_models.py --device cpu
  python3 convert_aoti_models.py --device cuda --model resnet50
  python3 convert_aoti_models.py --device cuda --model resnet50,swin_t,fcn_resnet50
"""

import argparse
import sys
import torch
import torch._inductor
import torch.export
import torchvision.models as models
from pathlib import Path

# ==========================================
# MODEL REGISTRY (must match model_adapter.py)
# ==========================================
MODEL_REGISTRY = {
    # Segmentation
    "deeplabv3_resnet50":  (models.segmentation.deeplabv3_resnet50,  "segmentation"),
    "deeplabv3_resnet101": (models.segmentation.deeplabv3_resnet101, "segmentation"),
    "fcn_resnet50":        (models.segmentation.fcn_resnet50,        "segmentation"),
    "fcn_resnet101":       (models.segmentation.fcn_resnet101,       "segmentation"),
    # Classification
    "resnet50":  (models.resnet50,  "classification"),
    "swin_t":    (models.swin_t,    "classification"),
    "swin_s":    (models.swin_s,    "classification"),
    "swin_v2_b": (models.swin_v2_b, "classification"),
    # Video classification
    "mc3_18":      (models.video.mc3_18,      "video_classification"),
    "r3d_18":      (models.video.r3d_18,      "video_classification"),
    "r2plus1d_18": (models.video.r2plus1d_18, "video_classification"),
    "swin3d_s":    (models.video.swin3d_s,    "video_classification"),
    "swin3d_b":    (models.video.swin3d_b,    "video_classification"),
}

INPUT_SHAPES = {
    "classification":       (1, 3, 224, 224),
    "segmentation":         (1, 3, 224, 224),
    "video_classification": (1, 3, 16, 112, 112),
}


class _OutOnly(torch.nn.Module):
    """Unwrap dict/tuple outputs to a single tensor — required for aot_compile."""
    def __init__(self, m: torch.nn.Module, model_type: str):
        super().__init__()
        self.m = m
        self.model_type = model_type

    def forward(self, x):
        out = self.m(x)
        if self.model_type == "segmentation" and isinstance(out, dict):
            return out["out"]
        if isinstance(out, (list, tuple)):
            return out[0]
        return out


def _has_aux_loss(state_dict: dict) -> bool:
    return any(k.startswith("aux_classifier.") for k in state_dict)


def convert_one(model_name: str, device: str, models_dir: Path) -> bool:
    builder, model_type = MODEL_REGISTRY[model_name]
    torch_device = torch.device(device)
    out_path = models_dir / model_name / f"{model_name}_{device}.pt2"

    if out_path.exists():
        print(f"  [skip] {out_path.name} already exists")
        return True

    sd_path = models_dir / model_name / f"{model_name}_state_dict.pt"
    if not sd_path.exists():
        print(f"  ❌ state dict not found: {sd_path}")
        print(f"     Run download_baseline_model.py --model {model_name} first.")
        return False

    print(f"  Loading state dict from {sd_path} ...")
    state_dict = torch.load(sd_path, map_location="cpu", weights_only=True)

    if model_type == "segmentation":
        aux_loss = _has_aux_loss(state_dict)
        model = builder(weights=None, weights_backbone=None, aux_loss=aux_loss)
    else:
        model = builder(weights=None)

    model.load_state_dict(state_dict, strict=True)
    model = model.to(torch_device).eval()
    wrapped = _OutOnly(model, model_type)

    shape = INPUT_SHAPES[model_type]
    example_input = torch.zeros(shape, dtype=torch.float32, device=torch_device)

    print(f"  Exporting {model_name} ({model_type}, {device}) ...")
    with torch.no_grad():
        try:
            ep = torch.export.export(wrapped, (example_input,))
        except Exception as e:
            print(f"  [retry] strict export failed ({e}), retrying with strict=False ...")
            try:
                ep = torch.export.export(wrapped, (example_input,), strict=False)
            except Exception as e2:
                print(f"  ❌ Export failed: {e2}")
                return False

    print(f"  Compiling via AOTInductor ...")
    try:
        torch._inductor.aoti_compile_and_package(ep, package_path=str(out_path))
    except Exception as e:
        print(f"  ❌ AOT compile failed: {e}")
        return False

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  ✅ Saved: {out_path} ({size_mb:.1f} MB)")
    return True


def main():
    ap = argparse.ArgumentParser(
        description="Convert state-dict models to AOTI (.pt2) for vaccel-ptc backends.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--device", required=True, choices=["cuda", "cpu"],
        help="Target device for the exported model.",
    )
    ap.add_argument(
        "--model", default=None,
        help="Comma-separated model name(s). Defaults to all models in the registry.",
    )
    ap.add_argument(
        "--models-dir", default=None,
        help="Directory containing model subdirectories. Defaults to the directory of this script.",
    )
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("❌ CUDA requested but not available.")
        sys.exit(1)

    models_dir = Path(args.models_dir) if args.models_dir else Path(__file__).parent

    if args.model:
        names = [m.strip() for m in args.model.split(",")]
        unknown = [n for n in names if n not in MODEL_REGISTRY]
        if unknown:
            print(f"❌ Unknown model(s): {unknown}")
            print(f"   Available: {list(MODEL_REGISTRY)}")
            sys.exit(1)
    else:
        names = list(MODEL_REGISTRY)

    print(f"Converting {len(names)} model(s) → device={args.device}\n")

    ok, fail = 0, 0
    for name in names:
        print(f"[{name}]")
        if convert_one(name, args.device, models_dir):
            ok += 1
        else:
            fail += 1
        print()

    print(f"Done. ✅ {ok} succeeded, ❌ {fail} failed.")
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
