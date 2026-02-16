#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import torch
from pathlib import Path
from torchvision import models

# ==========================================
# MODEL REGISTRY
# ==========================================
MODEL_REGISTRY = {
    # 1. Segmentation Models
    "deeplabv3_resnet50": {
        "builder": models.segmentation.deeplabv3_resnet50,
        "weights": models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
    },
    "fcn_resnet50": {
        "builder": models.segmentation.fcn_resnet50,
        "weights": models.segmentation.FCN_ResNet50_Weights.DEFAULT
    },
    "deeplabv3_resnet101": {
        "builder": models.segmentation.deeplabv3_resnet101,
        "weights": models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT
    },
    "deeplabv3_mobilenet_v3_large": {
        "builder": models.segmentation.deeplabv3_mobilenet_v3_large,
        "weights": models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
    },
    "fcn_resnet101": {
        "builder": models.segmentation.fcn_resnet101,
        "weights": models.segmentation.FCN_ResNet101_Weights.DEFAULT
    },
    "lraspp_mobilenet_v3_large": {
        "builder": models.segmentation.lraspp_mobilenet_v3_large,
        "weights": models.segmentation.LRASPP_MobileNet_V3_Large_Weights.DEFAULT
    },
    # 2. Classification Models
    "resnet50": {
        "builder": models.resnet50,
        "weights": models.ResNet50_Weights.DEFAULT
    },
    "mobilenet_v3_large": {
        "builder": models.mobilenet_v3_large,
        "weights": models.MobileNet_V3_Large_Weights.DEFAULT
    },
    "swin_t": {
        "builder": models.swin_t,
        "weights": models.Swin_T_Weights.DEFAULT
    },
    "swin_s": {
        "builder": models.swin_s,
        "weights": models.Swin_S_Weights.DEFAULT
    },
    "swin_v2_b": {
        "builder": models.swin_v2_b,
        "weights": models.Swin_V2_B_Weights.DEFAULT
    },
    # 3. Video Models
    "mc3_18": {
        "builder": models.video.mc3_18,
        "weights": models.video.MC3_18_Weights.DEFAULT
    },
    "r3d_18": {
        "builder": models.video.r3d_18,
        "weights": models.video.R3D_18_Weights.DEFAULT
    },
    "r2plus1d_18": {
        "builder": models.video.r2plus1d_18,
        "weights": models.video.R2Plus1D_18_Weights.DEFAULT
    },
    # Swin3D (Video Swin Transformer)
    "swin3d_t": {
        "builder": models.video.swin3d_t,
        "weights": models.video.Swin3D_T_Weights.DEFAULT
    },
    "swin3d_s": {
        "builder": models.video.swin3d_s,
        "weights": models.video.Swin3D_S_Weights.DEFAULT
    },
    "swin3d_b": {
        "builder": models.video.swin3d_b,
        "weights": models.video.Swin3D_B_Weights.DEFAULT
    },
    # ==========================================
    # YOLOv5 (PyTorch Hub)
    # ==========================================
    "yolov5s": {
        "source": "torchhub",
        "repo": "ultralytics/yolov5",
        "hub_model": "yolov5s",
        "pretrained": True,
    },
}


def _cleanup_caches(outdir: Path, source: str) -> None:
    """
    Keep only relevant artifacts.
      - torchvision: remove outdir/hub/ (download cache)
      - torchhub: remove outdir/_torchhub_cache/* EXCEPT trusted_list
    """
    if source == "torchvision":
        hub_dir = outdir / "hub"
        if hub_dir.exists():
            shutil.rmtree(hub_dir, ignore_errors=True)

    elif source == "torchhub":
        cache_dir = outdir / "_torchhub_cache"
        if not cache_dir.exists():
            return

        for child in cache_dir.iterdir():
            # Keep trust list so torch doesn't warn/prompt again
            if child.name == "trusted_list":
                continue
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                try:
                    child.unlink()
                except Exception:
                    pass


def main():
    example_text = """examples:
  python3 download_baseline_model.py --model deeplabv3_resnet50
  python3 download_baseline_model.py --model resnet50 --outdir ./my_resnet_models
  python3 download_baseline_model.py --model yolov5s
  python3 download_baseline_model.py -h"""

    ap = argparse.ArgumentParser(
        description="Download and serialize PyTorch Baseline Models",
        epilog=example_text,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    ap.add_argument("--model", required=True, choices=MODEL_REGISTRY.keys(),
                    help="The specific model architecture to download.")

    ap.add_argument("--outdir", default=None,
                    help="Custom output directory. Defaults to '<model_name>'.")

    ap.add_argument("--force-reload", action="store_true",
                    help="Force torch.hub to re-download and reload the repo (useful if cache is stale).")

    ap.add_argument(
        "--keep-cache",
        action="store_true",
        help="Keep cache folders (hub/, _torchhub_cache/). Default: caches are removed after successful download."
    )

    args = ap.parse_args()

    model_name = args.model
    entry = MODEL_REGISTRY[model_name]
    source = entry.get("source", "torchvision")

    # Keep unified folder naming (same as you wanted)
    outdir = Path(args.outdir) if args.outdir else Path(model_name)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Preparing to download: {model_name}")
    print(f"📂 Output folder: {outdir}")
    print("   Downloading weights...")

    weights_obj = None

    try:
        if source == "torchhub":
            # IMPORTANT: hub_dir must be absolute, because we chdir(outdir) below
            hub_dir = (outdir / "_torchhub_cache").resolve()
            hub_dir.mkdir(parents=True, exist_ok=True)
            torch.hub.set_dir(str(hub_dir))

            repo = entry["repo"]
            hub_model = entry["hub_model"]
            pretrained = entry.get("pretrained", True)

            # Remove broken nested folder from previous runs if it exists
            bad_nested = outdir / model_name
            if bad_nested.exists() and bad_nested.is_dir():
                shutil.rmtree(bad_nested)

            # Load with cwd=outdir so yolov5s.pt lands inside yolov5s/
            old_cwd = os.getcwd()
            os.chdir(str(outdir))
            try:
                model = torch.hub.load(
                    repo,
                    hub_model,
                    pretrained=pretrained,
                    trust_repo=True,
                    force_reload=bool(args.force_reload),
                )
            finally:
                os.chdir(old_cwd)

            model.eval()
            target = model.model if hasattr(model, "model") else model

        else:
            # torchvision path: cache weights inside outdir
            os.environ["TORCH_HOME"] = str(outdir)

            builder_func = entry["builder"]
            weights_obj = entry["weights"]
            model = builder_func(weights=weights_obj).eval()
            target = model

    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return

    # ------------------------------------------------------------------
    # B) Save State Dict
    # ------------------------------------------------------------------
    sd_path = outdir / f"{model_name}_state_dict.pt"
    try:
        torch.save(target.state_dict(), sd_path)
    except Exception as e:
        print(f"❌ Error saving state_dict: {e}")
        return

    # ------------------------------------------------------------------
    # C) TorchScript (Option A: skip for torchhub models)
    # ------------------------------------------------------------------
    ts_path = outdir / f"{model_name}.torchscript.pt"
    if source == "torchhub":
        ts_status = "Skipped (torchhub model)"
        # Ensure we don't leave a stale file if it exists from previous runs
        if ts_path.exists():
            try:
                ts_path.unlink()
            except Exception:
                pass
    else:
        ts_status = "Skipped"
        try:
            scripted = torch.jit.script(target)
            scripted.save(str(ts_path))
            ts_status = "✅ Saved (script)"
        except Exception as e:
            ts_status = f"❌ Failed ({e})"
            print("   [Warning] TorchScript export failed. This is common for some models.")

    # ------------------------------------------------------------------
    # D) Save Metadata
    # ------------------------------------------------------------------
    try:
        num_params = sum(p.numel() for p in target.parameters())
    except Exception:
        num_params = None

    meta = {
        "model": model_name,
        "source": source,
        "weights_enum": str(weights_obj) if weights_obj is not None else None,
        "num_params": num_params,
        "files": {
            "state_dict": str(sd_path.name),
            "torchscript": str(ts_path.name) if "✅ Saved" in ts_status else None
        }
    }
    (outdir / "metadata.json").write_text(json.dumps(meta, indent=2))

    # ------------------------------------------------------------------
    # E) Cleanup caches (default)
    # ------------------------------------------------------------------
    if not args.keep_cache:
        _cleanup_caches(outdir, source)

    print("\n✅ Success!")
    print(f" - State Dict:  {sd_path}")
    print(f" - TorchScript: {ts_path} ({ts_status})")
    print(f" - Metadata:    {outdir / 'metadata.json'}")


if __name__ == "__main__":
    main()
