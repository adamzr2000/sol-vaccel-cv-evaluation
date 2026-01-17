#!/usr/bin/env python3
import argparse
import json
import os
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

    # 3. Video Models
    "mc3_18": {
        "builder": models.video.mc3_18,
        "weights": models.video.MC3_18_Weights.DEFAULT
    },
    "r3d_18": {
        "builder": models.video.r3d_18,
        "weights": models.video.R3D_18_Weights.DEFAULT
    }
}
def main():
    # Helper text to show examples at the bottom of the help message
    example_text = """examples:
  # Download DeepLabV3 (Default folder: baseline_deeplabv3_resnet50)
  python3 download_baseline_model.py --model deeplabv3_resnet50

  # Download ResNet50 to a custom folder
  python3 download_baseline_model.py --model resnet50 --outdir ./my_resnet_models

  # Show this help message
  python3 download_baseline_model.py -h"""

    ap = argparse.ArgumentParser(
        description="Download and serialize PyTorch Baseline Models",
        epilog=example_text,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    ap.add_argument("--model", required=True, choices=MODEL_REGISTRY.keys(),
                    help="The specific model architecture to download.")
    
    ap.add_argument("--outdir", default=None,
                    help="Custom output directory. Defaults to 'baseline_<model_name>'.")

    args = ap.parse_args()

    # Determine Output Directory
    model_name = args.model
    if args.outdir:
        outdir = Path(args.outdir)
    else:
        outdir = Path(f"{model_name}")

    outdir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Preparing to download: {model_name}")
    print(f"📂 Output folder: {outdir}")

    # A) Download via torchvision
    os.environ["TORCH_HOME"] = str(outdir)
    
    entry = MODEL_REGISTRY[model_name]
    builder_func = entry["builder"]
    weights_obj = entry["weights"]

    print("   Downloading weights...")
    try:
        model = builder_func(weights=weights_obj).eval()
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return

    # B) Save State Dict
    sd_path = outdir / f"{model_name}_state_dict.pt"
    torch.save(model.state_dict(), sd_path)

    # C) Save TorchScript
    ts_path = outdir / f"{model_name}.torchscript.pt"
    ts_status = "Skipped"
    try:
        scripted = torch.jit.script(model)
        scripted.save(str(ts_path))
        ts_status = "✅ Saved"
    except Exception as e:
        ts_status = f"❌ Failed ({e})"
        print(f"   [Warning] TorchScript export failed. This is common for some video/detection models.")

    # D) Save Metadata
    meta = {
        "model": model_name,
        "weights_enum": str(weights_obj),
        "num_params": sum(p.numel() for p in model.parameters()),
        "files": {
            "state_dict": str(sd_path.name),
            "torchscript": str(ts_path.name) if "Saved" in ts_status else None
        }
    }
    (outdir / "metadata.json").write_text(json.dumps(meta, indent=2))

    print("\n✅ Success!")
    print(f" - State Dict:  {sd_path}")
    print(f" - TorchScript: {ts_path} ({ts_status})")
    print(f" - Metadata:    {outdir / 'metadata.json'}")

if __name__ == "__main__":
    main()