#!/usr/bin/env python3
"""
test.py — quick Torch sanity check (CPU/GPU)

- Verifies import
- Prints torch/torchvision versions
- Reports CUDA availability, device count, device name, and cuDNN
- Runs a small tensor op on the selected device
- Exits with non-zero code if torch import fails

Usage:
  python3 test.py               # auto-selects CUDA if available, else CPU
  python3 test.py --cpu         # force CPU
  python3 test.py --cuda 0      # force CUDA device 0
"""

import sys
import argparse

def main():
    try:
        import torch
    except Exception as e:
        print("[ERROR] Could not import torch:", e)
        sys.exit(1)

    # argparse
    p = argparse.ArgumentParser()
    p.add_argument("--cpu", action="store_true", help="force CPU")
    p.add_argument("--cuda", type=int, default=None, help="force a specific CUDA device id")
    args = p.parse_args()

    # Basic versions
    tv_ver = None
    try:
        import torchvision
        tv_ver = torchvision.__version__
    except Exception:
        pass

    print("=== PyTorch Environment Check ===")
    print(f"torch version       : {torch.__version__}")
    print(f"torchvision version : {tv_ver if tv_ver else '(not installed)'}")

    # CUDA info
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA available      : {cuda_avail}")

    if cuda_avail:
        print(f"CUDA device count   : {torch.cuda.device_count()}")
        # cuDNN
        try:
            print(f"cuDNN version       : {torch.backends.cudnn.version()}")
            print(f"cuDNN enabled       : {torch.backends.cudnn.enabled}")
        except Exception:
            print("cuDNN info          : (unavailable)")

    # Choose device
    device = torch.device("cpu")
    if not args.cpu and cuda_avail:
        if args.cuda is not None:
            idx = int(args.cuda)
            if idx < 0 or idx >= torch.cuda.device_count():
                print(f"[WARN] Requested CUDA device {idx} not present, falling back to 0")
                idx = 0
            device = torch.device(f"cuda:{idx}")
        else:
            device = torch.device("cuda:0")

    # Device details
    if device.type == "cuda":
        idx = device.index or 0
        name = torch.cuda.get_device_name(idx)
        cap = torch.cuda.get_device_capability(idx)
        print(f"Selected device     : cuda:{idx} ({name}), capability {cap[0]}.{cap[1]}")
    else:
        print("Selected device     : cpu")

    # Tiny compute test
    try:
        x = torch.randn(1024, 1024, device=device)
        y = torch.randn(1024, 1024, device=device)
        z = (x @ y).sum().item()
        print(f"Compute test        : OK (dot sum = {z:.4f}) on {device}")
    except Exception as e:
        print("[ERROR] Compute test failed:", e)
        sys.exit(2)

    print("Status              : SUCCESS")
    sys.exit(0)

if __name__ == "__main__":
    main()
