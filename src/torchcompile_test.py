# torchcompile_test.py
import os
import time
import statistics
import platform
import traceback
from pathlib import Path

import torch
import torch.hub
import torch._dynamo as dynamo
from torchvision import models

# -------------------------------------------------------
# Offline / cache control
# -------------------------------------------------------
os.environ.setdefault("TORCH_HOME", "/src/models/_torch_cache")

# -------------------------------------------------------
# TEMP DEBUG: print a short stack trace if anything tries to download weights
# -------------------------------------------------------
_ORIG_DOWNLOAD = torch.hub.download_url_to_file

def _download_url_to_file_debug(*args, **kwargs):
    url = args[0] if args else "<unknown>"
    print("\n🚨 Download triggered! URL:", url)
    traceback.print_stack(limit=16)
    return _ORIG_DOWNLOAD(*args, **kwargs)

if os.environ.get("TORCHCOMPILE_DEBUG_DOWNLOAD", "1").strip() not in ("0", "false", "False"):
    torch.hub.download_url_to_file = _download_url_to_file_debug

# -------------------------------------------------------
# Helpers
# -------------------------------------------------------
def _sync_if_cuda(device: str):
    if device == "cuda":
        torch.cuda.synchronize()

def time_inference(fn, x, iters=50, warmup=10, device="cpu"):
    with torch.inference_mode():
        for _ in range(warmup):
            _ = fn(x)
        _sync_if_cuda(device)

        times_ms = []
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = fn(x)
            _sync_if_cuda(device)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    return times_ms

def summarize(label, times_ms):
    times_sorted = sorted(times_ms)
    avg = sum(times_ms) / len(times_ms)
    med = statistics.median(times_ms)
    p90 = times_sorted[int(0.90 * (len(times_sorted) - 1))]
    print(f"\n{label}")
    print(f"  iters:  {len(times_ms)}")
    print(f"  avg:    {avg:.3f} ms")
    print(f"  median: {med:.3f} ms")
    print(f"  p90:    {p90:.3f} ms")
    print(f"  min:    {min(times_ms):.3f} ms")
    print(f"  max:    {max(times_ms):.3f} ms")

class OutOnly(torch.nn.Module):
    def __init__(self, m: torch.nn.Module):
        super().__init__()
        self.m = m
    def forward(self, x):
        return self.m(x)["out"]

def infer_aux_loss_from_state_dict(state: dict) -> bool:
    # If aux head params exist, checkpoint expects aux_loss=True
    for k in state.keys():
        if k.startswith("aux_classifier."):
            return True
    return False

def build_deeplab_from_state_dict(state: dict, device: str) -> torch.nn.Module:
    aux_loss = infer_aux_loss_from_state_dict(state)
    print(f"Checkpoint aux_loss detected: {aux_loss}")

    # No downloads: weights=None and weights_backbone=None are the key pieces
    model = models.segmentation.deeplabv3_resnet50(
        weights=None,
        weights_backbone=None,
        aux_loss=aux_loss,
    )

    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return OutOnly(model)

# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n========== SYSTEM INFO ==========")
    print("Python:", platform.python_version())
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    backend_name = "inductor"
    print("torch.compile backend:", backend_name)
    print("Device:", device)
    print("TORCH_HOME:", os.environ.get("TORCH_HOME"))

    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print("SM count:", torch.cuda.get_device_properties(0).multi_processor_count)
        print("CUDA version:", torch.version.cuda)

    print("Torch threads:", torch.get_num_threads())
    print("Deterministic:", torch.are_deterministic_algorithms_enabled())
    print("=================================\n")

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    # ---------------------------------------------------
    # Load local checkpoint
    # ---------------------------------------------------
    sd_path = Path("models/deeplabv3_resnet50/deeplabv3_resnet50_state_dict.pt")
    if not sd_path.exists():
        raise FileNotFoundError(f"Missing local checkpoint: {sd_path}")

    state = torch.load(sd_path, map_location="cpu")

    # ---------------------------------------------------
    # Build two identical models (eager vs compiled)
    # ---------------------------------------------------
    wrapped_eager = build_deeplab_from_state_dict(state, device=device)
    wrapped_for_compile = build_deeplab_from_state_dict(state, device=device)

    # ---------------------------------------------------
    # Input
    # ---------------------------------------------------
    x = torch.randn(1, 3, 520, 520, device=device)
    print("Input shape:", tuple(x.shape))
    print("Input dtype:", x.dtype)

    # ---------------------------------------------------
    # Benchmark eager FIRST
    # ---------------------------------------------------
    WARMUP = 10
    ITERS = 50
    eager_times = time_inference(wrapped_eager, x, iters=ITERS, warmup=WARMUP, device=device)

    # ---------------------------------------------------
    # Reset + Compile
    # ---------------------------------------------------
    torch._dynamo.reset()

    print("\nCompiling model...")
    t0 = time.perf_counter()
    compiled_fn = torch.compile(
        wrapped_for_compile,
        backend=backend_name,
        mode="reduce-overhead",
        dynamic=False,
    )

    with torch.inference_mode():
        _ = compiled_fn(x)  # compile + first run
    _sync_if_cuda(device)

    compile_time_ms = (time.perf_counter() - t0) * 1000.0
    print(f"Compilation time: {compile_time_ms:.2f} ms")

    # ---------------------------------------------------
    # Benchmark compiled
    # ---------------------------------------------------
    compiled_times = time_inference(compiled_fn, x, iters=ITERS, warmup=WARMUP, device=device)

    summarize("Eager (no torch.compile)", eager_times)
    summarize('Compiled (torch.compile, mode="reduce-overhead")', compiled_times)

    eager_avg = sum(eager_times) / len(eager_times)
    comp_avg = sum(compiled_times) / len(compiled_times)
    speedup = eager_avg / comp_avg if comp_avg > 0 else float("inf")
    print(f"\nSpeedup (avg): {speedup:.2f}×")

    # ---------------------------------------------------
    # Output sanity check (numerical)
    # ---------------------------------------------------
    with torch.inference_mode():
        y_e = wrapped_eager(x)
        y_c = compiled_fn(x)
    max_abs_diff = (y_e - y_c).abs().max().item()
    mean_abs_diff = (y_e - y_c).abs().mean().item()
    print("\nOutput check:")
    print("  Output shape:", tuple(y_c.shape))
    print(f"  Max abs diff:  {max_abs_diff:.6e}")
    print(f"  Mean abs diff: {mean_abs_diff:.6e}")

    # ---------------------------------------------------
    # Memory stats
    # ---------------------------------------------------
    if device == "cuda":
        mem_mb = torch.cuda.memory_allocated() / 1024**2
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        print("\nGPU Memory Usage:")
        print(f"  current: {mem_mb:.1f} MB")
        print(f"  peak:    {peak_mb:.1f} MB")

    # ---------------------------------------------------
    # Dynamo stats
    # ---------------------------------------------------
    print("\nTorchDynamo Stats:")
    print(dynamo.utils.counters)

if __name__ == "__main__":
    main()
