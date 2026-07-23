# app/rapl.py
import os
from dataclasses import dataclass
from typing import List, Optional

RAPL_ROOT = "/sys/class/powercap/intel-rapl"

@dataclass
class RaplPackage:
    energy_path: str
    max_range_path: Optional[str] = None

def _read_int(path: str) -> Optional[int]:
    try:
        with open(path, "r") as f:
            return int(f.read().strip())
    except Exception:
        return None

def discover_rapl_packages() -> List[RaplPackage]:
    """
    Discover intel-rapl package paths.
    Typical: /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj
    Also:   /sys/class/powercap/intel-rapl/intel-rapl:1/energy_uj, etc.
    """
    pkgs: List[RaplPackage] = []
    if not os.path.isdir(RAPL_ROOT):
        return pkgs

    # Find directories named intel-rapl:<N>
    for name in sorted(os.listdir(RAPL_ROOT)):
        if not name.startswith("intel-rapl:"):
            continue
        base = os.path.join(RAPL_ROOT, name)
        energy_path = os.path.join(base, "energy_uj")
        if not os.path.exists(energy_path):
            continue
        max_range = os.path.join(base, "max_energy_range_uj")
        pkgs.append(RaplPackage(
            energy_path=energy_path,
            max_range_path=max_range if os.path.exists(max_range) else None
        ))
    return pkgs

def read_energy_uj(packages: List[RaplPackage]) -> Optional[List[int]]:
    """
    Read energy counters for all discovered packages.
    Returns list of energy_uj per package, or None if none readable.
    """
    if not packages:
        return None
    vals = []
    for p in packages:
        v = _read_int(p.energy_path)
        if v is None:
            return None
        vals.append(v)
    return vals

def compute_energy_diff_uj(
    last_energy_uj: Optional[List[int]],
    curr_energy_uj: Optional[List[int]],
    packages: List[RaplPackage],
) -> Optional[int]:
    """
    Compute the total wrap-corrected energy delta (microjoules) across all
    packages between two consecutive reads. Returns None if not computable.

    Callers should poll frequently enough that at most one wrap can occur
    between `last_energy_uj` and `curr_energy_uj` for a given package -
    this is what makes summing these deltas over a run's lifetime an exact
    total, rather than an approximation reconstructed from average power.
    """
    if (
        last_energy_uj is None
        or curr_energy_uj is None
        or not packages
        or len(last_energy_uj) != len(curr_energy_uj)
        or len(packages) != len(curr_energy_uj)
    ):
        return None

    total_diff_uj = 0
    for i, p in enumerate(packages):
        last = last_energy_uj[i]
        curr = curr_energy_uj[i]
        diff = curr - last

        if diff < 0:
            # Counter wrapped; try to fix if max range is available
            max_range = _read_int(p.max_range_path) if p.max_range_path else None
            if max_range and max_range > 0:
                diff = (max_range - last) + curr
            else:
                # Fallback: clamp (keeps old behavior)
                diff = 0

        total_diff_uj += diff

    return total_diff_uj
