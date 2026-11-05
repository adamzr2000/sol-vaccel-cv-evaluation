#!/usr/bin/env python3
"""
_dual_import.py

Loads same-named modules from ../vaccel/ and ../ros2/ into this script's
namespace without collisions. Both sibling directories have their own
plot_config.py (and vaccel/ has plot_data_e2e.py; ros2/ has
barplot_e2e_latency_and_fps.py, barplot_energy_consumption.py,
barplot_energy_per_frame.py) - a plain `sys.path.insert` + `import` approach
caches whichever one loads first under sys.modules["plot_config"], so the
second directory's same-named internal imports silently resolve to the
wrong file (or ImportError if the names don't match).

import_from(dir_path, filename) loads exactly one module by path, having
temporarily made dir_path the front of sys.path and evicted any stale
same-named entries from sys.modules first, so that module's own top-level
`from plot_config import ...`-style statements resolve within its own
directory - then reverts both, so the next call (e.g. the other directory)
starts from a clean slate.
"""
from __future__ import annotations

import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path

# Every bare module name either directory might import internally - evicted
# from sys.modules (and restored afterward) around each load so a name
# already cached from one directory can't leak into the other's import.
_SHARED_NAMES = ("plot_config", "plot_data_e2e")


@contextmanager
def _isolated(dir_path: Path):
    added = str(dir_path)
    sys.path.insert(0, added)
    saved = {name: sys.modules.pop(name, None) for name in _SHARED_NAMES}
    try:
        yield
    finally:
        sys.path.remove(added)
        for name, mod in saved.items():
            if mod is not None:
                sys.modules[name] = mod
            else:
                sys.modules.pop(name, None)


def import_from(dir_path: Path, filename: str, alias: str | None = None):
    """Load `filename` from `dir_path` with import isolation (see module
    docstring). `alias` must be unique across all calls in the same script
    (used as the sys.modules key); defaults to f"_dual_import_{filename}"."""
    alias = alias or f"_dual_import_{dir_path.name}_{filename}"
    path = dir_path / filename
    with _isolated(dir_path):
        spec = importlib.util.spec_from_file_location(alias, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[alias] = mod
        spec.loader.exec_module(mod)
    return mod
