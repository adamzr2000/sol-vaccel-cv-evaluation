import logging
import os
import subprocess
from pathlib import Path
from graphlib import TopologicalSorter

import click

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])

MANIFEST_NAME = "dlopen_manifest.txt"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_vaccel_lib_dir() -> str:
    try:
        result = subprocess.run(
            ["pkg-config", "--variable", "libdir", "vaccel"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()

    except subprocess.CalledProcessError as e:
        logger.error("Failed to parse vaccel.pc")
        logger.error("pkg-config output:\n%s", e.stderr)
        raise


def generate_dependency_order(
    dst_dir: str, common_lib_dirs: list[Path] | None
) -> list[str]:
    """Generate a topologically sorted list of shared library dependencies.

    Args:
        dst_dir: Directory containing .so files to analyze

    Returns:
        List of relative paths to .so files in dependency order
    """
    unsorted = []
    dep_graph = {}

    # Set library search paths
    lib_dirs = [_get_vaccel_lib_dir()]
    if common_lib_dirs is not None:
        for d in common_lib_dirs:
            lib_dirs.append(str(d))

    env = {"LD_LIBRARY_PATH": f"{':'.join(lib_dirs)}:{os.getenv('LD_LIBRARY_PATH')}"}

    # Build dependency graph
    for so_file in Path(dst_dir).rglob("*wrapper.so*"):
        rel = str(so_file.relative_to(dst_dir))
        unsorted.append(rel)
        dep_graph[rel] = []

        # Parse dependencies using ldd
        try:
            result = subprocess.run(
                ["ldd", str(so_file)],
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )

            deps = []
            for line in result.stdout.splitlines():
                if "=>" in line:
                    parts = line.split()
                    if len(parts) >= 3:
                        if parts[2] == "not":
                            msg = f"Dependency not found: {parts[0]}"
                            raise FileNotFoundError(msg)
                        deps.append(parts[2])

            for dep in deps:
                dst_path = Path(dst_dir).resolve()
                dep_path = Path(dep).resolve()
                if dep_path.is_relative_to(dst_path):
                    dep_graph[rel].append(str(dep_path.relative_to(dst_path)))
                elif dst_path.is_relative_to(dep_path.parent):
                    dep_graph[rel].append(os.path.relpath(dep_path, dst_path))
                elif "libsol" in dep_path.name:
                    dep_graph[rel].append(os.path.relpath(dep_path, dst_path))

        except FileNotFoundError:
            raise
        except Exception:
            # If ldd fails, just skip this file's dependencies
            pass

        dep_graph[rel].reverse()

    # Topologically sort deps
    if dep_graph:
        try:
            ts = TopologicalSorter(dep_graph)
            return list(ts.static_order())
        except Exception:
            return unsorted
    else:
        return unsorted


def generate_dlopen_manifest(dst_dir: Path, common_lib_dirs: list[Path] | None = None):
    """Generate a dlopen manifest for a vAccel SOL wrapper lib."""
    deps = generate_dependency_order(dst_dir, common_lib_dirs)
    dst_dir.joinpath(MANIFEST_NAME).write_text("\n".join(deps))


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument(
    "dst_dir",
    type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path),
)
def generate_manifest(dst_dir):
    """Generate a dlopen manifest for a vAccel SOL wrapper lib."""
    generate_dlopen_manifest(dst_dir)


if __name__ == "__main__":
    generate_manifest()
