import logging
import os
import shutil
import subprocess
from pathlib import Path
from string import Template

import click

from gen_dlopen_manifest import generate_dlopen_manifest

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])

SCRIPT_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = SCRIPT_DIR.joinpath("templates").resolve()
MAKEFILE_IN = TEMPLATES_DIR.joinpath("Makefile.in").read_text()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomTemplate(Template):
    delimiter = "@@"
    pattern = r"""
        @@(?:
            (?P<escaped>@@) |
            (?P<named>[_a-zA-Z][_a-zA-Z0-9]*)@@ |
            (?P<braced>[_a-zA-Z][_a-zA-Z0-9]*)@@ |
            (?P<invalid>)
        )
    """


def compile_wrappers(vaccel_dir: Path):
    try:
        subprocess.run(
            ["make", "-C", str(vaccel_dir)], capture_output=True, text=True, check=True
        )
    except subprocess.CalledProcessError as e:
        logger.error("Compilation failed with exit code %s", e.returncode)
        logger.error("Make output:\n%s", e.stderr)
        raise


def generate_vaccel_directory(
    model_lib_dir: Path, model_name: str, common_lib_dirs: list[str]
):
    """Generate 'vaccel' directory for a SOL model."""
    model_wrapper_dir = TEMPLATES_DIR.joinpath(model_name)
    if not model_wrapper_dir.is_dir():
        msg = f"Unsupported model: {model_name}"
        raise ValueError(msg)

    out_vaccel_dir = model_lib_dir.joinpath("vaccel")
    out_vaccel_dir.mkdir(parents=True, exist_ok=True)

    common_lib_dirs_rel = [".", ".."]
    for common_lib_dir in common_lib_dirs:
        if model_lib_dir.is_relative_to(common_lib_dir):
            vaccel_lib_dir = model_lib_dir.joinpath("vaccel")
            rel_lib_dir = Path(os.path.relpath(common_lib_dir, vaccel_lib_dir))
            if str(rel_lib_dir) not in common_lib_dirs_rel:
                common_lib_dirs_rel.append(str(rel_lib_dir))

    makefile_substitutes = {
        "model": model_name,
        "common_lib_dirs": ":".join(common_lib_dirs_rel),
    }

    try:
        makefile_template = CustomTemplate(MAKEFILE_IN)
        out_makefile = makefile_template.substitute(makefile_substitutes)
        out_vaccel_dir.joinpath("Makefile").write_text(out_makefile)
    except Exception as e:
        msg = f"Failed to generate Makefile for {model_name}"
        raise RuntimeError(msg) from e

    sol_wrappers = list(model_wrapper_dir.glob("sol_*"))
    if not sol_wrappers:
        msg = f"No wrapper files found for {model_name}"
        raise FileNotFoundError(msg)

    for sol_wrapper in sol_wrappers:
        try:
            shutil.copy2(sol_wrapper, out_vaccel_dir.joinpath(sol_wrapper.name))
        except Exception as e:
            msg = f"Failed to copy wrapper file: {sol_wrapper}"
            raise RuntimeError(msg) from e

    compile_wrappers(out_vaccel_dir)
    generate_dlopen_manifest(out_vaccel_dir, common_lib_dirs)


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument(
    "models_dir",
    type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path),
)
def generate_sol_vaccel_wrappers(models_dir):
    """Generate vAccel wrappers for SOL models in MODELS_DIR."""
    common_lib_dirs = []
    for lib_path in models_dir.rglob("libsol-dnn*.so"):
        lib_dir = lib_path.parent.resolve()
        if lib_dir not in common_lib_dirs:
            common_lib_dirs.append(lib_dir)

    model_modules = list(models_dir.rglob("sol_*.py"))
    if not model_modules:
        logger.warning("No model modules (sol_*.py) found in %s", models_dir)
        click.echo("No model modules found")
        return

    success_count = 0
    error_count = 0

    for model_module in model_modules:
        model_lib_dir = model_module.parent.resolve()
        if model_lib_dir.name == "vaccel":
            continue

        model_name = model_module.stem.replace("sol_", "").replace("_gpt", "")
        model_common_lib_dirs = [model_lib_dir] + common_lib_dirs
        click.echo(f"Processing model {model_lib_dir}...")

        try:
            generate_vaccel_directory(model_lib_dir, model_name, model_common_lib_dirs)
            success_count += 1
        except Exception:
            logger.exception("Failed to process %s", model_module.name)
            error_count += 1

    click.echo(f"Successfully processed {success_count} model(s)")
    if error_count > 0:
        msg = f"Failed to process {error_count} model(s)"
        click.echo(f"Failed: {error_count} model(s)", err=True)
        raise click.ClickException(msg)


if __name__ == "__main__":
    generate_sol_vaccel_wrappers()
