#!/usr/bin/env python3
"""Run official GenEval with the detector config from the selected environment."""

from __future__ import annotations

import argparse
import importlib.util
import os
import runpy
import sys
from pathlib import Path


MODEL_CONFIG = "mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"


def resolve_model_config(explicit: str | None = None) -> Path:
    """Locate the original config tree without importing MMDetection/CUDA."""

    override = explicit if explicit is not None else os.environ.get("GENEVAL_MODEL_CONFIG")
    if override is not None:
        path = Path(override).expanduser().resolve()
        if not path.is_file():
            source = "--model-config" if explicit is not None else "GENEVAL_MODEL_CONFIG"
            raise FileNotFoundError(f"{source} is not an existing config file: {path}")
        return path
    spec = importlib.util.find_spec("mmdet")
    if spec is None or spec.origin is None:
        raise FileNotFoundError(
            f"MMDetection is not installed in {sys.executable}. Install the compatible "
            "MMDetection 2.x package or set GENEVAL_MODEL_CONFIG to the detector config "
            "inside a complete MMDetection config tree."
        )
    package = Path(spec.origin).resolve().parent
    candidates = (
        package / ".mim" / "configs" / MODEL_CONFIG,
        package.parent / "configs" / MODEL_CONFIG,
    )
    for path in candidates:
        if path.is_file():
            return path.resolve()
    searched = "\n".join(f"  {path}" for path in candidates)
    raise FileNotFoundError(
        f"GenEval's Mask2Former config was not found in {sys.executable}. Searched:\n"
        f"{searched}\nInstall a complete MMDetection 2.x wheel/source checkout, or set "
        "GENEVAL_MODEL_CONFIG to the matching config in its original tree "
        "so relative _base_ files remain available."
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--official-script")
    parser.add_argument("--check-config", action="store_true")
    args, official_args = parser.parse_known_args(argv)
    if not args.check_config and not args.official_script:
        parser.error("--official-script is required unless --check-config is used")
    config_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    config_parser.add_argument("--model-config")
    selected, _ = config_parser.parse_known_args(official_args)
    try:
        config = resolve_model_config(selected.model_config)
    except (OSError, ImportError, ValueError) as exc:
        parser.exit(2, f"GenEval config error: {exc}\n")
    print(f"[GenEval] model config: {config}", flush=True)
    if args.check_config:
        return
    script = Path(args.official_script).expanduser().resolve()
    if not script.is_file():
        parser.error(f"official GenEval script does not exist: {script}")
    if selected.model_config is None:
        official_args.extend(["--model-config", str(config)])
    original_argv, original_path = sys.argv, sys.path[:]
    try:
        sys.argv = [str(script), *official_args]
        sys.path.insert(0, str(script.parent))
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = original_argv
        sys.path[:] = original_path


if __name__ == "__main__":
    main()
