#!/usr/bin/env python3
"""Use the official DPG scorer with all CSV question rows retained."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from t2i_dpg import load_dpg_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--official-script", required=True)
    args, official_args = parser.parse_known_args()
    script = Path(args.official_script).resolve()
    spec = importlib.util.spec_from_file_location("_official_dpg_bench", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load DPG evaluator: {script}")
    evaluator = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = evaluator
    spec.loader.exec_module(evaluator)

    def prepare_dpg_data(evaluation_args):
        # ELLA's parser drops the first *question*, not the first prompt.
        # Only repair data loading; VQA, crops and all scoring stay upstream.
        return load_dpg_data(evaluation_args.csv)[1]

    evaluator.prepare_dpg_data = prepare_dpg_data
    sys.argv = [str(script), *official_args]
    evaluator.main()


if __name__ == "__main__":
    main()
