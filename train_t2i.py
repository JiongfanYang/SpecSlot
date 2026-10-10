"""Distributed rectified-flow training for text-conditioned JiT and SpecSlot."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from t2i_data import T2IShardDataset, collate_t2i
from t2i_model import TextConditionedDenoiser
from t2i_text import FrozenOpenCLIPTextEncoder
from t2i_logging import T2IRunLogger, reduce_loss
from util.misc import add_weight_decay
from t2i_eval import (
    compute_fid_kid,
    evaluate_generated,
    generate_dpg_t2i,
    generate_t2i,
    load_dpg_prompt_manifest,
    load_prompt_manifest,
    shard_prompt_indices,
)
from t2i_eval_merge import merge_evaluation_shards


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", choices=("specslot", "jit"), required=True)
    parser.add_argument("--image_size", type=int, choices=(256,), required=True)
    parser.add_argument(
        "--jit_variant", choices=("auto", "b8", "b16"), default="auto",
        help="JiT variant; auto selects B/8 at 256px",
    )
    parser.add_argument("--bottleneck", type=int, default=None)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--steps_per_epoch", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--shuffle_buffer", type=int, default=10_000)
    parser.add_argument("--shard_shuffle_buffer", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--caption_drop_prob", type=float, default=0.1)
    parser.add_argument("--P_mean", type=float, default=-0.8)
    parser.add_argument("--P_std", type=float, default=0.8)
    parser.add_argument("--t_eps", type=float, default=5e-2)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--lr_schedule", choices=("constant", "cosine"), default="constant")
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument(
        "--activation_checkpointing", action="store_true",
        help="Checkpoint transformer blocks during training to reduce activation memory",
    )
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--text_model", default="ViT-L-14")
    parser.add_argument("--text_pretrained", default="laion2b_s32b_b82k")
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument(
        "--eval_interval", type=int, default=10,
        help="Run text-to-image generation/evaluation after every N completed epochs; 0 disables it",
    )
    parser.add_argument(
        "--eval_prompt_file", default=None,
        help="GenEval JSON/JSONL prompt manifest; defaults to the official asset manifest",
    )
    parser.add_argument(
        "--eval_assets_dir", default=None,
        help="Directory containing geneval/, geneval_object_detector/, ella/, and modelscope/",
    )
    parser.add_argument(
        "--eval_reference_dir", default=None,
        help="Real-image directory for FID/KID; leave unset to skip those metrics",
    )
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument(
        "--eval_cfg", type=float, default=None,
        help="Fixed guidance scale; defaults to SpecSlot 4.0 / JiT 2.9",
    )
    parser.add_argument(
        "--eval_max_images", type=int, default=0,
        help="Maximum GenEval prompts; 0 evaluates the complete manifest",
    )
    parser.add_argument(
        "--eval_num_samples", type=int, default=4,
        help="Number of images per GenEval prompt",
    )
    parser.add_argument(
        "--eval_dpg_max_prompts", type=int, default=0,
        help="Maximum DPG prompts; 0 evaluates the complete official CSV",
    )
    parser.add_argument("--eval_dpg_samples", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument(
        "--eval_workers", type=int, default=0,
        help="Number of training ranks used for generation and scoring; 0 uses all ranks",
    )
    parser.add_argument("--eval_geneval_command", default=None)
    parser.add_argument("--eval_dpg_command", default=None)
    parser.add_argument("--eval_timeout", type=int, default=7200)
    parser.add_argument(
        "--eval_require_metrics", action="store_true",
        help="Require FID, GenEval, and DPG-Bench inputs before training starts",
    )
    args = parser.parse_args(argv)
    if args.eval_cfg is None:
        args.eval_cfg = 4.0 if args.model == "specslot" else 2.9
    if args.bottleneck is None and args.model == "specslot":
        args.bottleneck = 1536
    if args.model == "specslot" and args.jit_variant != "auto":
        raise ValueError("--jit_variant is only valid with --model jit")
    return args


def init_distributed() -> tuple[int, int, int]:
    if not dist.is_available():
        raise RuntimeError("PyTorch distributed is unavailable")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
            raise RuntimeError("train_t2i.py must be launched with torchrun")
        dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size, local_rank


def set_seed(seed: int, rank: int) -> None:
    value = int(seed) + int(rank)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def _import_dependency(module: str) -> None:
    """Import a module, or a named compatibility symbol when requested."""

    if module == "mmdet.apis":
        result = subprocess.run(
            [sys.executable, "-c", "import mmdet.apis"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            detail = (result.stderr or result.stdout).strip().splitlines()
            raise RuntimeError(detail[-1] if detail else f"exit code {result.returncode}")
        return
    if module.endswith("::BasicTokenizer"):
        module_name, symbol = module.split("::", 1)
        getattr(importlib.import_module(module_name), symbol)
    else:
        importlib.import_module(module)


def validate_evaluation_config(args: argparse.Namespace) -> None:
    """Reject a periodic evaluation setup that cannot produce all metrics."""

    if args.eval_interval > 0 and not args.eval_prompt_file and not args.eval_assets_dir:
        raise ValueError(
            "--eval_prompt_file or --eval_assets_dir is required when --eval_interval is positive"
        )
    if args.eval_prompt_file and not Path(args.eval_prompt_file).expanduser().is_file():
        raise FileNotFoundError(
            f"evaluation prompt manifest does not exist: {Path(args.eval_prompt_file).expanduser()}"
        )
    if args.eval_interval <= 0:
        return
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.eval_workers < 0 or args.eval_workers > world_size:
        raise ValueError("--eval_workers must be 0 (all ranks) or between 1 and WORLD_SIZE")
    workers = args.eval_workers or world_size
    if workers > 1 and (args.eval_geneval_command or args.eval_dpg_command):
        raise ValueError(
            "Multi-rank evaluation requires the official evaluators in --eval_assets_dir; "
            "custom benchmark commands require --eval_workers 1"
        )
    if args.eval_reference_dir:
        reference_path = Path(args.eval_reference_dir).expanduser()
        if not reference_path.is_dir():
            raise FileNotFoundError(f"FID reference directory does not exist: {reference_path}")
        if not any(
            path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
            for path in reference_path.rglob("*")
        ):
            raise ValueError(f"FID reference directory contains no images: {reference_path}")
    if args.eval_assets_dir:
        assets = Path(args.eval_assets_dir).expanduser()
        if not assets.is_dir():
            raise FileNotFoundError(f"evaluation assets directory does not exist: {assets}")
        if not args.eval_prompt_file:
            default_prompt = assets / "geneval/prompts/evaluation_metadata.jsonl"
            if not default_prompt.is_file():
                raise FileNotFoundError(f"official GenEval prompt manifest does not exist: {default_prompt}")
        if args.eval_require_metrics:
            for variable in ("GENEVAL_PYTHON", "DPG_PYTHON"):
                executable = os.environ.get(variable)
                if executable and not (Path(executable).is_file() or shutil.which(executable)):
                    raise FileNotFoundError(
                        f"{variable} does not point to an executable: {executable}"
                    )
            required = (
                assets / "geneval/evaluation/evaluate_images.py",
                assets / "geneval_object_detector",
                assets / "ella/dpg_bench/compute_dpg_bench.py",
                assets / "ella/dpg_bench/dpg_bench.csv",
                assets / "modelscope",
            )
            missing = [str(path) for path in required if not path.exists()]
            if missing:
                raise FileNotFoundError("missing evaluation assets: " + ", ".join(missing))
        dependency_modules = {}
        if not os.environ.get("DPG_PYTHON"):
            dependency_modules.update({
                "modelscope": "modelscope",
                "modelscope pipelines": "modelscope.pipelines",
                "modelscope OFA": "modelscope.models.multi_modal.ofa_for_all_tasks",
                "decord": "decord",
                "modelscope multimodal preprocessors": "modelscope.preprocessors.multi_modal",
                "modelscope VQA pipeline": "modelscope.pipelines.multi_modal.visual_question_answering_pipeline",
                "addict": "addict",
                "accelerate": "accelerate",
                "datasets": "datasets",
                "librosa": "librosa",
                "soundfile": "soundfile",
                "rapidfuzz": "rapidfuzz",
                "unicodedata2": "unicodedata2",
                "zhconv": "zhconv",
                "fairseq": "fairseq",
                "cffi": "cffi",
                "cython": "Cython",
                "hydra-core": "hydra",
                "omegaconf": "omegaconf",
                "regex": "regex",
                "sacrebleu": "sacrebleu",
                "bitarray": "bitarray",
                "torchaudio": "torchaudio",
                "tensorboardX": "tensorboardX",
                "BasicTokenizer": "transformers.models.bert.tokenization_bert::BasicTokenizer",
            })
        if not os.environ.get("GENEVAL_PYTHON"):
            dependency_modules.update({
                "pandas": "pandas",
                "open_clip": "open_clip",
                "clip_benchmark": "clip_benchmark",
                "mmdet": "mmdet",
                "mmdet.apis": "mmdet.apis",
                "mmcv": "mmcv",
                "mmcv CUDA ops": "mmcv._ext",
            })
        missing_dependencies = []
        for package, module in dependency_modules.items():
            try:
                _import_dependency(module)
            except Exception as exc:
                missing_dependencies.append(f"{package} ({type(exc).__name__}: {exc})")
        if missing_dependencies:
            command_hint = "uv sync --locked --extra eval"
            if any(
                name.startswith(("mmdet", "mmcv", "mmcv CUDA ops"))
                for name in missing_dependencies
            ):
                command_hint += "; see the T2I evaluation section in README.md for the evaluator setup"
            dpg_missing = [
                name for name in missing_dependencies
                if name.startswith((
                    "modelscope", "modelscope OFA", "decord", "addict", "accelerate", "datasets", "librosa",
                    "soundfile", "rapidfuzz", "unicodedata2", "zhconv", "fairseq", "cffi",
                    "cython", "hydra-core", "omegaconf", "regex", "sacrebleu", "bitarray",
                    "torchaudio", "tensorboardX", "BasicTokenizer",
                ))
            ]
            geneval_missing = [
                name for name in missing_dependencies
                if name.startswith(("pandas", "open_clip", "clip_benchmark", "mmdet", "mmcv"))
            ]
            groups = []
            if dpg_missing:
                groups.append("DPG-Bench: " + ", ".join(dpg_missing))
            if geneval_missing:
                groups.append("GenEval: " + ", ".join(geneval_missing))
            message = (
                "Evaluation dependencies are missing ("
                + "; ".join(groups)
                + "). Run " + command_hint
                + ", or set GENEVAL_PYTHON to a Python executable with the official "
                "mmdetection 2.x/mmcv detector stack."
            )
            if args.eval_require_metrics:
                raise RuntimeError(message)
            print(f"WARNING: {message}", flush=True)
        if not args.eval_geneval_command:
            # Resolve in the scorer's interpreter: wheels and source checkouts
            # store the same detector config in different locations.
            command = [
                os.environ.get("GENEVAL_PYTHON") or sys.executable,
                str(Path(__file__).resolve().parent / "tools/evaluate_geneval.py"),
                "--check-config",
            ]
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
                if result.returncode:
                    raise RuntimeError((result.stderr or result.stdout).strip())
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
                message = f"GenEval detector config preflight failed: {exc}"
                if args.eval_require_metrics:
                    raise RuntimeError(message) from exc
                print(f"WARNING: {message}", flush=True)
    for name, command in (("GenEval", args.eval_geneval_command), ("DPG-Bench", args.eval_dpg_command)):
        if command:
            for token in ("{images}", "{prompts}", "{output}"):
                if token not in command:
                    raise ValueError(f"{name} command must contain {token}")
    if args.eval_require_metrics and not args.eval_assets_dir:
        if not args.eval_geneval_command or not args.eval_dpg_command:
            raise ValueError(
                "--eval_require_metrics needs --eval_assets_dir or both official benchmark commands"
            )
    if args.eval_max_images < 0:
        raise ValueError("--eval_max_images must be non-negative (0 means all prompts)")
    if args.eval_dpg_max_prompts < 0:
        raise ValueError("--eval_dpg_max_prompts must be non-negative (0 means all prompts)")
    if args.eval_steps < 2 or args.eval_batch_size < 1 or args.eval_timeout < 1:
        raise ValueError("evaluation requires steps >= 2 and positive batch size/timeout")
    if args.eval_num_samples < 1 or args.eval_dpg_samples != 4:
        raise ValueError("--eval_num_samples must be positive and --eval_dpg_samples must equal 4")


def save_checkpoint(
    path: Path,
    model: DDP,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    args: argparse.Namespace,
    ema_state: dict[str, torch.Tensor],
) -> None:
    state = {
        "model": model.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": step,
        "ema": {name: value.detach().cpu() for name, value in ema_state.items()},
        "config": vars(args),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_evaluation_plan(args: argparse.Namespace, completed_epoch: int) -> tuple[list, list, dict]:
    prompt_file = args.eval_prompt_file
    if prompt_file is None and args.eval_assets_dir:
        prompt_file = str(Path(args.eval_assets_dir).expanduser()
                          / "geneval/prompts/evaluation_metadata.jsonl")
    if prompt_file is None:
        raise RuntimeError("no GenEval prompt manifest was configured")
    prompts = [record for record in load_prompt_manifest(prompt_file) if record["prompt"].strip()]
    total_prompts = len(prompts)
    if args.eval_max_images > 0:
        prompts = prompts[:args.eval_max_images]
    if not prompts:
        raise ValueError("no GenEval prompts available")
    dpg_prompts = []
    if args.eval_assets_dir:
        dpg_csv = Path(args.eval_assets_dir).expanduser() / "ella/dpg_bench/dpg_bench.csv"
        dpg_prompts = load_dpg_prompt_manifest(dpg_csv, max_prompts=args.eval_dpg_max_prompts)
    manifest_hash = hashlib.sha256(json.dumps(
        {"geneval": prompts, "dpg_bench": dpg_prompts},
        sort_keys=True, ensure_ascii=False,
    ).encode("utf-8")).hexdigest()
    config = {
        "epoch": completed_epoch, "model": args.model,
        "jit_variant": args.jit_variant,
        "image_size": args.image_size, "bottleneck": args.bottleneck,
        "prompt_file": str(Path(prompt_file).resolve()),
        "eval_assets_dir": str(Path(args.eval_assets_dir).expanduser().resolve())
            if args.eval_assets_dir else None,
        "gen_eval_prompts": len(prompts), "gen_eval_total_prompts": total_prompts,
        "gen_eval_complete": len(prompts) == total_prompts,
        "gen_eval_samples_per_prompt": args.eval_num_samples,
        "dpg_prompts": len(dpg_prompts), "dpg_max_prompts": args.eval_dpg_max_prompts,
        "dpg_samples_per_prompt": args.eval_dpg_samples,
        "steps": args.eval_steps, "cfg_scale": args.eval_cfg,
        "cfg_protocol": "fixed_per_model", "seed": args.seed,
        "manifest_sha256": manifest_hash,
    }
    return prompts, dpg_prompts, config


def evaluate_epoch(
    args: argparse.Namespace,
    network: torch.nn.Module,
    text_encoder: torch.nn.Module,
    ema_state: dict[str, torch.Tensor],
    eval_root: Path,
    device: torch.device,
    completed_epoch: int,
    *,
    shard_rank: int | None = None,
    shard_count: int = 1,
) -> dict:
    """Evaluate EMA weights while preserving the training state and RNG."""

    started = time.perf_counter()
    parallel = shard_rank is not None
    if parallel:
        eval_root = eval_root / "shards" / f"rank-{shard_rank:03d}"
    eval_root.mkdir(parents=True, exist_ok=True)
    for stale in (
        eval_root / "generated",
        eval_root / "generated_geneval",
        eval_root / "generated_dpg",
        eval_root / "generated_dpg_prompts.jsonl",
    ):
        if stale.is_symlink() or stale.is_file():
            stale.unlink()
        elif stale.is_dir():
            shutil.rmtree(stale)
    prompts, dpg_prompts, config = load_evaluation_plan(args, completed_epoch)
    architecture_contract = getattr(network, "architecture_contract", None)
    if architecture_contract is not None:
        config["architecture_contract"] = architecture_contract
    geneval_indices = dpg_indices = None
    if parallel:
        geneval_indices = shard_prompt_indices(
            len(prompts), shard_rank, shard_count, args.eval_batch_size,
        )
        dpg_indices = shard_prompt_indices(
            len(dpg_prompts), shard_rank, shard_count,
            max(1, args.eval_batch_size // args.eval_dpg_samples),
        )
        prompts = [prompts[index] for index in geneval_indices]
        dpg_prompts = [dpg_prompts[index] for index in dpg_indices]
    rank_label = shard_rank if parallel else 0
    print(f"[Eval][rank {rank_label}] generation: GenEval={len(prompts)} prompts, "
          f"DPG={len(dpg_prompts)} prompts", flush=True)
    timings = {}

    parameters = dict(network.named_parameters())
    original_parameters = {}
    was_training = network.training
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] \
        if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            with torch.no_grad():
                for name, parameter in parameters.items():
                    if name in ema_state:
                        original_parameters[name] = parameter.detach().clone()
                        parameter.copy_(ema_state[name].to(parameter))
            phase_started = time.perf_counter()
            generated = eval_root / "generated"
            geneval_dir = eval_root / "generated_geneval"
            if prompts:
                generate_t2i(
                    network, text_encoder, prompts, generated,
                    image_size=args.image_size, steps=args.eval_steps,
                    cfg_scale=args.eval_cfg, seed=args.seed,
                    batch_size=args.eval_batch_size,
                    samples_per_prompt=args.eval_num_samples,
                    prompt_indices=geneval_indices,
                )
            else:
                generated.mkdir(parents=True, exist_ok=True)
                geneval_dir.mkdir(parents=True, exist_ok=True)
                (generated / "prompts.jsonl").write_text("", encoding="utf-8")
            timings["geneval_generation_seconds"] = time.perf_counter() - phase_started
            phase_started = time.perf_counter()
            dpg_generated = eval_root / "generated_dpg"
            if dpg_prompts:
                generate_dpg_t2i(
                    network, text_encoder, dpg_prompts, dpg_generated,
                    image_size=args.image_size, steps=args.eval_steps,
                    cfg_scale=args.eval_cfg, seed=args.seed,
                    batch_size=max(1, args.eval_batch_size // args.eval_dpg_samples),
                    samples_per_prompt=args.eval_dpg_samples,
                    prompt_indices=dpg_indices,
                )
            elif parallel:
                dpg_generated.mkdir(parents=True, exist_ok=True)
                (eval_root / "generated_dpg_prompts.jsonl").write_text("", encoding="utf-8")
            else:
                dpg_generated = None
            timings["dpg_generation_seconds"] = time.perf_counter() - phase_started
            # Release temporary generation allocations before each GPU loads
            # its detector/VQA subprocess. Training weights and optimizer stay.
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
            print(f"[Eval][rank {rank_label}] generation complete; scoring on {device}", flush=True)
            phase_started = time.perf_counter()
            metrics = evaluate_generated(
                generated,
                reference_dir=None if parallel else args.eval_reference_dir,
                geneval_command=args.eval_geneval_command,
                dpg_command=args.eval_dpg_command,
                eval_assets_dir=args.eval_assets_dir,
                geneval_image_dir=geneval_dir,
                dpg_image_dir=dpg_generated,
                image_size=args.image_size, device=device,
                benchmark_timeout=args.eval_timeout,
                skip_empty=parallel,
            )
            timings["scoring_seconds"] = time.perf_counter() - phase_started
    finally:
        with torch.no_grad():
            for name, value in original_parameters.items():
                parameters[name].copy_(value)
        network.train(was_training)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)

    metrics["evaluation_config"] = config
    metrics["rank"] = rank_label
    metrics.setdefault("timings", {}).update(timings)
    metrics["timings"]["total_seconds"] = time.perf_counter() - started
    if parallel:
        metrics["assigned_geneval_indices"] = geneval_indices
        metrics["assigned_dpg_indices"] = dpg_indices
        (eval_root / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"[Eval][rank {rank_label}] complete in {metrics['timings']['total_seconds']:.1f}s", flush=True)
    return metrics


def merge_epoch_evaluation(args, eval_root, shard_results, device, completed_epoch):
    prompts, dpg_prompts, config = load_evaluation_plan(args, completed_epoch)
    # Workers have joined the CPU gather. Only the merged views are replaced;
    # rank-local images and diagnostic files remain available for inspection.
    for stale in (
        eval_root / "generated", eval_root / "generated_geneval",
        eval_root / "generated_dpg", eval_root / "generated_dpg_prompts.jsonl",
    ):
        if stale.is_symlink() or stale.is_file():
            stale.unlink()
        elif stale.is_dir():
            shutil.rmtree(stale)
    for rank, result in enumerate(shard_results):
        recorded_hash = result.get("evaluation_config", {}).get("manifest_sha256")
        if recorded_hash is not None and recorded_hash != config["manifest_sha256"]:
            raise RuntimeError(f"Evaluation manifest changed during rank {rank}'s work")
        result.setdefault("generated_dir", str(eval_root / "shards" / f"rank-{rank:03d}" / "generated"))
    metrics = merge_evaluation_shards(
        eval_root, shard_results, prompts, dpg_prompts, args.eval_num_samples,
        Path(args.eval_assets_dir).expanduser() if args.eval_assets_dir else None,
    )
    metrics["evaluation_config"] = {**config, "eval_workers": len(shard_results)}
    if args.eval_reference_dir and metrics.get("generation", {}).get("status") == "ok":
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] \
            if device.type == "cuda" else []
        try:
            with torch.random.fork_rng(devices=cuda_devices):
                metrics["fid_kid"] = compute_fid_kid(
                    metrics["generated_dir"], args.eval_reference_dir, device=device,
                )
        finally:
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
    elif not args.eval_reference_dir:
        metrics["fid_kid"] = {"status": "not_run", "reason": "reference_dir_not_provided"}
    return metrics


def coordinate_evaluation(
    evaluation: Callable[[], dict],
    eval_root: Path,
    *,
    process_group=None,
    evaluation_config: dict | None = None,
    run_logger: T2IRunLogger | None = None,
    global_step: int | None = None,
    all_ranks: bool = False,
    merge_results: Callable[[list[dict]], dict] | None = None,
) -> dict:
    """Release every rank after evaluation, including reportable failures."""

    rank = dist.get_rank() if dist.is_initialized() else 0
    if all_ranks and merge_results is None:
        raise ValueError("all-rank evaluation requires a result merger")
    if dist.is_initialized() and dist.get_backend(process_group) != "gloo":
        raise ValueError("Evaluation coordination requires a Gloo control group")
    outcome = [None]
    started = time.perf_counter()
    if rank == 0 or all_ranks:
        try:
            metrics = evaluation()
            if not isinstance(metrics, dict):
                raise TypeError("Evaluation must return a metrics dictionary")
        except Exception as exc:
            metrics = {
                "status": "error", "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "evaluation_config": evaluation_config or {},
            }
        metrics["rank"] = rank
    if all_ranks:
        shard_results = [None] * (dist.get_world_size() if dist.is_initialized() else 1)
        if dist.is_initialized():
            dist.all_gather_object(shard_results, metrics, group=process_group)
        else:
            shard_results[0] = metrics
        if rank == 0:
            try:
                metrics = merge_results(shard_results)
                if not isinstance(metrics, dict):
                    raise TypeError("Evaluation merger must return a metrics dictionary")
            except Exception as exc:
                metrics = {
                    "status": "error", "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(), "shards": shard_results,
                    "evaluation_config": evaluation_config or {},
                }
    if rank == 0:
        failed = metrics.get("status") == "error" or any(
            value.get("status") == "error" for value in metrics.values()
            if isinstance(value, dict)
        )
        metrics["status"] = "error" if failed else "ok"
        metrics.setdefault("timings", {})["wall_seconds"] = time.perf_counter() - started
        if global_step is not None:
            metrics["global_step"] = int(global_step)
        if run_logger is not None and global_step is not None:
            try:
                run_logger.log_evaluation(metrics, global_step)
            except Exception as exc:
                metrics["status"] = "error"
                metrics["logging_error"] = f"{type(exc).__name__}: {exc}"
        try:
            message = f"Evaluation {eval_root.name}: {json.dumps(metrics, sort_keys=True, default=str)}"
            if run_logger is not None:
                run_logger.info(message)
            else:
                print(message, flush=True)
            if metrics["status"] == "error":
                print("Evaluation failed; recorded available diagnostics and continuing training.", flush=True)
        except Exception as exc:
            metrics["status"] = "error"
            metrics["logging_error"] = f"{type(exc).__name__}: {exc}"
        try:
            eval_root.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
            for directory in (eval_root, eval_root / "generated"):
                directory.mkdir(parents=True, exist_ok=True)
                temporary = directory / "metrics.json.tmp"
                temporary.write_text(payload, encoding="utf-8")
                temporary.replace(directory / "metrics.json")
        except Exception as exc:
            metrics["status"] = "error"
            metrics["write_error"] = f"{type(exc).__name__}: {exc}"
        outcome[0] = {"status": metrics["status"], "metrics_path": str(eval_root / "metrics.json")}
    if dist.is_initialized():
        # This CPU collective uses the long-timeout control group. Waiting
        # ranks never enter an NCCL collective while rank 0 runs benchmarks.
        dist.broadcast_object_list(outcome, src=0, group=process_group)
    return outcome[0]


def main() -> None:
    args = parse_args()
    validate_evaluation_config(args)
    if args.model == "jit" and args.bottleneck is not None:
        raise ValueError("--bottleneck is only valid for SpecSlot")
    if args.steps_per_epoch <= 0 or args.epochs <= 0:
        raise ValueError("--epochs and --steps_per_epoch must be positive")
    if args.log_every < 0 or args.save_every < 1:
        raise ValueError("--log_every must be non-negative and --save_every must be positive")
    if not 0.0 <= args.caption_drop_prob <= 1.0:
        raise ValueError("--caption_drop_prob must be in [0,1]")
    if not 0.0 < args.ema_decay < 1.0:
        raise ValueError("--ema_decay must be in (0,1)")

    rank, world_size, local_rank = init_distributed()
    eval_workers = args.eval_workers or world_size
    evaluation_group = None
    if args.eval_interval > 0 and (args.eval_prompt_file or args.eval_assets_dir):
        evaluation_group = dist.new_group(
            backend="gloo",
            timeout=timedelta(seconds=max(86400, 2 * args.eval_timeout + 3600)),
        )
    device = torch.device("cuda", local_rank)
    set_seed(args.seed, rank)
    output_dir = Path(args.output_dir).expanduser().resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(vars(args), handle, indent=2, sort_keys=True)
    dist.barrier(device_ids=[local_rank])

    dataset = T2IShardDataset(
        args.data_root,
        image_size=args.image_size,
        shuffle_buffer=args.shuffle_buffer,
        shard_shuffle_buffer=args.shard_shuffle_buffer,
        seed=args.seed,
    )
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_t2i,
    )
    if args.num_workers > 0:
        loader_kwargs.update(
            persistent_workers=True,
            prefetch_factor=2,
            multiprocessing_context="spawn",
        )
    loader = DataLoader(dataset, **loader_kwargs)
    iterator = iter(loader)

    text_encoder = FrozenOpenCLIPTextEncoder(
        model_name=args.text_model,
        pretrained=args.text_pretrained,
        device=device,
    )
    model = TextConditionedDenoiser(
        args.model,
        args.image_size,
        bottleneck=args.bottleneck or 1536,
        activation_checkpointing=args.activation_checkpointing,
        jit_variant=args.jit_variant,
    ).to(device)
    architecture_contract = model.architecture_contract
    if rank == 0:
        with (output_dir / "architecture_contract.json").open("w", encoding="utf-8") as handle:
            json.dump(architecture_contract, handle, indent=2, sort_keys=True)
    # The class table is not part of the text-conditioned contract.
    model.backbone.y_embedder.embedding_table.weight.requires_grad_(False)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    optimizer = torch.optim.AdamW(
        add_weight_decay(model, args.weight_decay),
        lr=args.lr,
        betas=(0.9, 0.95),
    )
    ema_state = {
        name: parameter.detach().float().clone()
        for name, parameter in model.module.named_parameters()
        if parameter.requires_grad and parameter.is_floating_point()
    }

    start_epoch = 0
    global_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.module.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "ema" in checkpoint:
            ema_state = {
                name: value.detach().to(device=device, dtype=torch.float32)
                for name, value in checkpoint["ema"].items()
            }
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        global_step = int(checkpoint.get("global_step", 0))
        del checkpoint
    null_states, null_mask = text_encoder([""])
    null_states = null_states[0].detach()
    null_mask = null_mask[0].detach()
    autocast_dtype = torch.bfloat16

    run_logger = T2IRunLogger(
        output_dir, resume_step=global_step if args.resume else None,
    ) if rank == 0 else None
    if run_logger is not None:
        if args.model == "specslot":
            model_label = "SpecSlot"
        elif args.jit_variant != "auto":
            model_label = f"JiT-B/{args.jit_variant[1:]}"
        else:
            model_label = "JiT-B/8"
        run_logger.info(
            f"Training {model_label} T2I at {args.image_size}px: "
            f"shards={len(dataset.shards)} world_size={world_size} "
            f"batch/GPU={args.batch_size} steps/epoch={args.steps_per_epoch} "
            f"start_epoch={start_epoch} global_step={global_step}",
        )
    try:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            epoch_start = time.time()
            loss_sum = 0.0
            window_loss = 0.0
            window_steps = 0
            for local_step in range(args.steps_per_epoch):
                batch = next(iterator)
                images = batch["images"].to(device, non_blocking=True)
                captions = batch["captions"]
                with torch.no_grad():
                    text_states, text_mask = text_encoder(captions)
                    dropped = torch.rand(images.shape[0], device=device) < args.caption_drop_prob
                    expanded_null_states = null_states.unsqueeze(0).expand_as(text_states)
                    expanded_null_mask = null_mask.unsqueeze(0).expand_as(text_mask)
                    text_states = torch.where(
                        dropped[:, None, None], expanded_null_states, text_states
                    )
                    text_mask = torch.where(
                        dropped[:, None], expanded_null_mask, text_mask
                    )
                    normal = torch.randn(images.shape[0], device=device)
                    timestep = torch.sigmoid(args.P_mean + args.P_std * normal)
                    noise = torch.randn_like(images)
                    state = timestep[:, None, None, None] * images + (
                        1.0 - timestep[:, None, None, None]
                    ) * noise

                optimizer.zero_grad(set_to_none=True)
                progress = (epoch * args.steps_per_epoch + local_step) / args.steps_per_epoch
                if progress < args.warmup_epochs:
                    learning_rate = args.lr * progress / max(1, args.warmup_epochs)
                elif args.lr_schedule == "cosine" and args.epochs > args.warmup_epochs:
                    cosine_progress = min(
                        1.0,
                        (progress - args.warmup_epochs)
                        / max(1, args.epochs - args.warmup_epochs),
                    )
                    learning_rate = args.min_lr + (args.lr - args.min_lr) * 0.5 * (
                        1.0 + np.cos(np.pi * cosine_progress)
                    )
                else:
                    learning_rate = args.lr
                for group in optimizer.param_groups:
                    group["lr"] = float(learning_rate)
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    prediction = model(state, timestep, text_states, text_mask)
                    velocity_prediction = (
                        prediction.float() - state.float()
                    ) / (1.0 - timestep[:, None, None, None]).clamp_min(args.t_eps)
                    target_velocity = images.float() - noise.float()
                    loss = (velocity_prediction - target_velocity).square().mean()
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                with torch.no_grad():
                    for name, parameter in model.module.named_parameters():
                        if name in ema_state and parameter.requires_grad and parameter.is_floating_point():
                            ema_state[name].mul_(args.ema_decay).add_(
                                parameter.detach(), alpha=1.0 - args.ema_decay
                            )
                global_step += 1
                loss_value = float(loss.detach().item())
                loss_sum += loss_value
                window_loss += loss_value
                window_steps += 1
                if args.log_every > 0 and (
                    (local_step + 1) % args.log_every == 0
                    or local_step + 1 == args.steps_per_epoch
                ):
                    mean_loss = reduce_loss(window_loss, window_steps, device)
                    if run_logger is not None:
                        run_logger.log_step(
                            epoch=epoch, epoch_step=local_step + 1,
                            steps_per_epoch=args.steps_per_epoch, global_step=global_step,
                            loss=mean_loss, lr=float(learning_rate),
                        )
                    window_loss, window_steps = 0.0, 0

            epoch_totals = torch.tensor(
                [loss_sum, float(args.steps_per_epoch)],
                device=device,
                dtype=torch.float64,
            )
            dist.all_reduce(epoch_totals, op=dist.ReduceOp.SUM)
            dist.barrier(device_ids=[local_rank])
            if rank == 0:
                record = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "loss": float((epoch_totals[0] / epoch_totals[1]).item()),
                    "seconds": time.time() - epoch_start,
                }
                assert run_logger is not None
                run_logger.log_epoch(record)
                if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
                    save_checkpoint(
                        output_dir / "checkpoint-last.pth",
                        model,
                        optimizer,
                        epoch,
                        global_step,
                        args,
                        ema_state,
                    )
                    save_checkpoint(
                        output_dir / f"checkpoint-{epoch + 1}.pth",
                        model,
                        optimizer,
                        epoch,
                        global_step,
                        args,
                        ema_state,
                    )
            dist.barrier(device_ids=[local_rank])
            if (
                args.eval_interval > 0
                and (epoch + 1) % args.eval_interval == 0
                and (args.eval_prompt_file or args.eval_assets_dir)
            ):
                eval_root = output_dir / "evaluations" / f"epoch-{epoch + 1:04d}"
                optimizer.zero_grad(set_to_none=True)
                parallel_eval = eval_workers > 1
                coordinate_evaluation(
                    lambda: evaluate_epoch(
                        args, model.module, text_encoder, ema_state,
                        eval_root, device, epoch + 1,
                        shard_rank=rank if parallel_eval else None,
                        shard_count=eval_workers,
                    ) if rank < eval_workers else {"status": "not_assigned", "rank": rank},
                    eval_root,
                    process_group=evaluation_group,
                    evaluation_config={
                        "epoch": epoch + 1, "model": args.model,
                        "jit_variant": args.jit_variant,
                        "image_size": args.image_size, "bottleneck": args.bottleneck,
                        "architecture_contract": architecture_contract,
                        "cfg_scale": args.eval_cfg, "cfg_protocol": "fixed_per_model",
                    },
                    run_logger=run_logger,
                    global_step=global_step,
                    all_ranks=parallel_eval,
                    merge_results=(lambda results: merge_epoch_evaluation(
                        args, eval_root, results[:eval_workers], device, epoch + 1,
                    )) if parallel_eval else None,
                )
    except BaseException:
        if run_logger is not None:
            run_logger.exception("Training interrupted or failed")
        raise
    finally:
        if run_logger is not None:
            run_logger.close()
        if evaluation_group is not None:
            dist.destroy_process_group(evaluation_group)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
