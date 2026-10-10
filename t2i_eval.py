"""Sampling and official benchmark evaluation for text-conditioned models."""

from __future__ import annotations

import json
import hashlib
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
from PIL import Image

from t2i_dpg import load_dpg_data


T2I_IMAGE_SIZE = 256


def _require_t2i_image_size(image_size: int) -> None:
    if int(image_size) != T2I_IMAGE_SIZE:
        raise ValueError(
            f"The released T2I protocol uses {T2I_IMAGE_SIZE}-pixel images, "
            f"got {image_size}"
        )


@dataclass(frozen=True)
class EvaluationConfig:
    """Sampling and evaluation settings recorded in ``metrics.json``."""

    image_size: int
    steps: int = 50
    cfg_scale: float = 5.0
    t_eps: float = 0.05
    seed: int = 0
    batch_size: int = 16
    max_images: int = 0
    autocast_dtype: str | None = "bfloat16"


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module without invoking a DDP wrapper on rank 0."""

    return getattr(model, "module", model)


def _autocast_context(device: torch.device, dtype_name: str | None):
    if device.type != "cuda" or dtype_name is None:
        return torch.autocast(device_type="cpu", enabled=False)
    dtype = getattr(torch, dtype_name)
    return torch.autocast(device_type="cuda", dtype=dtype)


def _x_to_velocity(
    clean_prediction: torch.Tensor,
    state: torch.Tensor,
    timestep: torch.Tensor,
    t_eps: float,
) -> torch.Tensor:
    denominator = (1.0 - timestep).clamp_min(float(t_eps))
    return (clean_prediction.float() - state.float()) / denominator[:, None, None, None]


@torch.no_grad()
def heun_sample(
    model: torch.nn.Module,
    text_states: torch.Tensor,
    text_mask: torch.Tensor | None,
    null_text_states: torch.Tensor | None,
    null_text_mask: torch.Tensor | None,
    *,
    image_size: int,
    steps: int = 50,
    cfg_scale: float = 5.0,
    t_eps: float = 0.05,
    initial_noise: torch.Tensor | None = None,
    autocast_dtype: str | None = "bfloat16",
) -> torch.Tensor:
    """Sample a batch with clean-image prediction and Heun integration.

    This follows the repository's endpoint-aware schedule: ``steps-1`` Heun
    intervals run through ``1-t_eps`` and the final interval is Euler to
    ``t=1``.  Consequently a nominal 50-step rollout has 99 velocity
    evaluations, or 198 conditional/unconditional model forwards when
    ``cfg_scale != 1``, exactly as the source ImageNet sampler.  ``text_states`` and
    ``null_text_states`` are the frozen text-encoder outputs; the model's
    trainable 32-token adapter is applied in its normal forward path.
    """

    _require_t2i_image_size(image_size)
    if steps < 2:
        raise ValueError(f"steps must be at least 2, got {steps}")
    if not 0.0 < t_eps < 1.0:
        raise ValueError(f"t_eps must lie in (0,1), got {t_eps}")
    if text_states.ndim != 3:
        raise ValueError(f"text_states must be [B,L,D], got {tuple(text_states.shape)}")
    device = text_states.device
    batch = text_states.shape[0]
    expected_shape = (batch, 3, image_size, image_size)
    if initial_noise is None:
        noise = torch.randn(expected_shape, device=device)
    else:
        if tuple(initial_noise.shape) != expected_shape or initial_noise.device != device:
            raise ValueError(
                f"initial_noise must have shape {expected_shape} on {device}, "
                f"got {tuple(initial_noise.shape)} on {initial_noise.device}"
            )
        noise = initial_noise
    state = noise.float()
    network = _unwrap_model(model)
    network_was_training = network.training
    network.eval()

    # Source endpoint-aware x-prediction schedule.  The final endpoint is
    # handled by a single Euler step after the Heun loop.
    learned_grid = torch.linspace(
        0.0, 1.0 - float(t_eps), steps, device=device, dtype=torch.float32
    )
    grid = torch.cat([learned_grid, torch.ones(1, device=device)])

    def velocity(z: torch.Tensor, t_value: torch.Tensor) -> torch.Tensor:
        with _autocast_context(device, autocast_dtype):
            conditional_x = network(z, t_value, text_states, text_mask)
            conditional_v = _x_to_velocity(conditional_x, z, t_value, t_eps)
            if cfg_scale == 1.0:
                return conditional_v
            if null_text_states is None:
                raise ValueError("null_text_states is required when cfg_scale != 1")
            unconditional_x = network(
                z, t_value, null_text_states,
                null_text_mask if null_text_mask is not None else text_mask,
            )
            unconditional_v = _x_to_velocity(unconditional_x, z, t_value, t_eps)
        return unconditional_v + float(cfg_scale) * (conditional_v - unconditional_v)

    for index in range(steps - 1):
        t = grid[index].expand(batch)
        t_next = grid[index + 1].expand(batch)
        delta = t_next - t
        current_v = velocity(state, t)
        euler_state = state + delta[:, None, None, None] * current_v
        next_v = velocity(euler_state, t_next)
        state = state + delta[:, None, None, None] * 0.5 * (current_v + next_v)

    # Keep the final endpoint behavior identical to the source sampler.
    t = grid[-2].expand(batch)
    t_next = grid[-1].expand(batch)
    state = state + (t_next - t)[:, None, None, None] * velocity(state, t)
    if network_was_training:
        network.train()
    return state.clamp(-1.0, 1.0)


def load_prompt_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSON/JSONL prompt manifest with stable ``id`` and ``prompt`` fields."""

    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    else:
        value = json.loads(source.read_text())
        records = value if isinstance(value, list) else value.get("prompts", value.get("data", []))
    result: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if isinstance(record, str):
            normalized = {"id": str(index), "prompt": record}
        elif isinstance(record, Mapping):
            prompt = record.get("prompt", record.get("caption", record.get("text")))
            sample_id = record.get("id", record.get("sample_id", index))
            normalized = dict(record)
            normalized.update({"id": str(sample_id), "prompt": prompt})
        else:
            raise ValueError(f"manifest record {index} is neither text nor object")
        if not isinstance(normalized["prompt"], str) or not normalized["prompt"].strip():
            raise ValueError(f"manifest record {index} has an empty prompt")
        result.append(normalized)
    if not result:
        raise ValueError(f"prompt manifest is empty: {source}")
    return result


def _write_image_batch(images: torch.Tensor, indices: Sequence[int], output_dir: Path) -> None:
    images = ((images.float().cpu().clamp(-1, 1) + 1.0) * 127.5).round().byte()
    for image, index in zip(images, indices):
        array = image.permute(1, 2, 0).numpy()
        Image.fromarray(array, mode="RGB").save(output_dir / f"{int(index):06d}.png")


def _write_geneval_sample(
    image: torch.Tensor,
    index: int,
    record: Mapping[str, Any],
    output_dir: Path,
    sample_index: int = 0,
) -> None:
    """Write a sample in the per-prompt layout expected by official GenEval."""

    prompt_dir = output_dir / f"{int(index):05d}"
    sample_dir = prompt_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    image = ((image.float().cpu().clamp(-1, 1) + 1.0) * 127.5).round().byte()
    array = image.permute(1, 2, 0).numpy()
    Image.fromarray(array, mode="RGB").save(sample_dir / f"{int(sample_index):04d}.png")
    (prompt_dir / "metadata.jsonl").write_text(
        json.dumps(dict(record), ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _validate_prompt_indices(count: int, prompt_indices: Sequence[int] | None) -> list[int]:
    if prompt_indices is None:
        return list(range(count))
    indices = list(prompt_indices)
    if len(indices) != count:
        raise ValueError("prompt_indices must have one global index per prompt")
    if any(isinstance(index, bool) or not isinstance(index, Integral) or index < 0 for index in indices):
        raise ValueError("prompt_indices must contain non-negative integers")
    if len(set(indices)) != len(indices):
        raise ValueError("prompt_indices must be unique")
    return [int(index) for index in indices]


def shard_prompt_indices(
    count: int,
    rank: int,
    world_size: int,
    prompts_per_batch: int,
) -> list[int]:
    """Assign complete global prompt batches to ranks in round-robin order.

    Preserving the original batch grouping keeps a prompt's model execution
    shape independent of the number of evaluation workers. Empty ranks return
    an empty list and should skip generation for that benchmark.
    """

    values = (count, rank, world_size, prompts_per_batch)
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in values):
        raise ValueError("shard arguments must be integers")
    if count < 0 or world_size < 1 or prompts_per_batch < 1 or not 0 <= rank < world_size:
        raise ValueError("invalid count, rank, world_size, or prompts_per_batch")
    return [
        index
        for start in range(rank * prompts_per_batch, count, world_size * prompts_per_batch)
        for index in range(start, min(start + prompts_per_batch, count))
    ]


@torch.no_grad()
def _sample_records(
    model: torch.nn.Module,
    text_encoder: torch.nn.Module,
    records: Sequence[Mapping[str, Any]],
    *,
    image_size: int,
    steps: int,
    cfg_scale: float,
    t_eps: float,
    seed: int,
    batch_size: int,
    samples_per_record: int,
    autocast_dtype: str | None,
    prompt_indices: Sequence[int] | None = None,
) -> Iterator[torch.Tensor]:
    """Yield samples in prompt order without retaining the full evaluation set."""

    _require_t2i_image_size(image_size)
    if samples_per_record < 1:
        raise ValueError(f"samples_per_record must be positive, got {samples_per_record}")
    global_indices = _validate_prompt_indices(len(records), prompt_indices)
    device = next(_unwrap_model(model).parameters()).device
    null_states, null_mask = text_encoder([""])
    null_states = null_states[:1].detach()
    null_mask = null_mask[:1].detach()
    expanded_batch = max(1, int(batch_size))
    for start in range(0, len(records), expanded_batch):
        prompt_records = records[start : start + expanded_batch]
        captions: list[str] = []
        seeds: list[int] = []
        for local_index, record in enumerate(prompt_records, start=start):
            prompt_index = global_indices[local_index]
            captions.extend([str(record["prompt"])] * samples_per_record)
            seeds.extend(
                int(seed) + prompt_index * samples_per_record + sample_index
                for sample_index in range(samples_per_record)
            )
        text_states, text_mask = text_encoder(captions)
        text_states = text_states.to(device)
        text_mask = text_mask.to(device)
        noise = torch.stack([
            torch.randn(
                3, image_size, image_size,
                device=device,
                generator=torch.Generator(device=device).manual_seed(sample_seed),
            )
            for sample_seed in seeds
        ])
        sampled = heun_sample(
            model,
            text_states,
            text_mask,
            null_states.expand(len(captions), -1, -1),
            null_mask.expand(len(captions), -1),
            image_size=image_size,
            steps=steps,
            cfg_scale=cfg_scale,
            t_eps=t_eps,
            initial_noise=noise,
            autocast_dtype=autocast_dtype,
        )
        for sample in sampled:
            yield sample.detach().cpu()


@torch.no_grad()
def generate_t2i(
    model: torch.nn.Module,
    text_encoder: torch.nn.Module,
    prompts: Sequence[Mapping[str, Any] | str],
    output_dir: str | Path,
    *,
    image_size: int,
    steps: int = 50,
    cfg_scale: float = 5.0,
    t_eps: float = 0.05,
    seed: int = 0,
    batch_size: int = 16,
    max_images: int | None = None,
    samples_per_prompt: int = 1,
    autocast_dtype: str | None = "bfloat16",
    prompt_indices: Sequence[int] | None = None,
) -> Path:
    """Generate a deterministic prompt-aligned image set and manifest.

    ``prompt_indices`` associates each supplied prompt with its original global
    position, preserving seeds and filenames when generating a local shard.
    """

    _require_t2i_image_size(image_size)
    global_indices = _validate_prompt_indices(len(prompts), prompt_indices)
    normalized = []
    for index, item in zip(global_indices, prompts):
        if isinstance(item, str):
            normalized.append({"id": str(index), "prompt": item})
        else:
            record = dict(item)
            record.update({
                "id": str(item.get("id", item.get("sample_id", index))),
                "prompt": str(item.get("prompt", item.get("caption", item.get("text", "")))),
            })
            normalized.append(record)
    indexed = [(index, item) for index, item in zip(global_indices, normalized) if item["prompt"].strip()]
    global_indices = [index for index, _ in indexed]
    normalized = [item for _, item in indexed]
    if prompt_indices is None:
        global_indices = list(range(len(normalized)))
    if max_images is not None and int(max_images) > 0:
        normalized = normalized[: int(max_images)]
        global_indices = global_indices[: int(max_images)]
    if not normalized:
        raise ValueError("no prompts available for generation")

    if samples_per_prompt < 1:
        raise ValueError(f"samples_per_prompt must be positive, got {samples_per_prompt}")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest = []
    geneval_root = root.with_name(root.name + "_geneval")
    images = _sample_records(
        model,
        text_encoder,
        normalized,
        image_size=image_size,
        steps=steps,
        cfg_scale=cfg_scale,
        t_eps=t_eps,
        seed=seed,
        batch_size=batch_size,
        samples_per_record=samples_per_prompt,
        autocast_dtype=autocast_dtype,
        prompt_indices=global_indices,
    )
    for prompt_index, record in zip(global_indices, normalized):
        for sample_index in range(samples_per_prompt):
            flat_index = prompt_index * samples_per_prompt + sample_index
            sample = next(images)
            _write_image_batch(sample.unsqueeze(0), [flat_index], root)
            _write_geneval_sample(sample, prompt_index, record, geneval_root, sample_index)
            manifest.append({
                **record,
                "index": flat_index,
                "prompt_index": prompt_index,
                "sample_index": sample_index,
                "seed": int(seed) + flat_index,
            })
    (root / "prompts.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in manifest),
        encoding="utf-8",
    )
    return root


def load_dpg_prompt_manifest(csv_path: str | Path, max_prompts: int | None = None) -> list[dict[str, str]]:
    """Read one prompt per item without dropping any question or prompt."""

    records, _ = load_dpg_data(csv_path)
    if max_prompts is not None:
        if int(max_prompts) < 0:
            raise ValueError("max_prompts must be non-negative (0 means all prompts)")
        if int(max_prompts) > 0:
            records = records[:int(max_prompts)]
    return records


@torch.no_grad()
def generate_dpg_t2i(
    model: torch.nn.Module,
    text_encoder: torch.nn.Module,
    prompts: Sequence[Mapping[str, Any] | str],
    output_dir: str | Path,
    *,
    image_size: int,
    steps: int = 50,
    cfg_scale: float = 5.0,
    t_eps: float = 0.05,
    seed: int = 0,
    batch_size: int = 1,
    samples_per_prompt: int = 4,
    max_prompts: int | None = None,
    autocast_dtype: str | None = "bfloat16",
    prompt_indices: Sequence[int] | None = None,
) -> Path:
    """Generate DPG's required four samples and save each prompt as a 2x2 grid.

    ``prompt_indices`` preserves global seeds and manifest positions for shards;
    grid filenames always use the benchmark's original item IDs.
    """

    _require_t2i_image_size(image_size)
    if samples_per_prompt != 4:
        raise ValueError("DPG-Bench requires exactly four samples per prompt")
    global_indices = _validate_prompt_indices(len(prompts), prompt_indices)
    normalized = []
    retained_indices = []
    for index, item in zip(global_indices, prompts):
        record = {"id": str(index), "prompt": str(item)} if isinstance(item, str) else dict(item)
        record["id"] = str(record.get("id", record.get("item_id", index)))
        record["prompt"] = str(record.get("prompt", record.get("text", ""))).strip()
        if record["prompt"]:
            normalized.append(record)
            retained_indices.append(index)
    global_indices = retained_indices if prompt_indices is not None else list(range(len(normalized)))
    if max_prompts is not None:
        if int(max_prompts) < 0:
            raise ValueError("max_prompts must be non-negative (0 means all prompts)")
        if int(max_prompts) > 0:
            normalized = normalized[:int(max_prompts)]
            global_indices = global_indices[:int(max_prompts)]
    if not normalized:
        raise ValueError("no DPG prompts available")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    images = _sample_records(
        model,
        text_encoder,
        normalized,
        image_size=image_size,
        steps=steps,
        cfg_scale=cfg_scale,
        t_eps=t_eps,
        seed=seed,
        batch_size=batch_size,
        samples_per_record=4,
        autocast_dtype=autocast_dtype,
        prompt_indices=global_indices,
    )
    manifest = []
    for prompt_index, record in zip(global_indices, normalized):
        grid = Image.new("RGB", (2 * image_size, 2 * image_size))
        for sample_index in range(4):
            image = next(images)
            image = ((image.float().clamp(-1, 1) + 1.0) * 127.5).round().byte()
            pil = Image.fromarray(image.permute(1, 2, 0).numpy(), mode="RGB")
            grid.paste(pil, ((sample_index % 2) * image_size, (sample_index // 2) * image_size))
        grid.save(root / f"{record['id']}.png")
        manifest.append({
            **record, "index": prompt_index, "prompt_index": prompt_index,
            "samples": 4,
            "seeds": [int(seed) + prompt_index * 4 + sample_index for sample_index in range(4)],
        })
    # The official evaluator iterates over every directory entry and assumes
    # each entry is an image.  Keep the audit manifest beside, not inside, the
    # image directory.
    root.with_name(root.name + "_prompts.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in manifest),
        encoding="utf-8",
    )
    return root


def _json_from_text(text: str) -> Any | None:
    candidates = [text.strip()]
    candidates.extend(line.strip() for line in reversed(text.splitlines()))
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _benchmark_environment(device: torch.device | str | None = None) -> dict[str, str]:
    """Isolate a scorer from the launcher's distributed state and GPU pool."""

    env = os.environ.copy()
    distributed_keys = {
        "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
        "GROUP_RANK", "GROUP_WORLD_SIZE", "ROLE_RANK", "ROLE_WORLD_SIZE", "ROLE_NAME",
        "MASTER_ADDR", "MASTER_PORT", "NODE_RANK", "NNODES", "NPROC_PER_NODE",
        "TORCH_DIST_INIT_BARRIER",
    }
    distributed_prefixes = (
        "ACCELERATE_", "TORCHELASTIC_", "TORCHRUN_", "PMI_", "PMIX_",
        "OMPI_COMM_WORLD_", "MV2_COMM_WORLD_",
    )
    for key in list(env):
        if key in distributed_keys or key.startswith(distributed_prefixes):
            env.pop(key)
    if device is None or torch.device(device).type != "cuda":
        return env
    selected = torch.device(device).index
    if selected is None:
        # CPU-only tests and callers that do not select CUDA retain their
        # inherited visibility. A CUDA worker resolves its current local GPU.
        if not torch.cuda.is_available():
            return env
        selected = torch.cuda.current_device()
    visible = env.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        env["CUDA_VISIBLE_DEVICES"] = str(selected)
    else:
        identifiers = [value.strip() for value in visible.split(",") if value.strip()]
        if selected >= len(identifiers) or identifiers[selected] == "-1":
            raise ValueError(f"CUDA device {selected} is outside CUDA_VISIBLE_DEVICES={visible!r}")
        # Keep physical ordinals, GPU UUIDs, and MIG identifiers verbatim.
        env["CUDA_VISIBLE_DEVICES"] = identifiers[selected]
    return env


def run_external_benchmark(
    command: str,
    *,
    images_dir: str | Path,
    prompts_file: str | Path,
    output_dir: str | Path,
    timeout: int = 3600,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Run an official benchmark command and parse its JSON result.

    Commands must contain ``{images}``, ``{prompts}``, and ``{output}``
    placeholders.  The wrapper never invents a metric if the runner fails or
    emits no JSON; it returns a structured error for ``metrics.json``.
    """

    required = ("{images}", "{prompts}", "{output}")
    if any(token not in command for token in required):
        return {
            "status": "not_run",
            "error": "benchmark command must contain {images}, {prompts}, and {output}",
        }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    command = command.format(
        images=str(Path(images_dir).resolve()),
        prompts=str(Path(prompts_file).resolve()),
        output=str(destination.resolve()),
    )
    env = _benchmark_environment(device)
    try:
        completed = subprocess.run(
            shlex.split(command), capture_output=True, text=True,
            timeout=int(timeout), check=False, env=env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "error", "error": repr(exc)}
    parsed = None
    for candidate in sorted(destination.glob("*.json")):
        try:
            parsed = json.loads(candidate.read_text())
            break
        except (OSError, json.JSONDecodeError):
            continue
    if parsed is None:
        parsed = _json_from_text(completed.stdout)
    jsonl_files = sorted(destination.glob("*.jsonl"))
    result: dict[str, Any] = {
        "status": "ok" if completed.returncode == 0 and (parsed is not None or jsonl_files) else "error",
        "returncode": int(completed.returncode),
    }
    if parsed is not None:
        result["metrics"] = parsed
    if jsonl_files:
        result["jsonl_outputs"] = [str(path) for path in jsonl_files]
        result["num_jsonl_records"] = sum(
            sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
            for path in jsonl_files
        )
    if completed.stdout:
        result["stdout_tail"] = completed.stdout[-4000:]
    if completed.stderr:
        result["stderr_tail"] = completed.stderr[-4000:]
    if completed.returncode != 0 and "error" not in result:
        result["error"] = "external benchmark exited with a non-zero status"
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    if not path.is_file():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                records.append(value)
    return records


def _benchmark_subprocess_error(
    exc: OSError | subprocess.TimeoutExpired,
    command: Sequence[str],
    destination: Path,
    result_path: Path,
) -> dict[str, Any]:
    """Preserve partial subprocess output, including timeout byte streams."""

    output = {}
    for name in ("stdout", "stderr"):
        value = getattr(exc, name, None)
        text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value or "")
        (destination / f"{name}.log").write_text(text, encoding="utf-8")
        output[f"{name}_tail"] = text[-4000:]
    timed_out = isinstance(exc, subprocess.TimeoutExpired)
    result = {
        "status": "error", "error": repr(exc), "command": list(command),
        "reason": "benchmark_timeout" if timed_out else "benchmark_launch_error",
        "result_path": str(result_path), **output,
    }
    if timed_out:
        result["timeout_seconds"] = exc.timeout
    return result


def run_geneval_official(
    images_dir: str | Path,
    assets_dir: str | Path,
    output_dir: str | Path,
    *,
    timeout: int = 3600,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Run the pinned GenEval evaluator and summarize its JSONL output."""

    assets = Path(assets_dir)
    repository = assets / "geneval"
    evaluator = repository / "evaluation/evaluate_images.py"
    detector = assets / "geneval_object_detector"
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    result_path = destination / "results.jsonl"
    if not evaluator.is_file():
        return {"status": "not_run", "reason": "missing_geneval_evaluator", "path": str(evaluator)}
    if not detector.is_dir():
        return {"status": "not_run", "reason": "missing_geneval_detector", "path": str(detector)}
    metadata_files = sorted(Path(images_dir).glob("*/metadata.jsonl"))
    if not metadata_files:
        return {"status": "error", "reason": "missing_geneval_metadata", "path": str(images_dir)}
    # ``exclude`` is used by counting prompts but is intentionally absent from
    # the single/two-object and attribute sections of the official manifest.
    required_fields = {"prompt", "tag", "include"}
    for metadata_file in metadata_files:
        rows = _read_jsonl(metadata_file)
        if not rows:
            return {"status": "error", "reason": "empty_geneval_metadata", "path": str(metadata_file)}
        missing = sorted(required_fields.difference(rows[0]))
        if missing:
            return {
                "status": "error",
                "reason": "invalid_geneval_metadata",
                "path": str(metadata_file),
                "missing_fields": missing,
            }
    expected_images = {
        str(path.resolve())
        for metadata_file in metadata_files
        for path in (metadata_file.parent / "samples").glob("*.png")
    }
    empty_prompts = [
        str(path.parent) for path in metadata_files
        if not any((path.parent / "samples").glob("*.png"))
    ]
    if empty_prompts:
        return {"status": "error", "reason": "empty_geneval_samples", "paths": empty_prompts}
    geneval_python = os.environ.get("GENEVAL_PYTHON", sys.executable)
    command = [
        geneval_python,
        str(Path(__file__).resolve().parent / "tools/evaluate_geneval.py"),
        "--official-script",
        str(evaluator.resolve()),
        str(Path(images_dir).resolve()),
        "--outfile",
        str(result_path.resolve()),
        "--model-path",
        str(detector.resolve()),
    ]
    env = _benchmark_environment(device)
    try:
        completed = subprocess.run(
            command,
            cwd=str(repository),
            capture_output=True,
            text=True,
            timeout=int(timeout),
            check=False,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            **_benchmark_subprocess_error(exc, command, destination, result_path),
            "expected_images": len(expected_images),
        }
    records = _read_jsonl(result_path)
    processed_images = [str(Path(row.get("filename", "")).resolve()) for row in records]
    complete = len(processed_images) == len(expected_images) and set(processed_images) == expected_images
    ok = completed.returncode == 0 and bool(records) and complete
    summary: dict[str, Any] = {
        "num_images": len(records),
        "num_prompts": None,
        "image_accuracy": float(sum(bool(row.get("correct", False)) for row in records) / len(records))
        if records else None,
    }
    prompt_keys = ["metadata", "prompt"]
    groups: dict[str, list[bool]] = {}
    task_groups: dict[str, list[bool]] = {}
    for row in records:
        prompt_key = next((str(row[key]) for key in prompt_keys if key in row), "")
        groups.setdefault(prompt_key, []).append(bool(row.get("correct", False)))
        task_groups.setdefault(str(row.get("tag", "unknown")), []).append(bool(row.get("correct", False)))
    if groups:
        summary["num_prompts"] = len(groups)
        summary["prompt_accuracy"] = float(sum(any(values) for values in groups.values()) / len(groups))
    summary["task_accuracy"] = {
        key: float(sum(values) / len(values)) for key, values in sorted(task_groups.items()) if values
    }
    if summary["task_accuracy"]:
        summary["overall_score"] = float(sum(summary["task_accuracy"].values()) / len(summary["task_accuracy"]))
    else:
        summary["overall_score"] = None
    if not ok:
        summary["partial_overall_score"] = summary["overall_score"]
        summary["overall_score"] = None
    summary_path = destination / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (destination / "stdout.log").write_text(completed.stdout or "", encoding="utf-8")
    (destination / "stderr.log").write_text(completed.stderr or "", encoding="utf-8")
    return {
        "status": "ok" if ok else "error",
        "returncode": int(completed.returncode),
        "result_path": str(result_path),
        "expected_images": len(expected_images),
        "processed_images": len(processed_images),
        "all_generated_images_evaluated": complete,
        "summary": summary,
        "stdout_tail": (completed.stdout or "")[-4000:],
        "stderr_tail": (completed.stderr or "")[-4000:],
    }


def run_dpg_official(
    images_dir: str | Path,
    assets_dir: str | Path,
    output_dir: str | Path,
    *,
    image_size: int,
    timeout: int = 7200,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Run the official ELLA DPG-Bench VQA evaluator on generated grids."""

    _require_t2i_image_size(image_size)
    assets = Path(assets_dir)
    repository = assets / "ella"
    evaluator = repository / "dpg_bench/compute_dpg_bench.py"
    csv_path = repository / "dpg_bench/dpg_bench.csv"
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    result_path = destination / "dpg_results.txt"
    if not evaluator.is_file() or not csv_path.is_file():
        missing = str(evaluator if not evaluator.is_file() else csv_path)
        return {"status": "not_run", "reason": "missing_dpg_evaluator", "path": missing}

    source = Path(images_dir)
    if not source.is_dir():
        return {"status": "error", "reason": "missing_dpg_image_dir", "path": str(source)}
    image_paths = sorted(
        path for path in source.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    )
    if not image_paths:
        return {"status": "error", "reason": "empty_dpg_image_dir", "path": str(source)}
    try:
        csv_sha256 = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        prompt_records, question_data = load_dpg_data(csv_path)
        known_ids = {record["id"] for record in prompt_records}
    except (OSError, ValueError, KeyError) as exc:
        return {"status": "error", "reason": "invalid_dpg_csv", "error": repr(exc)}
    unknown_ids = sorted(path.stem for path in image_paths if path.stem not in known_ids)
    if unknown_ids:
        return {
            "status": "error",
            "reason": "unknown_dpg_image_ids",
            "path": str(source),
            "unknown_ids": unknown_ids,
        }
    generation_manifest = source.with_name(source.name + "_prompts.jsonl")
    if generation_manifest.is_file():
        try:
            planned_ids = {record["id"] for record in load_prompt_manifest(generation_manifest)}
        except (OSError, ValueError, KeyError) as exc:
            return {"status": "error", "reason": "invalid_dpg_generation_manifest", "error": repr(exc)}
        actual_ids = {path.stem for path in image_paths}
        if planned_ids != actual_ids:
            return {
                "status": "error", "reason": "incomplete_dpg_generation",
                "missing_ids": sorted(planned_ids - actual_ids),
                "extra_ids": sorted(actual_ids - planned_ids),
            }
    invalid_images: list[str] = []
    expected_size = (2 * int(image_size), 2 * int(image_size))
    for image_path in image_paths:
        try:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                if image.size != expected_size:
                    invalid_images.append(
                        f"{image_path.name}: expected {expected_size}, got {image.size}"
                    )
        except Exception as exc:
            invalid_images.append(f"{image_path.name}: {exc}")
    if invalid_images:
        return {
            "status": "error",
            "reason": "invalid_dpg_images",
            "path": str(source),
            "invalid_images": invalid_images,
        }

    dpg_python = os.environ.get("DPG_PYTHON", sys.executable)
    for stale in (result_path, destination / "dpg_results_detail.txt"):
        if stale.is_file():
            stale.unlink()
    # The upstream script calls os.listdir() rather than filtering by suffix.
    # Run it on a temporary image-only view so manifests and logs cannot be
    # interpreted as images. Symlinks avoid copying a potentially large set.
    stage = Path(tempfile.mkdtemp(prefix="dpg-images-", dir=str(destination.parent)))
    try:
        for image_path in image_paths:
            staged_path = stage / image_path.name
            try:
                staged_path.symlink_to(image_path.resolve())
            except OSError:
                shutil.copy2(image_path, staged_path)
        command = [
            dpg_python,
            str(Path(__file__).resolve().parent / "tools/evaluate_dpg_bench.py"),
            "--official-script",
            str(evaluator.resolve()),
            "--image-root-path",
            str(stage.resolve()),
            "--resolution",
            str(int(image_size)),
            "--csv",
            str(csv_path.resolve()),
            "--res-path",
            str(result_path.resolve()),
            "--pic-num",
            "4",
            "--vqa-model",
            "mplug",
        ]
        env = _benchmark_environment(device)
        cache_dir = assets / "modelscope"
        env["MODELSCOPE_CACHE"] = str(cache_dir.resolve())
        env["MODELSCOPE_CACHE_DIR"] = str(cache_dir.resolve())
        try:
            completed = subprocess.run(
                command,
                cwd=str(repository),
                capture_output=True,
                text=True,
                timeout=int(timeout),
                check=False,
                env=env,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {
                **_benchmark_subprocess_error(exc, command, destination, result_path),
                "num_images": len(image_paths), "csv_sha256": csv_sha256,
                "score": None, "score_percent": None,
            }
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    text = result_path.read_text(encoding="utf-8", errors="replace") if result_path.is_file() else ""
    match = re.findall(r"DPG-Bench score:\s*([0-9]+(?:\.[0-9]+)?)", text)
    score = float(match[-1]) if match else None
    combined_output = "\n".join((completed.stdout or "", completed.stderr or ""))
    failures = [line.strip() for line in combined_output.splitlines() if "Failed filename:" in line]
    try:
        csv_sha256_after = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    except OSError as exc:
        csv_sha256_after = None
        failures.append(f"DPG CSV unavailable after evaluation: {exc}")
    if csv_sha256_after != csv_sha256:
        failures.append("DPG CSV changed during evaluation")
    processed_ids = [
        Path(line.split(",", 1)[0]).stem for line in text.splitlines()
        if ".png," in line and line.lstrip().startswith(("/", "\\"))
    ]
    expected_ids = {path.stem for path in image_paths}
    processed = len(processed_ids)
    if processed != len(image_paths) or set(processed_ids) != expected_ids:
        failures.append(
            f"official evaluator processed {processed}/{len(image_paths)} image files"
        )
    (destination / "stdout.log").write_text(completed.stdout or "", encoding="utf-8")
    (destination / "stderr.log").write_text(completed.stderr or "", encoding="utf-8")
    ok = completed.returncode == 0 and score is not None and not failures
    return {
        "status": "ok" if ok else "error",
        "returncode": int(completed.returncode),
        "result_path": str(result_path),
        "num_images": len(image_paths),
        "processed_images": processed,
        "failed_files": failures,
        "complete_benchmark": expected_ids == known_ids,
        "num_benchmark_prompts": len(known_ids),
        "num_questions": sum(len(question_data[key]["qid2question"]) for key in expected_ids),
        "csv_sha256": csv_sha256,
        "csv_sha256_after": csv_sha256_after,
        "csv_unchanged": csv_sha256_after == csv_sha256,
        "question_loader": "all_csv_data_rows",
        "score_percent": score if ok else None,
        "score": score / 100.0 if ok else None,
        "partial_score_percent": score if not ok else None,
        "stdout_tail": (completed.stdout or "")[-4000:],
        "stderr_tail": (completed.stderr or "")[-4000:],
    }


def compute_fid_kid(
    generated_dir: str | Path,
    reference_dir: str | Path,
    *,
    device: torch.device | str = "cuda",
) -> dict[str, Any]:
    """Compute FID and KID against an explicit real-image directory."""

    try:
        import torch_fidelity
    except ImportError as exc:
        return {"status": "not_run", "error": f"torch-fidelity unavailable: {exc}"}
    reference = Path(reference_dir)
    if not reference.is_dir():
        return {"status": "not_run", "reason": "missing_reference_dir", "path": str(reference)}
    reference_images = [
        path for path in reference.rglob("*")
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    ]
    if not reference_images:
        return {"status": "not_run", "reason": "empty_reference_dir", "path": str(reference)}
    try:
        result = torch_fidelity.calculate_metrics(
            input1=str(generated_dir),
            input2=str(reference_dir),
            cuda=torch.device(device).type == "cuda",
            fid=True,
            kid=True,
            isc=False,
            verbose=False,
        )
        return {
            "status": "ok",
            "fid": float(result["frechet_inception_distance"]),
            "kid_mean": float(result["kernel_inception_distance_mean"]),
            "kid_std": float(result["kernel_inception_distance_std"]),
        }
    except Exception as exc:  # external model/cache/data failure is reportable
        return {"status": "error", "error": repr(exc)}


def evaluate_generated(
    generated_dir: str | Path,
    *,
    reference_dir: str | Path | None = None,
    geneval_command: str | None = None,
    dpg_command: str | None = None,
    eval_assets_dir: str | Path | None = None,
    geneval_image_dir: str | Path | None = None,
    dpg_image_dir: str | Path | None = None,
    image_size: int | None = None,
    device: torch.device | str = "cuda",
    benchmark_timeout: int = 3600,
    skip_empty: bool = False,
) -> dict[str, Any]:
    """Evaluate an existing image set and write a machine-readable result.

    With ``skip_empty``, a benchmark with no local images is ``not_assigned``;
    other benchmarks still run. Sharded callers should omit ``reference_dir``
    and compute FID/KID once after combining all generated images.
    """

    total_start = time.perf_counter()
    root = Path(generated_dir)
    root.mkdir(parents=True, exist_ok=True)
    prompts_file = root / "prompts.jsonl"
    metrics: dict[str, Any] = {
        "generated_dir": str(root.resolve()),
        "num_images": len(list(root.glob("*.png"))),
    }
    started = time.perf_counter()
    if reference_dir is not None:
        metrics["fid_kid"] = compute_fid_kid(root, reference_dir, device=device)
    else:
        metrics["fid_kid"] = {
            "status": "not_run", "reason": "reference_dir_not_provided",
        }
    timings = {"fid_kid_seconds": time.perf_counter() - started}
    started = time.perf_counter()
    geneval_source = (
        root.with_name(root.name + "_geneval") if geneval_command
        else Path(geneval_image_dir) if geneval_image_dir is not None else None
    )
    dpg_source = root if dpg_command else Path(dpg_image_dir) if dpg_image_dir is not None else None
    if skip_empty and geneval_source is not None and not any(geneval_source.glob("*/samples/*.png")):
        metrics["geneval"] = {
            "status": "not_assigned", "num_images": 0, "num_prompts": 0,
            "expected_images": 0, "processed_images": 0,
        }
    elif geneval_command:
        metrics["geneval"] = run_external_benchmark(
            geneval_command, images_dir=root.with_name(root.name + "_geneval"), prompts_file=prompts_file,
            output_dir=root / "geneval", timeout=benchmark_timeout, device=device,
        )
    elif eval_assets_dir is not None and geneval_image_dir is not None:
        metrics["geneval"] = run_geneval_official(
            geneval_image_dir,
            eval_assets_dir,
            root / "geneval",
            timeout=benchmark_timeout,
            device=device,
        )
    else:
        metrics["geneval"] = {
            "status": "not_run", "reason": "geneval_command_not_provided",
        }
    timings["geneval_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    if skip_empty and dpg_source is not None and not any(dpg_source.glob("*.png")):
        metrics["dpg_bench"] = {
            "status": "not_assigned", "num_images": 0, "num_prompts": 0,
            "processed_images": 0, "num_questions": 0,
        }
    elif dpg_command:
        metrics["dpg_bench"] = run_external_benchmark(
            dpg_command, images_dir=root, prompts_file=prompts_file,
            output_dir=root / "dpg_bench", timeout=benchmark_timeout, device=device,
        )
    elif eval_assets_dir is not None and dpg_image_dir is not None and image_size is not None:
        metrics["dpg_bench"] = run_dpg_official(
            dpg_image_dir,
            eval_assets_dir,
            root / "dpg_bench",
            image_size=image_size,
            timeout=benchmark_timeout,
            device=device,
        )
    else:
        metrics["dpg_bench"] = {
            "status": "not_run", "reason": "dpg_command_not_provided",
        }
    timings["dpg_bench_seconds"] = time.perf_counter() - started
    timings["total_seconds"] = time.perf_counter() - total_start
    metrics["timings"] = timings
    temporary = root / ".metrics.json.tmp"
    temporary.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(root / "metrics.json")
    return metrics


__all__ = [
    "EvaluationConfig", "compute_fid_kid", "evaluate_generated",
    "generate_t2i", "heun_sample", "load_prompt_manifest",
    "generate_dpg_t2i", "load_dpg_prompt_manifest", "run_external_benchmark",
    "run_geneval_official", "run_dpg_official", "shard_prompt_indices",
]
