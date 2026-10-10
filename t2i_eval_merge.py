"""Validate and aggregate official benchmark records from distributed evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from t2i_dpg import load_dpg_data


def _read_jsonl(path: Path, errors: list[str]) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        errors.append(f"Cannot read {path}: {exc}")
        return []
    records = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("record is not an object")
        except (ValueError, TypeError) as exc:
            errors.append(f"Invalid {path}:{number}: {exc}")
            continue
        records.append(record)
    return records


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")


def _coverage(expected: set[str], observed: Sequence[str]) -> dict[str, Any]:
    counts = Counter(observed)
    missing = sorted(expected - counts.keys())
    unexpected = sorted(counts.keys() - expected)
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    return {
        "expected": len(expected), "observed": len(observed),
        "missing_ids": missing, "unexpected_ids": unexpected, "duplicate_ids": duplicates,
        "complete": not (missing or unexpected or duplicates),
    }


def _link(source: Path, target: Path, errors: list[str]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        try:
            same_file = os.path.samefile(source, target)
        except OSError:
            same_file = False
        if not same_file:
            errors.append(f"Conflicting output {target}: {source}")
        return
    try:
        os.link(source.resolve(), target)
    except OSError:
        # Exported image trees must remain readable without their shard trees.
        # Copy when hardlinks are unavailable, such as across filesystems.
        try:
            shutil.copy2(source, target)
        except OSError as exc:
            errors.append(f"Cannot copy {source} to {target}: {exc}")


def _image_key(path: Path) -> str:
    if path.parent.name != "samples" or path.suffix != ".png":
        raise ValueError(f"Not an official GenEval sample path: {path}")
    return f"{int(path.parent.parent.name)}:{int(path.stem)}"


def _int_field(record: Mapping[str, Any], field: str) -> int:
    value = record[field]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _rank_errors(shards: Sequence[Mapping[str, Any]], metric: str) -> list[dict[str, Any]]:
    failures = []
    for position, shard in enumerate(shards):
        value = shard.get(metric, {})
        if shard.get("status") == "error" or value.get("status") not in (None, "ok", "not_assigned"):
            failures.append({
                "rank": shard.get("rank", position), "status": shard.get("status"),
                "error": shard.get("error"), "phase": shard.get("phase"), "metric": value,
            })
    return failures


def _all_not_run(shards: Sequence[Mapping[str, Any]], metric: str) -> bool:
    """Identify an intentional generation-only run, excluding partial scoring."""
    statuses = [shard.get(metric, {}).get("status") for shard in shards]
    return (
        "not_run" in statuses
        and all(status in ("not_run", "not_assigned") for status in statuses)
        and all(shard.get("status") != "error" for shard in shards)
        and all(
            not shard.get(metric, {}).get("result_path")
            for shard in shards if shard.get(metric, {}).get("status") == "not_run"
        )
    )


def _timings(shards: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    per_rank = {str(shard.get("rank", index)): shard.get("timings", {}) for index, shard in enumerate(shards)}
    values: dict[str, list[float]] = defaultdict(list)
    for rank_values in per_rank.values():
        if not isinstance(rank_values, Mapping):
            continue
        for key, value in rank_values.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                values[key].append(float(value))
    return {
        "per_rank": per_rank,
        "max_rank_seconds": {key: max(items) for key, items in sorted(values.items())},
        "sum_rank_seconds": {key: sum(items) for key, items in sorted(values.items())},
    }


def merge_evaluation_shards(
    eval_root: Path,
    shard_results: list[dict[str, Any]],
    expected_geneval: list[dict[str, Any]],
    expected_dpg: list[dict[str, Any]],
    samples_per_prompt: int,
    assets_dir: Path | None = None,
) -> dict[str, Any]:
    """Merge image views and rescore raw rows, never averaging rank scores.

    ``expected_geneval`` is the complete selected manifest in global prompt
    order. GenEval's overall is the unweighted mean of task accuracies; for
    an explicitly selected subset, only its present tasks participate, as in
    the official summarizer. DPG's overall is the mean of per-prompt scores.
    Generation or evaluation coverage failures suppress the affected score.
    The caller computes FID/KID once on the merged ``generated_dir``.
    """
    if samples_per_prompt < 1:
        raise ValueError("samples_per_prompt must be positive")
    root = Path(eval_root)
    generated = root / "generated"
    geneval_images = root / "generated_geneval"
    dpg_images = root / "generated_dpg"
    for path in (generated, geneval_images, dpg_images):
        path.mkdir(parents=True, exist_ok=True)

    expected_samples = {
        f"{prompt_index}:{sample_index}"
        for prompt_index in range(len(expected_geneval))
        for sample_index in range(samples_per_prompt)
    }
    expected_flat = {str(index) for index in range(len(expected_geneval) * samples_per_prompt)}
    expected_dpg_ids = {str(row["id"]) for row in expected_dpg}
    expected_dpg_by_id = {str(row["id"]): row for row in expected_dpg}
    geneval_errors: list[str] = []
    generation_errors: list[str] = []
    dpg_errors: list[str] = []
    if len(expected_dpg_ids) != len(expected_dpg):
        dpg_errors.append("Expected DPG manifest contains duplicate prompt ids")
    flat_ids: list[str] = []
    sample_ids: list[str] = []
    dpg_ids: list[str] = []
    manifest_ids: list[str] = []
    metadata_ids: list[str] = []
    dpg_manifest_ids: list[str] = []
    manifest: list[dict[str, Any]] = []
    dpg_manifest: list[dict[str, Any]] = []
    geneval_records: list[dict[str, Any]] = []
    scored_sample_ids: list[str] = []
    dpg_records: list[dict[str, Any]] = []

    for shard in shard_results:
        generation_error_start = len(geneval_errors)
        if shard.get("status") == "error":
            # A worker can fail before clearing its previous output. File
            # coverage alone cannot establish that those images are current.
            generation_errors.append(
                f"Shard {shard.get('rank')} failed; current-run generation is unverified"
            )
        if not shard.get("generated_dir"):
            error = f"Shard {shard.get('rank')} has no generated_dir"
            geneval_errors.append(error)
            generation_errors.append(error)
            dpg_errors.append(f"Shard {shard.get('rank')} has no generated_dir")
            continue
        source = Path(shard["generated_dir"])
        source_geneval = source.with_name(source.name + "_geneval")
        source_dpg = source.with_name(source.name + "_dpg")
        for image in sorted(source.glob("*.png")):
            try:
                index = str(int(image.stem))
            except ValueError:
                geneval_errors.append(f"Invalid generated image name: {image}")
                continue
            flat_ids.append(index)
            _link(image, generated / f"{int(index):06d}.png", geneval_errors)

        source_samples: dict[str, str] = {}
        for image in sorted(source_geneval.glob("*/samples/*.png")):
            try:
                key = _image_key(image)
            except ValueError as exc:
                geneval_errors.append(str(exc))
                continue
            sample_ids.append(key)
            source_samples[str(image.resolve())] = key
            prompt_index, sample_index = map(int, key.split(":"))
            prompt_dir = geneval_images / f"{prompt_index:05d}"
            _link(image, prompt_dir / "samples" / f"{sample_index:04d}.png", geneval_errors)
        for metadata in sorted(source_geneval.glob("*/metadata.jsonl")):
            try:
                index = int(metadata.parent.name)
                metadata_ids.append(str(index))
                if not 0 <= index < len(expected_geneval):
                    raise ValueError("unexpected global prompt index")
                records = _read_jsonl(metadata, geneval_errors)
                if len(records) != 1:
                    raise ValueError("expected one metadata record")
                for field in ("id", "prompt", "tag", "include", "exclude"):
                    if field in expected_geneval[index] and records[0].get(field) != expected_geneval[index][field]:
                        raise ValueError(f"metadata {field} differs from expected prompt")
                _link(metadata, geneval_images / f"{index:05d}" / "metadata.jsonl", geneval_errors)
            except (ValueError, KeyError) as exc:
                geneval_errors.append(f"Invalid metadata {metadata}: {exc}")

        manifest_path = source / "prompts.jsonl"
        if manifest_path.is_file() or any(source.glob("*.png")):
            rows = _read_jsonl(manifest_path, geneval_errors)
            manifest.extend(rows)
            for row in rows:
                try:
                    index = _int_field(row, "index")
                    prompt_index = _int_field(row, "prompt_index")
                    sample_index = _int_field(row, "sample_index")
                    manifest_ids.append(str(index))
                    if not 0 <= prompt_index < len(expected_geneval) or not 0 <= sample_index < samples_per_prompt:
                        raise ValueError("unexpected prompt or sample index")
                    if index != prompt_index * samples_per_prompt + sample_index:
                        raise ValueError("flat and prompt/sample indices disagree")
                    for field in ("id", "prompt", "tag", "include", "exclude"):
                        if field in expected_geneval[prompt_index] and row.get(field) != expected_geneval[prompt_index][field]:
                            raise ValueError(f"{field} differs from expected prompt")
                except (KeyError, ValueError, TypeError) as exc:
                    geneval_errors.append(f"Invalid generation manifest {manifest_path}: {exc}")

        for image in sorted(source_dpg.glob("*.png")):
            dpg_ids.append(image.stem)
            _link(image, dpg_images / image.name, dpg_errors)
        dpg_manifest_path = source_dpg.with_name(source_dpg.name + "_prompts.jsonl")
        if dpg_manifest_path.is_file() or any(source_dpg.glob("*.png")):
            rows = _read_jsonl(dpg_manifest_path, dpg_errors)
            dpg_manifest.extend(rows)
            for row in rows:
                key = str(row.get("id", ""))
                dpg_manifest_ids.append(key)
                if key not in expected_dpg_ids or row.get("prompt") != expected_dpg_by_id[key]["prompt"] or row.get("samples") != 4:
                    dpg_errors.append(f"Invalid DPG generation manifest record for {key!r}")

        generation_errors.extend(geneval_errors[generation_error_start:])
        geneval = shard.get("geneval", {})
        if geneval.get("result_path"):
            records = _read_jsonl(Path(geneval["result_path"]), geneval_errors)
            for row in records:
                filename = row.get("filename")
                key = source_samples.get(str(Path(filename).resolve())) if isinstance(filename, str) else None
                if key is None:
                    geneval_errors.append(f"GenEval result does not match a generated shard image: {filename!r}")
                    continue
                scored_sample_ids.append(key)
                prompt_index, sample_index = map(int, key.split(":"))
                if not 0 <= prompt_index < len(expected_geneval):
                    geneval_errors.append(f"Unexpected GenEval result prompt index: {prompt_index}")
                    continue
                if row.get("tag") != expected_geneval[prompt_index].get("tag"):
                    geneval_errors.append(f"GenEval result tag differs from expected prompt: {filename}")
                    continue
                if not isinstance(row.get("correct"), bool):
                    geneval_errors.append(f"GenEval result lacks boolean correctness: {filename}")
                    continue
                geneval_records.append({
                    **row, "filename": str(geneval_images / f"{prompt_index:05d}" / "samples" / f"{sample_index:04d}.png"),
                    "source_filename": filename, "prompt_index": prompt_index, "sample_index": sample_index,
                })

        dpg = shard.get("dpg_bench", {})
        if dpg.get("result_path"):
            result_path = Path(dpg["result_path"])
            try:
                lines = result_path.read_text(encoding="utf-8").splitlines()
            except OSError as exc:
                dpg_errors.append(f"Cannot read {result_path}: {exc}")
                lines = []
            for number, line in enumerate(lines, 1):
                # ELLA writes: path.png, crop0, crop1, crop2, crop3, prompt_mean.
                # Split from the right so commas inside an absolute path work.
                if ".png," not in line:
                    continue
                try:
                    columns = line.rsplit(",", 5)
                    if len(columns) != 6 or not columns[0].strip().endswith(".png"):
                        raise ValueError("expected a filename, four crop scores and their mean")
                    filename = columns[0].strip()
                    scores = [float(value) for value in columns[1:]]
                    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in scores):
                        raise ValueError("scores must be finite fractions in [0, 1]")
                    if not math.isclose(sum(scores[:4]) / 4, scores[4], rel_tol=1e-9, abs_tol=1e-12):
                        raise ValueError("prompt score is not the mean of four crop scores")
                    key = Path(filename).stem
                    if not (source_dpg / f"{key}.png").is_file():
                        raise ValueError("result does not match a generated shard image")
                    dpg_records.append({"id": key, "crop_scores": scores[:4], "score": scores[4], "source_filename": filename})
                except ValueError as exc:
                    dpg_errors.append(f"Invalid DPG result {result_path}:{number}: {exc}")

    manifest.sort(key=lambda row: (str(row.get("index", "")).zfill(12)))
    dpg_order = {str(row["id"]): index for index, row in enumerate(expected_dpg)}
    dpg_manifest.sort(key=lambda row: dpg_order.get(str(row.get("id")), len(dpg_order)))
    geneval_records.sort(key=lambda row: (row["prompt_index"], row["sample_index"]))
    dpg_records.sort(key=lambda row: dpg_order.get(row["id"], len(dpg_order)))
    _write_jsonl(generated / "prompts.jsonl", manifest)
    _write_jsonl(root / "generated_dpg_prompts.jsonl", dpg_manifest)

    geneval_coverage = {
        "flat_images": _coverage(expected_flat, flat_ids),
        "samples": _coverage(expected_samples, sample_ids),
        "metadata": _coverage({str(index) for index in range(len(expected_geneval))}, metadata_ids),
        "manifest": _coverage(expected_flat, manifest_ids),
        "results": _coverage(expected_samples, scored_sample_ids),
    }
    dpg_coverage = {
        "images": _coverage(expected_dpg_ids, dpg_ids),
        "manifest": _coverage(expected_dpg_ids, dpg_manifest_ids),
        "results": _coverage(expected_dpg_ids, [row["id"] for row in dpg_records]),
    }
    dpg_metadata: dict[str, Any] = {}
    shard_csv_hashes = {
        str(shard.get("rank", index)): shard["dpg_bench"]["csv_sha256"]
        for index, shard in enumerate(shard_results)
        if shard.get("dpg_bench", {}).get("csv_sha256") is not None
    }
    if shard_csv_hashes:
        dpg_metadata["csv_sha256_per_rank"] = shard_csv_hashes
        if len(set(shard_csv_hashes.values())) != 1:
            dpg_errors.append("DPG CSV hashes differ across shards")
    if assets_dir is not None and expected_dpg:
        csv_path = Path(assets_dir) / "ella/dpg_bench/dpg_bench.csv"
        try:
            official_prompts, questions = load_dpg_data(csv_path)
            csv_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()
            dpg_metadata.update({
                "complete_benchmark": expected_dpg_ids == {row["id"] for row in official_prompts},
                "num_benchmark_prompts": len(official_prompts),
                "num_questions": sum(len(questions[key]["qid2question"]) for key in expected_dpg_ids),
                "csv_sha256": csv_hash, "question_loader": "all_csv_data_rows",
            })
            mismatched_ranks = sorted(rank for rank, value in shard_csv_hashes.items() if value != csv_hash)
            if mismatched_ranks:
                dpg_errors.append(f"DPG CSV changed between scoring and merge for ranks {mismatched_ranks}")
        except (OSError, ValueError, KeyError) as exc:
            dpg_metadata["asset_metadata_error"] = str(exc)
            if shard_csv_hashes:
                dpg_errors.append(f"Cannot verify shard DPG CSV hashes: {exc}")

    geneval_failures = _rank_errors(shard_results, "geneval")
    dpg_failures = _rank_errors(shard_results, "dpg_bench")
    geneval_complete = all(value["complete"] for value in geneval_coverage.values())
    dpg_complete = all(value["complete"] for value in dpg_coverage.values())
    geneval_generation_complete = all(
        geneval_coverage[key]["complete"] for key in ("flat_images", "samples", "metadata", "manifest")
    ) and not generation_errors
    geneval_not_run = (
        _all_not_run(shard_results, "geneval") and geneval_generation_complete
        and not geneval_records and not geneval_errors
    )
    dpg_not_run = (
        _all_not_run(shard_results, "dpg_bench")
        and all(dpg_coverage[key]["complete"] for key in ("images", "manifest"))
        and not dpg_records and not dpg_errors
    )
    if geneval_not_run:
        geneval_failures = []
    if dpg_not_run:
        dpg_failures = []
    geneval_ok = geneval_complete and not geneval_errors and not geneval_failures
    dpg_ok = dpg_complete and not dpg_errors and not dpg_failures

    task_groups: dict[str, list[bool]] = defaultdict(list)
    prompt_groups: dict[int, list[bool]] = defaultdict(list)
    for row in geneval_records:
        task_groups[row["tag"]].append(row["correct"])
        prompt_groups[row["prompt_index"]].append(row["correct"])
    task_accuracy = {key: sum(values) / len(values) for key, values in sorted(task_groups.items())}
    overall = sum(task_accuracy.values()) / len(task_accuracy) if task_accuracy else None
    summary = {
        "num_images": len(geneval_records), "num_prompts": len(prompt_groups),
        "image_accuracy": sum(row["correct"] for row in geneval_records) / len(geneval_records) if geneval_records else None,
        "prompt_accuracy": sum(any(values) for values in prompt_groups.values()) / len(prompt_groups) if prompt_groups else None,
        "task_accuracy": task_accuracy, "overall_score": overall if geneval_ok else None,
        "partial_overall_score": overall if not geneval_ok else None,
    }
    geneval_result = generated / "geneval" / "results.jsonl"
    _write_jsonl(geneval_result, geneval_records)
    geneval_result.with_name("summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    dpg_score = sum(row["score"] for row in dpg_records) / len(dpg_records) if dpg_records else None
    dpg_result = generated / "dpg_bench" / "dpg_results.txt"
    _write_jsonl(dpg_result.with_suffix(".jsonl"), dpg_records)
    dpg_text = "".join(
        f"{dpg_images / (row['id'] + '.png')}, " + ", ".join(str(value) for value in [*row["crop_scores"], row["score"]]) + "\n"
        for row in dpg_records
    )
    if dpg_ok and dpg_score is not None:
        dpg_text += f"DPG-Bench score: {dpg_score * 100}\n"
    dpg_result.write_text(dpg_text, encoding="utf-8")
    result: dict[str, Any] = {
        "generated_dir": str(generated.resolve()), "num_images": len(set(flat_ids)),
        "status": "ok" if (geneval_ok or geneval_not_run) and (dpg_ok or dpg_not_run) else "error",
        "aggregation": "official_raw_records", "shards": shard_results, "timings": _timings(shard_results),
        "generation": {
            "status": "ok" if geneval_generation_complete else "error",
            "expected_images": len(expected_flat),
            "errors": generation_errors,
        },
        "fid_kid": {"status": "not_run", "reason": "requires_global_image_evaluation"},
        "geneval": {
            "status": ("not_run" if geneval_not_run else "ok" if geneval_ok else "error") if expected_geneval else "not_assigned",
            "result_path": str(geneval_result), "summary": summary,
            "expected_images": len(expected_samples), "processed_images": len(geneval_records),
            "all_generated_images_evaluated": geneval_complete,
            "coverage": geneval_coverage, "errors": geneval_errors, "failed_shards": geneval_failures,
        },
        "dpg_bench": {
            "status": ("not_run" if dpg_not_run else "ok" if dpg_ok else "error") if expected_dpg else "not_assigned",
            "result_path": str(dpg_result), "num_images": len(dpg_ids), "processed_images": len(dpg_records),
            "score": dpg_score if dpg_ok else None,
            "score_percent": dpg_score * 100 if dpg_ok and dpg_score is not None else None,
            "partial_score_percent": dpg_score * 100 if not dpg_ok and dpg_score is not None else None,
            "coverage": dpg_coverage, "errors": dpg_errors, "failed_shards": dpg_failures,
            **dpg_metadata,
        },
    }
    for name, skipped in (("geneval", geneval_not_run), ("dpg_bench", dpg_not_run)):
        if skipped:
            result[name]["reason"] = "all_assigned_shards_not_run"
    return result


__all__ = ["merge_evaluation_shards"]
