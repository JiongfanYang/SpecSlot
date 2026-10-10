"""Read DPG-Bench prompts and their complete question dependencies."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any


def load_dpg_data(
    csv_path: str | Path,
) -> tuple[list[dict[str, str]], dict[str, dict[str, Any]]]:
    """Preserve every data row, including question 1 of the first prompt."""

    path = Path(csv_path)
    prompts: dict[str, str] = {}
    questions: dict[str, dict[str, Any]] = {}
    required = {
        "item_id", "text", "proposition_id", "dependency", "tuple",
        "question_natural_language",
    }
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"DPG CSV lacks columns: {', '.join(sorted(missing))}")
        for row_number, row in enumerate(reader, start=2):
            item_id = str(row.get("item_id") or "").strip()
            prompt = str(row.get("text") or "").strip()
            question = str(row.get("question_natural_language") or "").strip()
            category = str(row.get("tuple") or "").strip()
            # The official scorer obtains the item id with basename.split('.')[0].
            if not re.fullmatch(r"[A-Za-z0-9_-]+", item_id):
                raise ValueError(f"DPG CSV row {row_number}: unsafe item_id {item_id!r}")
            if not prompt or not question or not category:
                raise ValueError(f"DPG CSV row {row_number}: empty prompt/question/tuple")
            try:
                qid = int(row["proposition_id"])
                parents = [int(value.strip()) for value in row["dependency"].split(",")]
            except (TypeError, ValueError, AttributeError) as exc:
                raise ValueError(f"DPG CSV row {row_number}: invalid question/dependency ids") from exc
            if qid <= 0 or any(parent < 0 for parent in parents):
                raise ValueError(f"DPG CSV row {row_number}: question ids must be positive")
            if item_id in prompts and prompts[item_id] != prompt:
                raise ValueError(f"DPG CSV row {row_number}: inconsistent text for {item_id}")
            prompts[item_id] = prompt
            item = questions.setdefault(item_id, {
                "qid2tuple": {}, "qid2dependency": {}, "qid2question": {},
            })
            if qid in item["qid2question"]:
                raise ValueError(f"DPG CSV row {row_number}: duplicate {item_id}/question {qid}")
            item["qid2tuple"][qid] = category
            item["qid2dependency"][qid] = parents
            item["qid2question"][qid] = question

    if not questions:
        raise ValueError(f"DPG CSV contains no questions: {path}")
    for item_id, item in questions.items():
        qids = set(item["qid2question"])
        for qid, parents in item["qid2dependency"].items():
            missing_parents = set(parents).difference(qids, {0})
            if missing_parents:
                raise ValueError(
                    f"DPG CSV {item_id}/question {qid} references absent questions "
                    f"{sorted(missing_parents)}"
                )
    return [{"id": key, "prompt": value} for key, value in prompts.items()], questions
