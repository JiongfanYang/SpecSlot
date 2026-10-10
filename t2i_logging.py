"""Rank-zero training logs and TensorBoard summaries for T2I runs."""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path

import torch
import torch.distributed as dist


def reduce_loss(loss_sum: float, steps: int, device: torch.device) -> float:
    """Average a logging window over all workers without retaining a graph."""
    totals = torch.tensor([loss_sum, float(steps)], device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return float((totals[0] / totals[1]).item())


class T2IRunLogger:
    """Create this only on rank zero, after restoring the checkpoint step."""

    def __init__(self, output_dir: Path, *, resume_step: int | None = None):
        from torch.utils.tensorboard import SummaryWriter

        self.output_dir = Path(output_dir)
        self.log_dir = self.output_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir = self.output_dir / "tensorboard"
        self.writer = SummaryWriter(
            log_dir=str(self.tensorboard_dir),
            purge_step=resume_step + 1 if resume_step is not None else None,
            flush_secs=10,
        )
        self.logger = logging.Logger("t2i", level=logging.INFO)
        formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        for handler in (
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(self.log_dir / "train.log", mode="a", encoding="utf-8"),
        ):
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.steps = (self.log_dir / "steps.jsonl").open("a", encoding="utf-8")
        # Preserve the existing per-epoch file and its schema.
        self.epochs = (self.output_dir / "train.jsonl").open("a", encoding="utf-8")
        self.info(f"TensorBoard: {self.tensorboard_dir}")
        if resume_step is not None:
            self.info(f"Resuming logs at global_step={resume_step}")

    def info(self, message: str) -> None:
        self.logger.info(message)

    def exception(self, message: str) -> None:
        self.logger.exception(message)

    @staticmethod
    def _write_json(handle, record: dict) -> None:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()

    def log_step(
        self, *, epoch: int, epoch_step: int, steps_per_epoch: int,
        global_step: int, loss: float, lr: float,
    ) -> None:
        record = {
            "epoch": epoch, "epoch_step": epoch_step, "global_step": global_step,
            "loss": loss, "lr": lr,
        }
        self._write_json(self.steps, record)
        self.info(
            f"Epoch [{epoch}] step [{epoch_step}/{steps_per_epoch}] "
            f"global_step={global_step} loss={loss:.6f} lr={lr:.6g}"
        )
        self.writer.add_scalar("train/loss", loss, global_step)
        self.writer.add_scalar("train/lr", lr, global_step)
        self.writer.add_scalar("train/epoch", epoch + epoch_step / steps_per_epoch, global_step)
        self.writer.flush()

    def log_epoch(self, record: dict) -> None:
        self._write_json(self.epochs, record)
        self.info(json.dumps(record, sort_keys=True))
        step = record["global_step"]
        self.writer.add_scalar("train/epoch_loss", record["loss"], step)
        self.writer.add_scalar("train/epoch_seconds", record["seconds"], step)
        self.writer.flush()

    def _scalar(self, name: str, value, step: int) -> None:
        if isinstance(value, (int, float)) and math.isfinite(value):
            self.writer.add_scalar(name, value, step)

    def log_evaluation(self, metrics: dict, global_step: int) -> None:
        self._scalar("eval/wall_seconds", metrics.get("timings", {}).get("wall_seconds"), global_step)
        for name in ("geneval", "dpg_bench", "fid_kid"):
            result = metrics.get(name, {})
            if result.get("status") in {"ok", "error"}:
                self.writer.add_scalar(
                    f"eval_status/{name}_success", int(result["status"] == "ok"), global_step,
                )
        geneval = metrics.get("geneval", {})
        if geneval.get("status") == "ok":
            summary = geneval.get("summary", {})
            self._scalar("eval/geneval/overall", summary.get("overall_score"), global_step)
            for name, value in summary.get("task_accuracy", {}).items():
                self._scalar(f"eval/geneval/{name}", value, global_step)
        dpg = metrics.get("dpg_bench", {})
        if dpg.get("status") == "ok":
            self._scalar("eval/dpg_bench/score", dpg.get("score"), global_step)
        fid_kid = metrics.get("fid_kid", {})
        if fid_kid.get("status") == "ok":
            for name in ("fid", "kid_mean", "kid_std"):
                self._scalar(f"eval/{name}", fid_kid.get(name), global_step)
        self.writer.add_scalar("eval_status/success", int(metrics.get("status") == "ok"), global_step)
        self.writer.flush()

    def close(self) -> None:
        self.writer.close()
        self.steps.close()
        self.epochs.close()
        for handler in list(self.logger.handlers):
            handler.close()
            self.logger.removeHandler(handler)
