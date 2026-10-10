"""Deterministic streaming loader for the BLIP3o WebDataset mixture."""

from __future__ import annotations

import io
import json
import os
import random
import tarfile
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
from PIL import Image
from torch.utils.data import IterableDataset
from torchvision.transforms import Normalize
from torchvision.transforms.functional import to_tensor


_RESAMPLING = getattr(Image, "Resampling", Image)
_IMAGE_EXTENSIONS = ("jpg", "jpeg", "png", "webp", "bmp", "avif")
_CAPTION_EXTENSIONS = (
    "txt", "caption", "text", "prompt", "input_prompt",
    "json", "metadata", "meta",
)
_CAPTION_KEYS = (
    "caption", "text", "prompt", "input_prompt", "txt",
    "long_caption", "short_caption", "description",
)


def resize_short_side(image: Image.Image, image_size: int) -> Image.Image:
    """Match DeCo's antialiased short-side resize before cropping."""
    while min(image.size) >= 2 * image_size:
        image = image.resize(
            (max(1, image.width // 2), max(1, image.height // 2)),
            resample=_RESAMPLING.BOX,
        )
    scale = image_size / min(image.size)
    return image.resize(
        (max(image_size, round(image.width * scale)),
         max(image_size, round(image.height * scale))),
        resample=_RESAMPLING.BICUBIC,
    )


def center_crop(image: Image.Image, image_size: int) -> Image.Image:
    left = (image.width - image_size) // 2
    top = (image.height - image_size) // 2
    return image.crop((left, top, left + image_size, top + image_size))


def find_t2i_shards(data_root: str | os.PathLike[str]) -> list[str]:
    """Return a stable, duplicate-free list of local tar shards."""
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"T2I data directory does not exist: {root}")
    shards = {
        path.resolve()
        for suffix in ("*.tar", "*.tar.gz", "*.tgz")
        for path in root.rglob(suffix)
        if path.is_file()
    }
    result = sorted(str(path) for path in shards)
    if not result:
        raise FileNotFoundError(f"no tar shards found below {root}")
    return result


def _decode_text(value: Any) -> str | None:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).decode("utf-8", errors="ignore")
    if isinstance(value, str):
        return value
    return None


def _caption_from_value(value: Any) -> str:
    """Extract one deterministic caption from nested BLIP3o metadata."""
    text = _decode_text(value)
    if text is not None:
        return text
    if isinstance(value, dict):
        for key in _CAPTION_KEYS:
            if key in value:
                candidate = _caption_from_value(value[key])
                if candidate.strip():
                    return candidate
        # Do not fall back to arbitrary metadata values such as URLs or
        # sample IDs. Nested caption containers are explicit and auditable.
        for key in ("captions", "texts", "descriptions", "data"):
            if key in value:
                candidate = _caption_from_value(value[key])
                if candidate.strip():
                    return candidate
    if isinstance(value, (list, tuple)):
        for item in value:
            candidate = _caption_from_value(item)
            if candidate.strip():
                return candidate
    return ""


def extract_caption(sample: dict[str, Any]) -> str:
    """Read a caption from decoded WebDataset fields.

    Unparseable JSON is rejected rather than becoming a prompt containing
    serialized metadata.
    """
    for key in _CAPTION_EXTENSIONS:
        if key not in sample:
            continue
        value = sample[key]
        if key in {"json", "metadata", "meta"}:
            raw = _decode_text(value)
            if raw is None:
                continue
            try:
                value = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                continue
        caption = " ".join(_caption_from_value(value).split())
        if caption:
            return caption
    return ""


def extract_image(sample: dict[str, Any]) -> Image.Image | None:
    for key in _IMAGE_EXTENSIONS + ("image", "output_image"):
        value = sample.get(key)
        if isinstance(value, Image.Image):
            return value
        if isinstance(value, (bytes, bytearray, memoryview)):
            try:
                return Image.open(io.BytesIO(bytes(value)))
            except (OSError, ValueError):
                continue
    return None


def _member_key(name: str) -> tuple[str, str]:
    basename = name.replace("\\", "/").rsplit("/", 1)[-1]
    if "." not in basename:
        return basename, ""
    key, extension = basename.rsplit(".", 1)
    return key, extension.lower()


def _iter_tar_samples(path: str) -> Iterator[dict[str, Any]]:
    """Yield WebDataset-like samples from a tar, preserving sample keys."""
    try:
        archive = tarfile.open(path, mode="r:*")
    except (OSError, tarfile.TarError) as exc:
        raise RuntimeError(f"failed to open T2I shard {path}: {exc}") from exc
    with archive:
        current_key: str | None = None
        current: dict[str, Any] = {}

        def emit() -> dict[str, Any] | None:
            if current_key is None or not current:
                return None
            result = dict(current)
            result["__key__"] = current_key
            result["__url__"] = path
            return result

        for member in archive:
            if not member.isfile():
                continue
            key, extension = _member_key(member.name)
            if current_key is not None and key != current_key:
                result = emit()
                if result is not None:
                    yield result
                current.clear()
            current_key = key
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            try:
                current[extension] = extracted.read()
            finally:
                extracted.close()
        result = emit()
        if result is not None:
            yield result


class T2IShardDataset(IterableDataset):
    """Infinite deterministic stream over all BLIP3o tar shards.

    Shards are assigned by ``rank * num_workers + worker_id``.  This avoids
    relying on process-global WebDataset split helpers and keeps rank/worker
    streams disjoint. ``shuffle_buffer`` remains for API compatibility;
    decoded-image buffering is intentionally not used because a 10k image
    buffer would consume excessive host memory. Shard permutations provide
    deterministic mixing of all three BLIP3o sources.
    """

    def __init__(
        self,
        data_root: str | os.PathLike[str],
        image_size: int,
        shuffle_buffer: int = 10_000,
        shard_shuffle_buffer: int = 100,
        seed: int = 0,
        max_invalid_samples: int = 10_000,
    ) -> None:
        super().__init__()
        if image_size != 256:
            raise ValueError(f"T2I data uses a 256-pixel crop, got {image_size}")
        if shuffle_buffer < 0 or shard_shuffle_buffer < 0:
            raise ValueError("shuffle buffers must be non-negative")
        if max_invalid_samples <= 0:
            raise ValueError("max_invalid_samples must be positive")
        self.root = Path(data_root).expanduser().resolve()
        self.shards = find_t2i_shards(self.root)
        self.image_size = int(image_size)
        self.shuffle_buffer = int(shuffle_buffer)
        self.shard_shuffle_buffer = int(shard_shuffle_buffer)
        self.seed = int(seed)
        self.max_invalid_samples = int(max_invalid_samples)
        self.normalize = Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    def _worker_assignment(self) -> tuple[int, int, int]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if not (0 <= rank < world_size):
            raise RuntimeError(f"invalid distributed rank/world size: {rank}/{world_size}")
        global_workers = world_size * num_workers
        global_worker = rank * num_workers + worker_id
        # Partition samples across all workers; this also handles small local
        # shard subsets without assigning an empty worker.
        return global_worker, global_workers, rank

    def _process(self, sample: dict[str, Any]) -> dict[str, Any] | None:
        image = extract_image(sample)
        caption = extract_caption(sample)
        if image is None or not caption:
            return None
        image = image.convert("RGB")
        if min(image.size) < self.image_size:
            return None
        image = center_crop(resize_short_side(image, self.image_size), self.image_size)
        pixels = self.normalize(to_tensor(image))
        relative_shard = os.path.relpath(sample["__url__"], self.root)
        return {
            "image": pixels,
            "caption": caption,
            "sample_id": f"{relative_shard}::{sample['__key__']}",
        }

    def __iter__(self) -> Iterator[dict[str, Any]]:
        global_worker, global_workers, rank = self._worker_assignment()
        epoch = 0
        while True:
            order = list(range(len(self.shards)))
            random.Random(self.seed + 1_000_003 * epoch).shuffle(order)
            shared_shards = len(order) < global_workers
            assigned = order if shared_shards else order[global_worker::global_workers]
            if not assigned:
                raise RuntimeError(f"rank {rank} worker {global_worker} has no assigned shards")
            invalid = 0
            sample_index = 0
            for shard_index in assigned:
                shard = self.shards[shard_index]
                for sample in _iter_tar_samples(shard):
                    if shared_shards:
                        selected = sample_index % global_workers == global_worker
                        sample_index += 1
                        if not selected:
                            continue
                    try:
                        record = self._process(sample)
                    except (OSError, ValueError, RuntimeError):
                        record = None
                    if record is None:
                        invalid += 1
                        if invalid >= self.max_invalid_samples:
                            raise RuntimeError(
                                f"{invalid} invalid/undersized samples while reading {shard}; "
                                "check the BLIP3o tar schema and image resolution"
                            )
                        continue
                    invalid = 0
                    yield record
            epoch += 1


def collate_t2i(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("cannot collate an empty T2I batch")
    return {
        "images": torch.stack([item["image"] for item in batch]),
        "captions": [item["caption"] for item in batch],
        "sample_ids": [item["sample_id"] for item in batch],
    }
