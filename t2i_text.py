"""Frozen OpenCLIP token encoder used by the T2I training entry point."""

from __future__ import annotations

import torch
from torch import nn


class FrozenOpenCLIPTextEncoder(nn.Module):
    """Return the unprojected 77-token ViT-L/14 text states."""

    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "laion2b_s32b_b82k",
        device: torch.device | str = "cuda",
    ) -> None:
        super().__init__()
        try:
            import open_clip
        except ImportError as exc:
            raise RuntimeError(
                "T2I text conditioning requires open_clip_torch; run `uv sync --locked`"
            ) from exc
        # Load on CPU first so the unused vision tower never occupies device
        # memory.  Move the text-only module after removing that tower.
        model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device="cpu"
        )
        self.model = model.eval()
        # The vision tower is not part of T2I conditioning.  Drop it after
        # loading the official checkpoint to avoid keeping several GB of
        # frozen parameters in every training process.
        if hasattr(self.model, "visual"):
            self.model.visual = None
        self.model.to(device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.max_length = int(getattr(self.model, "context_length", 77))
        if self.max_length != 77:
            raise ValueError(f"expected CLIP context length 77, got {self.max_length}")
        text_width = int(self.model.token_embedding.weight.shape[1])
        if text_width != 768:
            raise ValueError(f"expected ViT-L/14 text width 768, got {text_width}")

    @property
    def device(self) -> torch.device:
        return self.model.token_embedding.weight.device

    @torch.no_grad()
    def forward(self, captions: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.tokenizer(captions).to(self.device)
        model = self.model
        x = model.token_embedding(tokens)
        x = x + model.positional_embedding.to(dtype=x.dtype, device=x.device)
        attention_mask = getattr(model, "attn_mask", None)
        # OpenCLIP releases have used both sequence-first and batch-first
        # transformer blocks.  Inspect the actual attention module instead of
        # assuming one layout, while retaining the official CLIP computation.
        first_block = model.transformer.resblocks[0]
        batch_first = bool(getattr(first_block.attn, "batch_first", False))
        if batch_first:
            x = model.transformer(x, attn_mask=attention_mask)
        else:
            x = model.transformer(x.permute(1, 0, 2), attn_mask=attention_mask)
            x = x.permute(1, 0, 2)
        x = model.ln_final(x).float()
        # CLIP uses the highest token id for EOT.  Everything through EOT is
        # valid; positions after it are padding and must be masked by the
        # text resampler.
        eot = tokens.argmax(dim=-1)
        positions = torch.arange(
            tokens.shape[1], device=tokens.device
        ).unsqueeze(0)
        valid_mask = positions <= eot.unsqueeze(1)
        return x, valid_mask
