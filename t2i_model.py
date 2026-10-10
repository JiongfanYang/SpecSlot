"""Text-conditioned JiT and SpecSlot backbones for the T2I experiment."""

from __future__ import annotations

import torch
from torch import nn

from model_jit import JiT_models


class TextContextAdapter(nn.Module):
    """Map frozen token-level CLIP states to the 32 context slots."""

    def __init__(self, text_width: int = 768, hidden_size: int = 768, slots: int = 32):
        super().__init__()
        self.slots = int(slots)
        self.input_norm = nn.LayerNorm(text_width)
        self.input_projection = (
            nn.Identity() if text_width == hidden_size else nn.Linear(text_width, hidden_size)
        )
        self.query = nn.Parameter(torch.randn(1, self.slots, hidden_size) * 0.02)
        self.cross_attention = nn.MultiheadAttention(
            hidden_size, num_heads=8, batch_first=True
        )
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(
        self, text_states: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if text_states.ndim != 3:
            raise ValueError(f"text_states must be [B,L,D], got {tuple(text_states.shape)}")
        keys = self.input_projection(self.input_norm(text_states.float()))
        # Materialize the expanded query batch so DDP receives a gradient with
        # the same contiguous layout as the learnable parameter.
        queries = self.query.expand(text_states.shape[0], -1, -1).to(
            device=keys.device, dtype=keys.dtype
        ).contiguous()
        padding_mask = None if valid_mask is None else ~valid_mask.bool()
        attended, _ = self.cross_attention(
            queries, keys, keys, key_padding_mask=padding_mask, need_weights=False
        )
        return self.output_norm(queries + attended)


def _specslot_kwargs(image_size: int, bottleneck: int) -> dict:
    if image_size != 256:
        raise ValueError(f"T2I SpecSlot is defined at 256 pixels, got {image_size}")
    window_size, stride = 32, 16
    return dict(
        input_size=image_size,
        window_size=window_size,
        num_classes=1000,
        specslot=True,
        spec_n_max=1024,
        spec_fixed_support=True,
        spec_frequency_order="radial",
        spec_window_stride=stride,
        spec_factorized_position=True,
        spec_hierarchical=True,
        spec_global_depth=6,
        spec_hierarchy_mode="interleave",
        spec_semantic_blocks=1,
        spec_spatialized_input=True,
        spec_spatialized_bottleneck_dim=bottleneck,
        spec_spatialized_activation="linear",
        spec_spatial_input_mixing=True,
        spec_hybrid_output=True,
    )


def _jit_kwargs(image_size: int) -> dict:
    if image_size != 256:
        raise ValueError(f"T2I JiT is defined at 256 pixels, got {image_size}")
    return dict(
        input_size=image_size,
        num_classes=1000,
        specslot=False,
    )


class TextConditionedDenoiser(nn.Module):
    """Shared flow-matching wrapper for JiT and SpecSlot."""

    def __init__(
        self,
        kind: str,
        image_size: int,
        bottleneck: int | None = None,
        activation_checkpointing: bool = False,
        jit_variant: str = "auto",
    ):
        super().__init__()
        kind = kind.lower()
        if kind not in {"specslot", "jit"}:
            raise ValueError(f"kind must be specslot or jit, got {kind!r}")
        jit_variant = str(jit_variant).lower()
        if jit_variant not in {"auto", "b8", "b16"}:
            raise ValueError(
                f"jit_variant must be auto, b8, or b16, got {jit_variant!r}"
            )
        if kind == "specslot" and jit_variant != "auto":
            raise ValueError("jit_variant is only valid for the JiT baseline")
        self.kind = kind
        if kind == "jit":
            self.jit_variant = (
                jit_variant if jit_variant != "auto"
                else "b8"
            )
            model_name = f"JiT-B/{self.jit_variant[1:]}"
        else:
            self.jit_variant = None
            # Use the B/8 backbone settings; SpecSlot replaces its spatial
            # tokenizer with the anchor-and-slot tokenizer below.
            model_name = "JiT-B/8"
        if image_size != 256:
            raise ValueError(f"T2I models are defined at 256 pixels, got {image_size}")
        if bottleneck is None:
            bottleneck = 1536
        kwargs = (
            _specslot_kwargs(image_size, bottleneck)
            if kind == "specslot"
            else _jit_kwargs(image_size)
        )
        kwargs["activation_checkpointing"] = bool(activation_checkpointing)
        self.model_name = "SpecSlot" if kind == "specslot" else model_name
        self.image_size = int(image_size)
        self.backbone = JiT_models[model_name](**kwargs)
        if kind == "specslot":
            # Match the clean-x initialization used by the ImageNet Denoiser:
            # the native spectral head and the additive spatial residual start
            # at zero, so the initial prediction is a stable zero field.
            output_projection = self.backbone.freq_output_proj
            nn.init.constant_(output_projection.weight, 0)
            nn.init.constant_(output_projection.bias, 0)
            if hasattr(self.backbone, "hybrid_spatial_residual_proj"):
                nn.init.constant_(self.backbone.hybrid_spatial_residual_proj.weight, 0)
                nn.init.constant_(self.backbone.hybrid_spatial_residual_proj.bias, 0)
        # Keep the adapter initialization identical across the two backbones;
        # their parameter construction order is intentionally different.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0x5EC5107)
            self.text_adapter = TextContextAdapter(
                text_width=768,
                hidden_size=self.backbone.hidden_size,
                slots=self.backbone.in_context_len,
            )

    @property
    def architecture_contract(self) -> dict:
        """Return the immutable backbone contract used by this run."""

        backbone = self.backbone
        patch_size = backbone.patch_size
        if isinstance(patch_size, (tuple, list)):
            if len(patch_size) != 1 or patch_size[0] != patch_size[-1]:
                raise ValueError(f"non-square patch size is unsupported: {patch_size!r}")
            patch_size = patch_size[0]
        return {
            "model": self.model_name,
            "kind": self.kind,
            "image_size": self.image_size,
            "depth": len(backbone.blocks),
            "hidden_size": int(backbone.hidden_size),
            "num_heads": int(backbone.num_heads),
            "patch_size": int(patch_size),
            "in_context_len": int(backbone.in_context_len),
            "in_context_start": int(backbone.in_context_start),
            "specslot": bool(backbone.specslot),
            "text_adapter": {
                "text_width": 768,
                "context_slots": int(self.text_adapter.slots),
                "attention_heads": int(self.text_adapter.cross_attention.num_heads),
                "output_width": int(self.text_adapter.output_norm.normalized_shape[0]),
            },
        }

    def forward(
        self,
        state: torch.Tensor,
        timestep: torch.Tensor,
        text_states: torch.Tensor,
        text_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = self.text_adapter(text_states, text_valid_mask)
        labels = torch.zeros(
            state.shape[0], device=state.device, dtype=torch.long
        )
        return self.backbone(state, timestep, labels, context_tokens=context)
