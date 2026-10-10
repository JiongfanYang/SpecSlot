import math
import os
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from util.model_util import (
    RMSNorm,
    VisionRotaryEmbeddingFast,
    get_2d_sincos_pos_embed,
    rotate_half,
)


def _jit_compile_if_enabled(function):
    if os.environ.get("SPECSLOT_DISABLE_TORCH_COMPILE", "0") == "1":
        return function
    if os.environ.get("WORLD_SIZE", "1") != "1" and os.environ.get(
        "SPECSLOT_ALLOW_DISTRIBUTED_BLOCK_COMPILE", "0"
    ) != "1":
        return function
    return torch.compile(function)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def create_dct_matrix(n, device=None, dtype=torch.float32):
    k = torch.arange(n, device=device, dtype=dtype).unsqueeze(1)
    i = torch.arange(n, device=device, dtype=dtype).unsqueeze(0)
    matrix = torch.cos(math.pi / n * (i + 0.5) * k)
    matrix[0] *= math.sqrt(1.0 / n)
    matrix[1:] *= math.sqrt(2.0 / n)
    return matrix


def dct_2d(x, dct_mat):
    return torch.matmul(dct_mat, torch.matmul(x, dct_mat.transpose(-1, -2)))


def idct_2d(x, dct_mat):
    return torch.matmul(dct_mat.transpose(-1, -2), torch.matmul(x, dct_mat))


def window_partition(x, window_size):
    batch, channels, height, width = x.shape
    if height % window_size or width % window_size:
        raise ValueError("image shape is not divisible by window_size")
    x = x.reshape(
        batch, channels, height // window_size, window_size,
        width // window_size, window_size,
    )
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    return x.reshape(
        batch, (height // window_size) * (width // window_size),
        channels, window_size, window_size,
    )


def window_reverse(windows, window_size, height, width):
    batch, count, channels, _, _ = windows.shape
    grid_h = height // window_size
    grid_w = width // window_size
    if count != grid_h * grid_w:
        raise ValueError("window count does not match image geometry")
    x = windows.reshape(batch, grid_h, grid_w, channels, window_size, window_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    return x.reshape(batch, channels, height, width)


def build_radial_indices(size):
    coordinates = [
        (row * row + col * col, row + col, row, col)
        for row in range(size)
        for col in range(size)
    ]
    coordinates.sort()
    return torch.tensor(
        [row * size + col for _, _, row, col in coordinates], dtype=torch.long
    )


class SpectralTokenizer(nn.Module):
    """Lapped DCT analysis and synthesis for fixed-support SpecSlot tokens."""

    def __init__(
        self, img_size, window_size, in_channels, embed_dim, n_max,
        fixed_support=True, frequency_order="radial", window_stride=None,
        create_frequency_projection=True,
    ):
        super().__init__()
        window_stride = window_size if window_stride is None else int(window_stride)
        if not 0 < window_stride <= window_size:
            raise ValueError("window_stride must lie in (0, window_size]")
        if img_size % window_stride or (window_size - window_stride) % 2:
            raise ValueError("invalid lapped-window geometry")
        if frequency_order != "radial" or not fixed_support:
            raise ValueError("SpecSlot requires radial fixed full-spectrum support")
        if n_max <= 0:
            raise ValueError("n_max must be positive")
        self.img_size = int(img_size)
        self.window_size = int(window_size)
        self.window_stride = window_stride
        self.window_padding = (window_size - window_stride) // 2
        self.lapped = window_stride < window_size
        self.in_channels = int(in_channels)
        self.embed_dim = int(embed_dim)
        self.fixed_support = True
        self.frequency_order = "radial"
        self.num_windows_h = img_size // window_stride
        self.num_windows_w = img_size // window_stride
        self.num_windows = self.num_windows_h * self.num_windows_w
        self.freq_per_window = window_size * window_size
        if n_max % self.num_windows:
            raise ValueError("n_max must be divisible by the window count")
        self.N_max = int(n_max)
        self.max_tokens_per_window = self.N_max // self.num_windows
        if self.max_tokens_per_window <= 0 or \
                self.freq_per_window % self.max_tokens_per_window:
            raise ValueError("token allocation must evenly cover each DCT window")
        self.output_expand = self.freq_per_window // self.max_tokens_per_window
        self.represented_coefficients_per_window = self.freq_per_window
        self.register_buffer(
            "frequency_order_idx", build_radial_indices(window_size), persistent=False
        )
        self.register_buffer(
            "dct_mat", create_dct_matrix(window_size), persistent=False
        )
        if self.lapped:
            one_dim = torch.hann_window(window_size, periodic=True)
            analysis_window = torch.sqrt(one_dim[:, None] * one_dim[None, :])
            padded_size = img_size + 2 * self.window_padding
            weights = analysis_window.square().reshape(1, -1, 1).expand(
                1, -1, self.num_windows
            )
            normalizer = F.fold(
                weights, output_size=(padded_size, padded_size),
                kernel_size=window_size, stride=window_stride,
            )
            crop = normalizer[
                :, :, self.window_padding:self.window_padding + img_size,
                self.window_padding:self.window_padding + img_size,
            ]
            if crop.min().item() <= 0:
                raise ValueError("lapped windows do not cover the full image")
        else:
            analysis_window = torch.ones(window_size, window_size)
            normalizer = torch.ones(1, 1, img_size, img_size)
        self.register_buffer("analysis_window", analysis_window, persistent=False)
        self.register_buffer("overlap_normalizer", normalizer, persistent=False)
        self.register_buffer(
            "spec_format_version", torch.tensor(8, dtype=torch.long), persistent=True
        )
        self.register_buffer(
            "spec_lapped_geometry",
            torch.tensor([
                img_size, window_size, window_stride, self.N_max,
                self.output_expand, 0,
            ], dtype=torch.long),
            persistent=True,
        )
        if create_frequency_projection:
            self.freq_proj = nn.Linear(
                in_channels * self.output_expand, embed_dim, bias=True
            )

    def analysis_windows(self, field):
        if field.ndim != 4 or field.shape[-2:] != (self.img_size, self.img_size):
            raise ValueError("analysis input has the wrong shape")
        if not self.lapped:
            return window_partition(field, self.window_size)
        padded = F.pad(
            field, (self.window_padding,) * 4, mode="reflect"
        )
        patches = F.unfold(
            padded, kernel_size=self.window_size, stride=self.window_stride
        )
        patches = patches.reshape(
            field.shape[0], field.shape[1], self.window_size,
            self.window_size, self.num_windows,
        ).permute(0, 4, 1, 2, 3).contiguous()
        window = self.analysis_window.to(device=field.device, dtype=field.dtype)
        return patches * window[None, None, None]

    def synthesis_windows(self, windows):
        if not self.lapped:
            return window_reverse(
                windows, self.window_size, self.img_size, self.img_size
            )
        window = self.analysis_window.to(device=windows.device, dtype=windows.dtype)
        weighted = windows * window[None, None, None]
        columns = weighted.permute(0, 2, 3, 4, 1).reshape(
            windows.shape[0], -1, self.num_windows
        )
        padded_size = self.img_size + 2 * self.window_padding
        folded = F.fold(
            columns, output_size=(padded_size, padded_size),
            kernel_size=self.window_size, stride=self.window_stride,
        )
        normalizer = self.overlap_normalizer.to(
            device=windows.device, dtype=windows.dtype
        ).clamp_min(torch.finfo(windows.dtype).eps)
        folded = folded / normalizer
        padding = self.window_padding
        return folded[
            :, :, padding:padding + self.img_size,
            padding:padding + self.img_size,
        ]

    def image_to_frequency(self, field):
        windows = self.analysis_windows(field)
        dct_mat = self.dct_mat.to(device=field.device, dtype=field.dtype)
        frequency = dct_2d(
            windows.reshape(-1, self.window_size, self.window_size), dct_mat
        )
        return frequency.reshape(
            field.shape[0], self.num_windows, self.in_channels,
            self.freq_per_window,
        )

    def frequency_to_image(self, frequency):
        expected = (self.num_windows, self.in_channels, self.freq_per_window)
        if frequency.shape[1:] != expected:
            raise ValueError("frequency tensor does not match tokenizer geometry")
        dct_mat = self.dct_mat.to(
            device=frequency.device, dtype=frequency.dtype
        )
        spatial = idct_2d(
            frequency.reshape(-1, self.window_size, self.window_size), dct_mat
        ).reshape(
            frequency.shape[0], self.num_windows, self.in_channels,
            self.window_size, self.window_size,
        )
        return self.synthesis_windows(spatial)

    def forward(self, x):
        frequency = self.image_to_frequency(x)
        ordered = frequency[:, :, :, self.frequency_order_idx.to(x.device)]
        tokens = ordered.reshape(
            x.shape[0], self.num_windows, self.in_channels,
            self.max_tokens_per_window, self.output_expand,
        ).permute(0, 1, 3, 4, 2).reshape(
            x.shape[0], self.N_max, self.output_expand * self.in_channels
        )
        tokens = self.freq_proj(tokens)
        return tokens

    def unpatchify(self, freq_tokens, k_per_window):
        if k_per_window != self.max_tokens_per_window:
            raise ValueError("SpecSlot synthesis requires full slot support")
        batch, token_count, width = freq_tokens.shape
        if token_count != self.N_max or width != self.output_expand * self.in_channels:
            raise ValueError("frequency token shape does not match tokenizer geometry")
        ordered = freq_tokens.reshape(
            batch, self.num_windows, self.max_tokens_per_window,
            self.output_expand, self.in_channels,
        ).reshape(
            batch, self.num_windows, self.freq_per_window, self.in_channels
        ).permute(0, 1, 3, 2).contiguous()
        native = torch.zeros_like(ordered)
        native.index_copy_(
            3, self.frequency_order_idx.to(freq_tokens.device), ordered
        )
        return self.frequency_to_image(native)


class SpatialEvidenceTokenizer(SpectralTokenizer):
    """Spatial anchor evidence expanded into fixed spectral slots."""

    def __init__(self, *args, bottleneck_dim, spatial_activation="linear", **kwargs):
        super().__init__(*args, create_frequency_projection=False, **kwargs)
        if spatial_activation not in ("linear", "gelu"):
            raise ValueError("spatial_activation must be linear or gelu")
        cell = self.window_stride
        self.spatial_activation_name = spatial_activation
        self.spatial_activation = (
            nn.Identity() if spatial_activation == "linear"
            else nn.GELU(approximate="none")
        )
        self.spatial_proj1 = nn.Conv2d(
            self.in_channels, bottleneck_dim, kernel_size=cell,
            stride=cell, bias=False,
        )
        self.spatial_proj2 = nn.Conv2d(
            bottleneck_dim, self.max_tokens_per_window * self.embed_dim,
            kernel_size=1, bias=True,
        )

    def enable_spatial_input_mixing(self):
        if hasattr(self, "spatial_input_mix"):
            return
        state = torch.get_rng_state()
        try:
            self.spatial_input_mix = nn.Conv2d(
                self.spatial_proj1.out_channels,
                self.spatial_proj1.out_channels,
                kernel_size=3, padding=1,
                groups=self.spatial_proj1.out_channels, bias=False,
                padding_mode="reflect",
            )
        finally:
            torch.set_rng_state(state)
        nn.init.zeros_(self.spatial_input_mix.weight)

    def forward(self, x):
        features = self.spatial_proj1(x)
        if hasattr(self, "spatial_input_mix"):
            features = features + self.spatial_input_mix(features)
        tokens = self.spatial_proj2(self.spatial_activation(features))
        tokens = tokens.flatten(2).transpose(1, 2).reshape(
            x.shape[0], self.N_max, self.embed_dim
        )
        return tokens


class SpatialResidualProjection(nn.Module):
    """Per-slot spatial residual head for the final hybrid decoder."""

    def __init__(self, bands, in_features, out_features):
        super().__init__()
        if out_features % bands:
            raise ValueError("residual output width must be divisible by slots")
        self.bands = int(bands)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.out_features_per_band = out_features // bands
        self.weight = nn.Parameter(
            torch.empty(bands, self.out_features_per_band, in_features)
        )
        self.bias = nn.Parameter(torch.empty(bands, self.out_features_per_band))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.reshape(-1, self.in_features))
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return torch.einsum("...kh,koh->...ko", x, self.weight) + self.bias


class BottleneckPatchEmbed(nn.Module):
    """Vanilla spatial patch embedding used by the reference baselines."""

    def __init__(self, img_size, patch_size, in_chans, pca_dim, embed_dim, bias=True):
        super().__init__()
        if img_size % patch_size:
            raise ValueError("img_size must be divisible by patch_size")
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (img_size // patch_size) ** 2
        self.in_chans = in_chans
        self.proj1 = nn.Conv2d(
            in_chans, pca_dim, kernel_size=patch_size,
            stride=patch_size, bias=False,
        )
        self.proj2 = nn.Conv2d(pca_dim, embed_dim, kernel_size=1, bias=bias)

    def forward(self, x):
        if x.shape[1] != self.in_chans or x.shape[-2:] != self.img_size:
            raise ValueError("input shape does not match the patch embedder")
        return self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        frequencies = torch.exp(
            -math.log(max_period)
            * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        angles = t[:, None].float() * frequencies[None]
        embedding = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes

    def forward(self, labels):
        return self.embedding_table(labels)


def scaled_dot_product_attention(query, key, value, dropout_p=0.0, attn_mask=None):
    if attn_mask is not None and attn_mask.dim() == 2:
        attn_mask = attn_mask[:, None, None, :].bool()
    return F.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask, dropout_p=dropout_p
    )


class Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, qk_norm=True,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope=None, attn_mask=None):
        batch, length, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch, length, 3, self.num_heads, channels // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        query, key = self.q_norm(query), self.k_norm(key)
        if rope is not None:
            if isinstance(rope, tuple):
                cos, sin = rope
                query = query * cos + rotate_half(query) * sin
                key = key * cos + rotate_half(key) * sin
            else:
                query, key = rope(query), rope(key)
        output = scaled_dot_product_attention(
            query, key, value,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            attn_mask=attn_mask,
        )
        output = output.transpose(1, 2).reshape(batch, length, channels)
        return self.proj_drop(self.proj(output))


class SwiGLUFFN(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.0, bias=True):
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(self.ffn_dropout(F.silu(x1) * x2))


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size)
        )

    @_jit_compile_if_enabled
    def forward(self, x, c, skip_linear=False):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return x if skip_linear else self.linear(x)


class JiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
            attn_drop=attn_drop, proj_drop=proj_drop,
        )
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = SwiGLUFFN(hidden_size, int(hidden_size * mlp_ratio), drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size)
        )

    @_jit_compile_if_enabled
    def forward(self, x, c, feat_rope=None, attn_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
            rope=feat_rope, attn_mask=attn_mask,
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class JiT(nn.Module):
    """JiT backbone with the final SpecSlot and vanilla baseline paths."""

    def __init__(
        self, input_size=256, patch_size=16, window_size=None,
        in_channels=3, out_channels=3, hidden_size=768, depth=12, num_heads=12,
        mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0,
        activation_checkpointing=False, num_classes=1000, specslot=False,
        spec_n_max=None, spec_fixed_support=True,
        spec_frequency_order="radial", spec_window_stride=None,
        spec_factorized_position=False, spec_hierarchical=False, spec_global_depth=6,
        spec_hierarchy_mode="interleave", spec_semantic_blocks=1,
        spec_spatialized_input=False, spec_spatialized_bottleneck_dim=1536,
        spec_spatialized_activation="linear", spec_spatial_input_mixing=False,
        spec_hybrid_output=False, spec_local_group_chunk_size=16384,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.patch_size = int(patch_size)
        self.window_size = input_size if window_size is None else int(window_size)
        self.hidden_size = int(hidden_size)
        self.out_channels = int(out_channels)
        self.num_classes = int(num_classes)
        self.num_heads = int(num_heads)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.specslot = bool(specslot)
        self.spec_factorized_position = bool(spec_factorized_position)
        self.spec_hierarchical = bool(spec_hierarchical)
        self.spec_global_depth = int(spec_global_depth)
        self.spec_hierarchy_mode = str(spec_hierarchy_mode)
        self.spec_semantic_blocks = int(spec_semantic_blocks)
        self.spec_spatialized_input = bool(spec_spatialized_input)
        self.spec_spatialized_bottleneck_dim = int(spec_spatialized_bottleneck_dim)
        self.spec_spatialized_activation = str(spec_spatialized_activation)
        self.spec_spatial_input_mixing = bool(spec_spatial_input_mixing)
        self.spec_hybrid_output = bool(spec_hybrid_output)
        self.spec_local_group_chunk_size = int(spec_local_group_chunk_size)

        if self.specslot:
            if not (
                spec_fixed_support and spec_factorized_position and spec_hierarchical and
                spec_hierarchy_mode == "interleave" and spec_semantic_blocks == 1 and
                spec_spatialized_input and spec_hybrid_output and
                spec_frequency_order == "radial"
            ):
                raise ValueError("only the final SpecSlot contract is supported")
            if spec_window_stride is None or spec_n_max is None:
                raise ValueError("SpecSlot requires explicit window, stride, and N")
            self.x_embedder = SpatialEvidenceTokenizer(
                img_size=input_size, window_size=self.window_size,
                window_stride=spec_window_stride, in_channels=in_channels,
                embed_dim=hidden_size, n_max=spec_n_max,
                fixed_support=True, frequency_order="radial",
                bottleneck_dim=self.spec_spatialized_bottleneck_dim,
                spatial_activation=self.spec_spatialized_activation,
            )
            if self.x_embedder.num_windows != 256:
                raise ValueError("release SpecSlot uses a 16x16 anchor grid")
            self.freq_window_pos_embed = nn.Parameter(
                torch.zeros(1, self.x_embedder.num_windows, hidden_size),
                requires_grad=False,
            )
            self.freq_band_pos_embed = nn.Parameter(
                torch.zeros(1, self.x_embedder.max_tokens_per_window, hidden_size)
            )
            self.freq_output_proj = nn.Linear(
                hidden_size, in_channels * self.x_embedder.output_expand
            )
            self.spec_global_block_indices = ((0, 1), (4, 5), (8, 9))
            self.spec_local_block_indices = ((2, 3), (6, 7), (10, 11))
            self.spec_hierarchy_cycles = 3
            self.register_buffer(
                "spec_architecture_contract",
                torch.tensor([
                    3, self.spec_global_depth, 1, 8, 1,
                    self.spec_spatialized_bottleneck_dim, 17, 1, 15, 1,
                ], dtype=torch.long), persistent=True,
            )
        else:
            self.x_embedder = BottleneckPatchEmbed(
                input_size, patch_size, in_channels, 128, hidden_size,
            )
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.x_embedder.num_patches, hidden_size),
                requires_grad=False,
            )
        # Class-context slots are part of the global sequence for both
        # SpecSlot and vanilla JiT.  SpecSlot carries their updated state
        # across its repeated global/local cycles.
        self.in_context_len = 32
        self.in_context_start = 4
        if self.in_context_len:
            self.in_context_posemb = nn.Parameter(
                torch.zeros(1, self.in_context_len, hidden_size)
            )
            nn.init.normal_(self.in_context_posemb, std=0.02)

        half_head_dim = hidden_size // num_heads // 2
        grid_side = input_size // patch_size
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim, pt_seq_len=grid_side, num_cls_token=0
        )
        self.feat_rope_incontext = VisionRotaryEmbeddingFast(
            dim=half_head_dim, pt_seq_len=grid_side,
            num_cls_token=self.in_context_len,
        )
        self.blocks = nn.ModuleList([
            JiTBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio,
                attn_drop=attn_drop if depth // 4 <= i < depth * 3 // 4 else 0.0,
                proj_drop=proj_drop if depth // 4 <= i < depth * 3 // 4 else 0.0,
            ) for i in range(depth)
        ])
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()
        if self.specslot:
            rng_state = torch.get_rng_state()
            try:
                self.hybrid_spatial_residual_proj = SpatialResidualProjection(
                    self.x_embedder.max_tokens_per_window, hidden_size,
                    in_channels * self.x_embedder.window_stride ** 2,
                )
            finally:
                torch.set_rng_state(rng_state)
            nn.init.zeros_(self.hybrid_spatial_residual_proj.weight)
            nn.init.zeros_(self.hybrid_spatial_residual_proj.bias)
        if self.specslot and self.spec_spatial_input_mixing:
            self.x_embedder.enable_spatial_input_mixing()

    def initialize_weights(self):
        def init_linear(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, SpatialResidualProjection):
                module.reset_parameters()
        self.apply(init_linear)
        if self.specslot:
            positions = get_2d_sincos_pos_embed(
                self.hidden_size, int(self.x_embedder.num_windows ** 0.5)
            )
            self.freq_window_pos_embed.data.copy_(
                torch.from_numpy(positions).float().unsqueeze(0)
            )
            nn.init.normal_(self.freq_band_pos_embed, std=0.02)
            for block in self.blocks:
                nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.02)
                nn.init.zeros_(block.adaLN_modulation[-1].bias)
            nn.init.normal_(self.final_layer.adaLN_modulation[-1].weight, std=0.02)
            nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
            nn.init.xavier_uniform_(self.x_embedder.spatial_proj1.weight.reshape(
                self.x_embedder.spatial_proj1.out_channels, -1
            ))
            nn.init.xavier_uniform_(self.x_embedder.spatial_proj2.weight.reshape(
                self.x_embedder.spatial_proj2.out_channels, -1
            ))
            if self.x_embedder.spatial_proj2.bias is not None:
                nn.init.zeros_(self.x_embedder.spatial_proj2.bias)
        else:
            positions = get_2d_sincos_pos_embed(
                self.hidden_size, int(self.x_embedder.num_patches ** 0.5)
            )
            self.pos_embed.data.copy_(torch.from_numpy(positions).float().unsqueeze(0))
            nn.init.xavier_uniform_(self.x_embedder.proj1.weight.reshape(
                self.x_embedder.proj1.weight.shape[0], -1
            ))
            nn.init.xavier_uniform_(self.x_embedder.proj2.weight.reshape(
                self.x_embedder.proj2.weight.shape[0], -1
            ))
            if self.x_embedder.proj2.bias is not None:
                nn.init.zeros_(self.x_embedder.proj2.bias)
            for block in self.blocks:
                nn.init.zeros_(block.adaLN_modulation[-1].weight)
                nn.init.zeros_(block.adaLN_modulation[-1].bias)
            nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
            nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
            nn.init.zeros_(self.final_layer.linear.weight)
            nn.init.zeros_(self.final_layer.linear.bias)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

    def unpatchify(self, x, patch_size):
        channels = self.out_channels
        side = int(x.shape[1] ** 0.5)
        if side * side != x.shape[1]:
            raise ValueError("patch token count is not a square")
        x = x.reshape(x.shape[0], side, side, patch_size, patch_size, channels)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], channels, side * patch_size, side * patch_size)

    def _run_block(self, block, x, c, **kwargs):
        if self.activation_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(partial(block, **kwargs), x, c, use_reentrant=False)
        return block(x, c, **kwargs)

    def _run_local_block(self, block, x, c, attn_mask=None):
        outputs = []
        for start in range(0, x.shape[0], self.spec_local_group_chunk_size):
            end = min(start + self.spec_local_group_chunk_size, x.shape[0])
            outputs.append(self._run_block(
                block, x[start:end], c[start:end],
                attn_mask=(
                    None if attn_mask is None else attn_mask[start:end]
                ),
            ))
        return torch.cat(outputs, dim=0) if len(outputs) > 1 else outputs[0]

    def _context_tokens(self, class_embedding, context_tokens=None):
        """Build the 32 global context slots from labels or external text."""
        if context_tokens is not None:
            if context_tokens.ndim != 3 or context_tokens.shape[1] != self.in_context_len:
                raise ValueError(
                    "context_tokens must have shape [B, in_context_len, D], got "
                    f"{tuple(context_tokens.shape)}"
                )
            if (
                context_tokens.shape[0] != class_embedding.shape[0]
                or context_tokens.shape[2] != self.hidden_size
            ):
                raise ValueError(
                    "context_tokens batch/width mismatch: "
                    f"tokens={tuple(context_tokens.shape)}, "
                    f"labels={tuple(class_embedding.shape)}, hidden_size={self.hidden_size}"
                )
            return context_tokens.to(
                device=class_embedding.device, dtype=class_embedding.dtype
            ) + self.in_context_posemb
        return (
            class_embedding[:, None, :].expand(-1, self.in_context_len, -1)
            + self.in_context_posemb
        )

    def _forward_specslot(
        self, tokens, condition, class_embedding, context_tokens=None
    ):
        batch, windows, slots, hidden = tokens.shape
        state = tokens
        context_state = self._context_tokens(class_embedding, context_tokens)
        context_mask = torch.ones(
            batch, self.in_context_len, dtype=torch.bool,
            device=tokens.device,
        )
        for global_indices, local_indices in zip(
            self.spec_global_block_indices, self.spec_local_block_indices
        ):
            carrier = state[:, :, 0, :]
            global_tokens = torch.cat([context_state, carrier], dim=1)
            global_mask = torch.cat([
                context_mask,
                torch.ones(
                    batch, windows, dtype=torch.bool, device=tokens.device
                ),
            ], dim=1)
            for index in global_indices:
                global_tokens = self._run_block(
                    self.blocks[index], global_tokens, condition,
                    attn_mask=global_mask,
                )
            context_state = global_tokens[:, :self.in_context_len]
            carrier = global_tokens[:, self.in_context_len:]
            routed = torch.cat([carrier[:, :, None, :], state[:, :, 1:, :]], dim=2)
            local = routed.reshape(batch * windows, slots, hidden)
            local_condition = condition[:, None, :].expand(
                batch, windows, hidden
            ).reshape(batch * windows, hidden)
            local_mask = torch.ones(
                batch * windows, slots, dtype=torch.bool,
                device=tokens.device,
            )
            for index in local_indices:
                local = self._run_local_block(
                    self.blocks[index], local, local_condition, local_mask
                )
            state = local.reshape(batch, windows, slots, hidden)
        return state.reshape(batch, windows * slots, hidden)

    def _decode_spatial_residual(self, hidden):
        batch = hidden.shape[0]
        windows = self.x_embedder.num_windows
        slots = self.x_embedder.max_tokens_per_window
        stride = self.x_embedder.window_stride
        slot_hidden = hidden.reshape(batch, windows, slots, self.hidden_size)
        rows = math.isqrt(slots)
        while slots % rows:
            rows -= 1
        cols = slots // rows
        if stride % rows or stride % cols:
            raise ValueError("slot grid does not tile the spatial anchor")
        tiles = self.hybrid_spatial_residual_proj(slot_hidden).reshape(
            batch, windows, rows, cols, self.x_embedder.in_channels,
            stride // rows, stride // cols,
        )
        patches = tiles.permute(0, 1, 4, 2, 5, 3, 6).reshape(
            batch, windows, -1
        )
        patches = patches.reshape(
            batch, self.x_embedder.num_windows_h,
            self.x_embedder.num_windows_w, self.x_embedder.in_channels,
            stride, stride,
        )
        return patches.permute(0, 3, 1, 4, 2, 5).reshape(
            batch, self.x_embedder.in_channels,
            self.x_embedder.img_size, self.x_embedder.img_size,
        )

    def _decode_spectral(self, hidden, condition):
        hidden = self.final_layer(hidden, condition, skip_linear=True)
        frequency = self.freq_output_proj(hidden)
        spectral = self.x_embedder.unpatchify(
            frequency, self.x_embedder.max_tokens_per_window
        )
        return spectral + self._decode_spatial_residual(hidden)

    def forward(self, x, t, labels, context_tokens=None):
        timestep_embedding = self.t_embedder(t)
        if context_tokens is None:
            class_embedding = self.y_embedder(labels)
        else:
            # Text conditioning is carried by the explicit global sequence.
            # Keep AdaLN label-free so the T2I path does not inject a class
            # embedding into every local slot.
            class_embedding = torch.zeros_like(timestep_embedding)
        condition = timestep_embedding + class_embedding
        if self.specslot:
            tokens = self.x_embedder(x)
            position = (
                self.freq_window_pos_embed[:, :, None, :]
                + self.freq_band_pos_embed[:, None, :, :]
            ).reshape(1, self.x_embedder.N_max, self.hidden_size)
            tokens = tokens + position.to(device=tokens.device, dtype=tokens.dtype)
            tokens = self._forward_specslot(
                tokens.reshape(
                    x.shape[0], self.x_embedder.num_windows,
                    self.x_embedder.max_tokens_per_window, self.hidden_size,
                ), condition, class_embedding, context_tokens=context_tokens
            )
            return self._decode_spectral(tokens, condition)

        tokens = self.x_embedder(x)
        tokens = tokens + self.pos_embed.to(device=tokens.device, dtype=tokens.dtype)
        for index, block in enumerate(self.blocks):
            if index == self.in_context_start:
                context = self._context_tokens(class_embedding, context_tokens)
                tokens = torch.cat([context, tokens], dim=1)
            tokens = self._run_block(
                block, tokens, condition,
                feat_rope=self.feat_rope if index < self.in_context_start
                else self.feat_rope_incontext,
            )
        tokens = tokens[:, self.in_context_len:]
        return self.unpatchify(self.final_layer(tokens, condition), self.patch_size)


def JiT_B_8(**kwargs):
    return JiT(depth=12, hidden_size=768, num_heads=12,
               patch_size=8, **kwargs)


def JiT_B_16(**kwargs):
    return JiT(depth=12, hidden_size=768, num_heads=12,
               patch_size=16, **kwargs)


def JiT_B_32(**kwargs):
    return JiT(depth=12, hidden_size=768, num_heads=12,
               patch_size=32, **kwargs)


JiT_models = {
    "JiT-B/8": JiT_B_8,
    "JiT-B/16": JiT_B_16,
    "JiT-B/32": JiT_B_32,
}
