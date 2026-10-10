import copy

import torch
import torch.nn as nn

from model_jit import JiT_models


class Denoiser(nn.Module):
    """Rectified-flow wrapper for SpecSlot and the vanilla JiT baselines."""

    def __init__(self, args):
        super().__init__()
        if args.pred_param != "x":
            raise ValueError("The release uses clean-image x prediction")

        self.img_size = int(args.img_size)
        self.num_classes = int(args.class_num)
        self.pred_param = "x"
        self.specslot = bool(args.specslot)
        self.label_drop_prob = float(args.label_drop_prob)
        self.P_mean = float(args.P_mean)
        self.P_std = float(args.P_std)
        self.t_eps = float(args.t_eps)
        self.noise_scale = float(args.noise_scale)
        self.ema_decay1 = float(args.ema_decay1)
        self.ema_decay2 = float(args.ema_decay2)
        self.ema_params1 = None
        self.ema_params2 = None

        self.net = JiT_models[args.model](
            input_size=args.img_size,
            window_size=args.window_size,
            in_channels=3,
            out_channels=3,
            num_classes=args.class_num,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
            activation_checkpointing=args.activation_checkpointing,
            specslot=args.specslot,
            spec_n_max=args.spec_n_max,
            spec_fixed_support=args.spec_fixed_support,
            spec_frequency_order=args.spec_frequency_order,
            spec_window_stride=args.spec_window_stride,
            spec_factorized_position=args.spec_factorized_position,
            spec_hierarchical=args.spec_hierarchical,
            spec_global_depth=args.spec_global_depth,
            spec_hierarchy_mode=args.spec_hierarchy_mode,
            spec_semantic_blocks=args.spec_semantic_blocks,
            spec_spatialized_input=args.spec_spatialized_input,
            spec_spatialized_bottleneck_dim=args.spec_spatialized_bottleneck_dim,
            spec_spatialized_activation=args.spec_spatialized_activation,
            spec_spatial_input_mixing=args.spec_spatial_input_mixing,
            spec_hybrid_output=args.spec_hybrid_output,
            spec_local_group_chunk_size=args.spec_local_group_chunk_size,
        )

        if self.specslot:
            nn.init.zeros_(self.net.freq_output_proj.weight)
            nn.init.zeros_(self.net.freq_output_proj.bias)
            nn.init.zeros_(self.net.hybrid_spatial_residual_proj.weight)
            nn.init.zeros_(self.net.hybrid_spatial_residual_proj.bias)

        self.method = args.sampling_method
        self.steps = int(args.num_sampling_steps)
        self.sampling_grid = args.sampling_grid
        self.cfg_scale = float(args.cfg)
        self.cfg_interval = (float(args.interval_min), float(args.interval_max))
        self.last_inference_latency_ms = None

    def drop_labels(self, labels):
        dropped = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        return torch.where(dropped, torch.full_like(labels, self.num_classes), labels)

    def sample_t(self, batch_size, device):
        values = torch.randn(batch_size, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(values)

    def _network(self, state, timestep, labels):
        return self.net(state, timestep.reshape(-1), labels)

    def _velocity(self, state, timestep, labels):
        prediction = self._network(state, timestep, labels)
        denominator = (1.0 - timestep).clamp_min(self.t_eps)
        return (prediction - state) / denominator.view(-1, 1, 1, 1)

    def forward(self, images, labels):
        labels = self.drop_labels(labels) if self.training else labels
        timestep = self.sample_t(images.shape[0], images.device)
        noise = torch.randn_like(images) * self.noise_scale
        t = timestep.view(-1, 1, 1, 1)
        state = t * images + (1.0 - t) * noise
        target = images - noise
        prediction = self._network(state, timestep, labels)
        velocity = (prediction - state) / (1.0 - t).clamp_min(self.t_eps)
        return (velocity.float() - target.float()).square().mean()

    def _sampling_timesteps(self, batch_size, device):
        if self.sampling_grid == "legacy_uniform":
            grid = torch.linspace(0.0, 1.0, self.steps + 1, device=device)
        elif self.sampling_grid == "endpoint_aware":
            learned = torch.linspace(
                0.0, 1.0 - self.t_eps, self.steps, device=device
            )
            grid = torch.cat([learned, torch.ones(1, device=device)])
        else:
            raise ValueError(f"unsupported sampling grid: {self.sampling_grid}")
        return grid[:, None, None, None].expand(-1, batch_size, 1, 1)

    @torch.no_grad()
    def _guided_velocity(self, state, timestep, labels):
        conditional = self._velocity(state, timestep, labels)
        low, high = self.cfg_interval
        active = self.cfg_scale != 1.0 and bool(
            ((timestep < high) & ((low == 0.0) | (timestep > low))).any()
        )
        if not active:
            return conditional
        unconditional = self._velocity(
            state, timestep, torch.full_like(labels, self.num_classes)
        )
        scale = torch.where(
            (timestep < high) & ((low == 0.0) | (timestep > low)),
            torch.full_like(timestep, self.cfg_scale),
            torch.ones_like(timestep),
        ).view(-1, 1, 1, 1)
        return unconditional + scale * (conditional - unconditional)

    @torch.no_grad()
    def _euler_step(self, state, timestep, next_timestep, labels):
        velocity = self._guided_velocity(state, timestep, labels)
        delta = (next_timestep - timestep).view(-1, 1, 1, 1)
        return state + delta * velocity

    @torch.no_grad()
    def _heun_step(self, state, timestep, next_timestep, labels):
        velocity = self._guided_velocity(state, timestep, labels)
        delta = (next_timestep - timestep).view(-1, 1, 1, 1)
        euler_state = state + delta * velocity
        next_velocity = self._guided_velocity(euler_state, next_timestep, labels)
        return state + delta * 0.5 * (velocity + next_velocity)

    @torch.no_grad()
    def generate(self, labels, initial_noise=None):
        shape = (labels.shape[0], 3, self.img_size, self.img_size)
        if initial_noise is None:
            state = torch.randn(shape, device=labels.device) * self.noise_scale
        else:
            if tuple(initial_noise.shape) != shape or initial_noise.device != labels.device:
                raise ValueError("initial_noise does not match the requested batch")
            state = initial_noise * self.noise_scale

        timesteps = self._sampling_timesteps(labels.shape[0], labels.device)
        if self.method == "heun":
            stepper = self._heun_step
        elif self.method == "euler":
            stepper = self._euler_step
        else:
            raise ValueError(f"unsupported sampling method: {self.method}")

        start = torch.cuda.Event(enable_timing=True) if state.is_cuda else None
        end = torch.cuda.Event(enable_timing=True) if state.is_cuda else None
        if start is not None:
            start.record()
        for index in range(self.steps - 1):
            state = stepper(
                state, timesteps[index], timesteps[index + 1], labels
            )
        state = self._euler_step(
            state, timesteps[-2], timesteps[-1], labels
        )
        if end is not None:
            end.record()
            torch.cuda.synchronize(state.device)
            self.last_inference_latency_ms = start.elapsed_time(end)
        return state

    @torch.no_grad()
    def update_ema(self):
        parameters = list(self.parameters())
        for target, source in zip(self.ema_params1, parameters):
            target.mul_(self.ema_decay1).add_(source, alpha=1.0 - self.ema_decay1)
        for target, source in zip(self.ema_params2, parameters):
            target.mul_(self.ema_decay2).add_(source, alpha=1.0 - self.ema_decay2)

    def initialize_ema(self):
        parameters = list(self.parameters())
        self.ema_params1 = copy.deepcopy(parameters)
        self.ema_params2 = copy.deepcopy(parameters)
