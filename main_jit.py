import argparse
import datetime
import math
import multiprocessing as mp
import os
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from denoiser import Denoiser
from engine_jit import evaluate, train_one_epoch
from util.crop import center_crop_arr
import util.misc as misc
from util.wandb_logger import WandbLogger


def _center_crop_transform(image, image_size):
    """Pickle-safe ImageFolder transform for spawned DataLoader workers."""
    return center_crop_arr(image, image_size)


def get_args_parser():
    parser = argparse.ArgumentParser("SpecSlot and JiT training")

    parser.add_argument("--model", default="JiT-B/16")
    parser.add_argument("--img_size", default=256, type=int)
    parser.add_argument("--window_size", default=None, type=int)

    parser.add_argument("--specslot", action="store_true")
    parser.add_argument("--spec_n_max", default=None, type=int)
    parser.add_argument("--expected_token_count", default=None, type=int)
    parser.add_argument("--spec_fixed_support", action="store_true")
    parser.add_argument("--spec_frequency_order", default="radial")
    parser.add_argument("--spec_window_stride", default=None, type=int)
    parser.add_argument("--spec_factorized_position", action="store_true")
    parser.add_argument("--spec_hierarchical", action="store_true")
    parser.add_argument("--spec_global_depth", default=6, type=int)
    parser.add_argument("--spec_hierarchy_mode", default="interleave")
    parser.add_argument("--spec_semantic_blocks", default=1, type=int)
    parser.add_argument("--spec_spatialized_input", action="store_true")
    parser.add_argument("--spec_spatialized_bottleneck_dim", default=1536, type=int)
    parser.add_argument("--spec_spatialized_activation", default="linear")
    parser.add_argument("--spec_spatial_input_mixing", action="store_true")
    parser.add_argument("--spec_hybrid_output", action="store_true")
    parser.add_argument("--spec_local_group_chunk_size", default=16384, type=int)

    parser.add_argument("--attn_dropout", default=0.0, type=float)
    parser.add_argument("--proj_dropout", default=0.0, type=float)
    parser.add_argument("--activation_checkpointing", action="store_true")

    parser.add_argument("--epochs", default=120, type=int)
    parser.add_argument("--warmup_epochs", default=5, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--accum_iter", default=1, type=int)
    parser.add_argument("--lr", default=None, type=float)
    parser.add_argument("--blr", default=5e-5, type=float)
    parser.add_argument("--min_lr", default=0.0, type=float)
    parser.add_argument("--lr_schedule", default="constant")
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--grad_clip", default=0.0, type=float)
    parser.add_argument("--ema_decay1", default=0.9999, type=float)
    parser.add_argument("--ema_decay2", default=0.9996, type=float)
    parser.add_argument("--P_mean", default=-0.8, type=float)
    parser.add_argument("--P_std", default=0.8, type=float)
    parser.add_argument("--noise_scale", default=1.0, type=float)
    parser.add_argument("--t_eps", default=0.05, type=float)
    parser.add_argument("--label_drop_prob", default=0.1, type=float)
    parser.add_argument("--pred_param", default="x")

    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--train_rng_seed", default=None, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--num_workers", default=12, type=int)
    parser.add_argument("--data_loader_context", default="spawn")
    parser.add_argument("--disable_train_augmentation", action="store_true")
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    parser.add_argument("--sampling_method", default="heun")
    parser.add_argument("--num_sampling_steps", default=50, type=int)
    parser.add_argument("--sampling_grid", default="endpoint_aware")
    parser.add_argument("--generation_seed", default=None, type=int)
    parser.add_argument("--cfg", default=1.0, type=float)
    parser.add_argument("--interval_min", default=0.0, type=float)
    parser.add_argument("--interval_max", default=1.0, type=float)
    parser.add_argument("--num_images", default=50000, type=int)
    parser.add_argument("--eval_freq", default=5, type=int)
    parser.add_argument("--eval_completed_epochs", action="store_true")
    parser.add_argument("--online_eval", action="store_true")
    parser.add_argument("--evaluate_gen", action="store_true")
    parser.add_argument("--eval_weights", default="ema1", choices=("raw", "ema1", "ema2"))
    parser.add_argument("--gen_bsz", default=512, type=int)

    parser.add_argument("--data_path", default="./data/imagenet")
    parser.add_argument("--class_num", default=1000, type=int)
    parser.add_argument("--output_dir", default="./output_dir")
    parser.add_argument("--resume", default="")
    parser.add_argument("--resume_checkpoint", default="")
    parser.add_argument("--save_last_freq", default=10, type=int)
    parser.add_argument("--save_periodic_freq", default=20, type=int)
    parser.add_argument("--save_completed_epochs", action="store_true")
    parser.add_argument("--log_freq", default=10, type=int)
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--wandb_project", default="jit")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_run_name", default="")
    parser.add_argument("--wandb_mode", default="online", choices=("online", "offline", "disabled"))
    parser.add_argument("--wandb_num_vis", default=16, type=int)

    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")
    return parser


def _validate_args(args):
    if args.model not in ("JiT-B/8", "JiT-B/16", "JiT-B/32"):
        raise ValueError(f"unsupported model: {args.model}")
    if args.pred_param != "x":
        raise ValueError("only clean-image x prediction is supported")
    if args.accum_iter != 1:
        raise ValueError("the release uses --accum_iter 1")
    if args.epochs <= 0 or args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("epochs, batch_size, and num_workers must be valid")
    if args.generation_seed is not None and args.generation_seed < 0:
        raise ValueError("generation_seed must be non-negative")
    if not 0.0 < args.t_eps < 1.0:
        raise ValueError("t_eps must lie in (0, 1)")
    if args.cfg < 0.0 or not 0.0 <= args.interval_min <= args.interval_max <= 1.0:
        raise ValueError("invalid classifier-free guidance interval")
    if args.num_sampling_steps < 2 or args.gen_bsz <= 0 or args.num_images <= 0:
        raise ValueError("invalid sampling configuration")
    if args.sampling_method not in ("euler", "heun"):
        raise ValueError("sampling_method must be euler or heun")
    if args.sampling_grid not in ("legacy_uniform", "endpoint_aware"):
        raise ValueError("unsupported sampling_grid")

    if not args.specslot:
        return
    required = (
        args.model == "JiT-B/16",
        args.spec_fixed_support,
        args.spec_frequency_order == "radial",
        args.spec_factorized_position,
        args.spec_hierarchical,
        args.spec_hierarchy_mode == "interleave",
        args.spec_global_depth == 6,
        args.spec_semantic_blocks == 1,
        args.spec_spatialized_input,
        args.spec_spatialized_activation == "linear",
        args.spec_spatial_input_mixing,
        args.spec_hybrid_output,
    )
    if not all(required):
        raise ValueError("arguments do not match the public SpecSlot contract")
    if args.spec_window_stride is None or args.spec_n_max is None:
        raise ValueError("SpecSlot requires --spec_window_stride and --spec_n_max")
    geometry = (
        args.img_size, args.window_size, args.spec_window_stride,
        args.spec_n_max, args.spec_spatialized_bottleneck_dim,
    )
    supported = {
        (256, 32, 16, 512, 512): 5.0,
        (256, 32, 16, 1024, 1536): 4.0,
        (512, 64, 32, 512, 3072): 10.5,
        (512, 64, 32, 1024, 4096): 8.25,
    }
    if geometry not in supported:
        raise ValueError(f"unsupported SpecSlot geometry: {geometry}")
    if not math.isclose(args.cfg, supported[geometry]):
        raise ValueError(
            f"CFG {args.cfg} does not match the release setting "
            f"{supported[geometry]} for {geometry}"
        )


def _build_loader(args):
    transform_items = [
        transforms.Lambda(
            partial(_center_crop_transform, image_size=args.img_size)
        )
    ]
    if not args.disable_train_augmentation:
        transform_items.append(transforms.RandomHorizontalFlip())
    transform_items.append(transforms.PILToTensor())
    dataset = datasets.ImageFolder(
        os.path.join(args.data_path, "train"),
        transform=transforms.Compose(transform_items),
    )
    if len(dataset.classes) != args.class_num:
        raise ValueError(
            f"expected {args.class_num} classes, found {len(dataset.classes)}"
        )
    sampler = torch.utils.data.DistributedSampler(
        dataset, num_replicas=misc.get_world_size(), rank=misc.get_rank(), shuffle=True
    )
    generator = None
    if args.train_rng_seed is not None:
        generator = torch.Generator()
        generator.manual_seed(args.train_rng_seed + misc.get_rank())
    loader_kwargs = dict(
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        generator=generator,
    )
    if args.num_workers:
        loader_kwargs.update(
            multiprocessing_context=mp.get_context(args.data_loader_context),
            persistent_workers=True,
            prefetch_factor=2,
        )
    loader = torch.utils.data.DataLoader(dataset, **loader_kwargs)
    if len(loader) == 0:
        raise ValueError("the training loader has no complete batches")
    return dataset, sampler, loader


def _checkpoint_path(args):
    if args.resume_checkpoint:
        return Path(args.resume_checkpoint)
    if args.resume:
        return Path(args.resume) / "checkpoint-last.pth"
    return None


def _load_checkpoint(path, model, optimizer, device, evaluate_only):
    if path is None:
        model.initialize_ema()
        return 0
    if not path.is_file():
        if evaluate_only:
            raise FileNotFoundError(f"checkpoint does not exist: {path}")
        model.initialize_ema()
        return 0
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    for source, attribute in (("model_ema1", "ema_params1"), ("model_ema2", "ema_params2")):
        state = checkpoint[source]
        setattr(
            model, attribute,
            [state[name].to(device=device) for name, _ in model.named_parameters()],
        )
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("epoch", -1)) + 1


def _setup_model(args, device):
    model = Denoiser(args).to(device)
    expected = args.expected_token_count
    if expected is not None:
        observed = (
            model.net.x_embedder.N_max
            if args.specslot else model.net.x_embedder.num_patches
        )
        if observed != expected:
            raise ValueError(f"expected {expected} tokens, constructed {observed}")
    return model


def _should_save(epoch, frequency, completed):
    if frequency <= 0:
        return False
    return ((epoch + 1) if completed else epoch) % frequency == 0


def _evaluate_with_fixed_seed(
    model, args, device, rank, epoch, batch_size, log_writer, log_step
):
    generation_seed = (
        args.seed if args.generation_seed is None else args.generation_seed
    )
    eval_seed = generation_seed + rank
    with torch.random.fork_rng():
        torch.manual_seed(eval_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(eval_seed)
        with torch.no_grad():
            evaluate(
                model, args, epoch, batch_size=batch_size,
                log_writer=log_writer, log_step=log_step,
            )


def main(args):
    misc.init_distributed_mode(args)
    _validate_args(args)
    rank = misc.get_rank()
    device = torch.device(args.device)
    if device.type == "cuda":
        device = torch.device("cuda", args.gpu if args.distributed else 0)
        torch.cuda.set_device(device)
        cudnn.benchmark = True

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logger = None
    if misc.is_main_process() and args.wandb_mode != "disabled":
        logger = WandbLogger(args)

    loader = sampler = dataset = None
    if not args.evaluate_gen:
        dataset, sampler, loader = _build_loader(args)
        if misc.is_main_process():
            print(dataset)
            print("Sampler_train =", sampler)
            print(
                f"[Data Contract] classes={len(dataset.classes)}, "
                f"images={len(dataset)}, steps/epoch={len(loader)}, "
                f"class_head={dataset.classes[:min(5, len(dataset.classes))]}"
            )

    # Keep DDP eager during training.  This is the source recipe's setting;
    # block compilation is reserved for explicit single-process benchmarks.
    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.optimize_ddp = False
    model = _setup_model(args, device)

    optimizer = None
    updates_per_epoch = 1
    if loader is not None:
        effective_batch = args.batch_size * misc.get_world_size() * args.accum_iter
        if args.lr is None:
            args.lr = args.blr * effective_batch / 256
        optimizer = torch.optim.AdamW(
            misc.add_weight_decay(model, args.weight_decay),
            lr=args.lr, betas=(0.9, 0.95),
        )
        updates_per_epoch = len(loader) // args.accum_iter
        if misc.is_main_process():
            print(f"Base lr: {args.blr:.2e}")
            print(f"Actual lr: {args.lr:.2e}")
            print(f"Effective batch size: {effective_batch}")

    checkpoint = _checkpoint_path(args)
    args.start_epoch = _load_checkpoint(
        checkpoint, model, optimizer, device, args.evaluate_gen
    )
    if args.train_rng_seed is not None and not args.evaluate_gen:
        seed = args.train_rng_seed + rank
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        if misc.is_main_process():
            print(f"[Random Contract] reset training RNG after model init: seed={seed}")

    if args.evaluate_gen:
        _evaluate_with_fixed_seed(
            model, args, device, rank, args.start_epoch, args.gen_bsz,
            logger, args.start_epoch * updates_per_epoch,
        )
        if logger is not None:
            logger.finish()
        if args.distributed:
            torch.distributed.destroy_process_group()
        return

    wrapped = model
    if args.distributed:
        wrapped = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
    if args.specslot:
        anchor_count = (args.img_size // args.spec_window_stride) ** 2
        slots_per_anchor = args.spec_n_max // anchor_count
        model_label = (
            f"SpecSlot ({args.img_size}px, K{slots_per_anchor}, "
            f"B{args.spec_spatialized_bottleneck_dim})"
        )
    else:
        model_label = args.model
    print(f"Training {model_label} for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler.set_epoch(epoch)
        train_one_epoch(
            wrapped, model, loader, optimizer, device, epoch,
            log_writer=logger, args=args,
        )

        if _should_save(epoch, args.save_last_freq, args.save_completed_epochs) or epoch + 1 == args.epochs:
            misc.save_model(args, model, optimizer, epoch, epoch_name="last")
        if args.save_periodic_freq and (epoch + 1) % args.save_periodic_freq == 0:
            misc.save_model(args, model, optimizer, epoch, epoch_name=str(epoch + 1))

        due = ((epoch + 1) if args.eval_completed_epochs else epoch) % args.eval_freq == 0
        if args.online_eval and (due or epoch + 1 == args.epochs):
            with torch.no_grad():
                evaluate(
                    model, args,
                    epoch + 1 if args.eval_completed_epochs else epoch,
                    batch_size=args.gen_bsz,
                    log_writer=logger,
                    log_step=(epoch + 1) * updates_per_epoch,
                )
        if logger is not None:
            logger.flush()

    if misc.is_main_process():
        elapsed = datetime.timedelta(seconds=int(time.time() - start_time))
        print(f"Training time: {elapsed}")
    if logger is not None:
        logger.finish()
    if args.distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main(get_args_parser().parse_args())
