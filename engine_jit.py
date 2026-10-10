import math
import os
import shutil

import cv2
import numpy as np
import torch
from torch_fidelity.datasets import ImagesPathDataset
from torch_fidelity.metric_fid import fid_statistics_to_metric
from torch_fidelity.metric_isc import isc_features_to_metric
from torch_fidelity.utils import (
    create_feature_extractor,
    get_featuresdict_from_dataset,
    glob_samples_paths,
)

import util.lr_sched as lr_sched
import util.misc as misc
from util.eval_weights import build_eval_state_dict


cv2.setNumThreads(0)


def _calculate_fid_and_is(save_folder, statistics_path, batch_size=64):
    """Compute FID against the packaged statistics and the Inception Score.

    torch-fidelity 0.4.0 requires a second sample set when FID and IS are
    requested together, while the paper uses a precomputed FID reference. A
    single Inception pass keeps the two metrics consistent without requiring
    the unreleased torch-fidelity extension used by the research codebase.
    """
    paths = glob_samples_paths(save_folder, False, "png", verbose=False)
    dataset = ImagesPathDataset(paths)
    extractor = create_feature_extractor(
        "inception-v3-compat", ["2048", "logits_unbiased"], cuda=True,
        verbose=False,
    )
    features = get_featuresdict_from_dataset(
        dataset, extractor, batch_size, cuda=True,
        save_cpu_ram=True, verbose=False,
    )

    generated = features["2048"].numpy().astype(np.float64, copy=False)
    generated_stats = {
        "mu": generated.mean(axis=0),
        "sigma": np.cov(generated, rowvar=False),
    }
    reference = np.load(statistics_path)
    fid = fid_statistics_to_metric(
        generated_stats,
        {
            "mu": reference["mu"].astype(np.float64, copy=False),
            "sigma": reference["sigma"].astype(np.float64, copy=False),
        },
        verbose=False,
    )
    inception = isc_features_to_metric(
        features["logits_unbiased"], splits=10, shuffle=True, rng_seed=2020,
    )
    return fid["frechet_inception_distance"], inception["inception_score_mean"]


def train_one_epoch(model, model_without_ddp, data_loader, optimizer, device,
                    epoch, log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        'lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}')
    )
    header = f'Epoch: [{epoch}]'
    optimizer.zero_grad()

    if log_writer is not None:
        print(f'log_dir: {log_writer.log_dir}')

    for step, (images, labels) in enumerate(
            metric_logger.log_every(data_loader, 20, header)):
        lr_sched.adjust_learning_rate(
            optimizer, step / len(data_loader) + epoch, args
        )
        images = images.to(device, non_blocking=True).float().div_(255).mul_(2).sub_(1)
        labels = labels.to(device, non_blocking=True)
        global_step = epoch * len(data_loader) + step + 1
        if hasattr(model_without_ddp, 'set_training_step'):
            model_without_ddp.set_training_step(global_step - 1)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = model(images, labels)
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise FloatingPointError(
                f'Non-finite loss at epoch {epoch}, step {step}'
            )

        optimizer.zero_grad()
        loss.backward()
        grad_clip = 0.0 if args.grad_clip is None else args.grad_clip
        if grad_clip < 0:
            raise ValueError(f'--grad_clip must be non-negative, got {grad_clip}')
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        model_without_ddp.update_ema()

        metric_logger.update(loss=loss_value, lr=optimizer.param_groups[0]['lr'])
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and step % args.log_freq == 0:
            log_writer.add_scalar(
                'train_loss', loss_value_reduce, global_step
            )
            log_writer.add_scalar(
                'lr', optimizer.param_groups[0]['lr'], global_step
            )

    metric_logger.synchronize_between_processes()
    print(f'Averaged stats: {metric_logger}')
    return {key: meter.global_avg for key, meter in metric_logger.meters.items()}


def evaluate(model_without_ddp, args, epoch, batch_size=64, log_writer=None,
             log_step=None):
    model_without_ddp.eval()
    device = next(model_without_ddp.parameters()).device
    world_size = misc.get_world_size()
    rank = misc.get_rank()
    num_steps = math.ceil(args.num_images / (batch_size * world_size))
    save_folder = os.path.join(
        args.output_dir,
        f'{model_without_ddp.method}-steps{model_without_ddp.steps}-'
        f'cfg{model_without_ddp.cfg_scale}-interval'
        f'{model_without_ddp.cfg_interval[0]}-'
        f'{model_without_ddp.cfg_interval[1]}-weights{args.eval_weights}-'
        f'epoch{epoch}-image{args.num_images}-res{args.img_size}',
    )
    if rank == 0:
        os.makedirs(save_folder, exist_ok=True)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    print(f'[Eval] Generation batch size: {batch_size} (per rank)')
    if rank == 0:
        print(f'Save to: {save_folder}')

    raw_state = build_eval_state_dict(model_without_ddp, 'raw')
    eval_state = build_eval_state_dict(model_without_ddp, args.eval_weights)
    model_without_ddp.load_state_dict(eval_state)
    print(f'[Eval] Weight source: {args.eval_weights}')
    if args.num_images % args.class_num:
        raise ValueError('Number of generated images must be divisible by class count')
    labels_all = np.arange(args.class_num).repeat(args.num_images // args.class_num)
    labels_all = np.pad(
        labels_all,
        (0, num_steps * batch_size * world_size - args.num_images),
        constant_values=0,
    )

    try:
        for step in range(num_steps):
            if rank == 0:
                print(f'Generation step {step + 1}/{num_steps}')
            start = step * batch_size * world_size + rank * batch_size
            labels = torch.as_tensor(
                labels_all[start:start + batch_size],
                dtype=torch.long,
                device=device,
            )
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                samples = model_without_ddp.generate(labels)
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            samples = samples.add(1).div(2).detach().cpu()
            if log_writer is not None and step == 0 and args.wandb_num_vis > 0:
                num_vis = min(args.wandb_num_vis, samples.size(0))
                captions = [
                    f'class_{int(labels[index].item())}'
                    for index in range(num_vis)
                ]
                log_step_value = int(epoch if log_step is None else log_step)
                log_writer.add_images(
                    f'samples/{args.eval_weights}/generated',
                    samples[:num_vis], log_step_value, captions=captions,
                )
            for index, sample in enumerate(samples):
                image_id = start + index
                if image_id >= args.num_images:
                    break
                image = np.round(
                    np.clip(sample.numpy().transpose(1, 2, 0) * 255, 0, 255)
                )
                path = os.path.join(save_folder, f'{image_id:05d}.png')
                if not cv2.imwrite(path, image.astype(np.uint8)[:, :, ::-1]):
                    raise IOError(f'Failed to write generated image: {path}')
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
    finally:
        model_without_ddp.load_state_dict(raw_state)
    print(f'[Eval] Restore raw weights after {args.eval_weights} evaluation')

    if rank == 0:
        if args.img_size not in (256, 512):
            raise ValueError('FID statistics are provided for 256 and 512 pixels')
        statistics = os.path.join(
            os.path.dirname(__file__), 'fid_stats',
            f'jit_in{args.img_size}_stats.npz',
        )
        fid, inception_score = _calculate_fid_and_is(save_folder, statistics)
        print(f'FID: {fid:.4f}, Inception Score: {inception_score:.4f}')
        if log_writer is not None:
            step = int(epoch if log_step is None else log_step)
            log_writer.add_scalar('fid', fid, step)
            log_writer.add_scalar('is', inception_score, step)
        shutil.rmtree(save_folder)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
