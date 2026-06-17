"""Self-contained DDPM trainer/evaluator for the prepared INBreast dataset.

Training uses only healthy mammograms. A synthetic local corruption is used as
conditioning, while the diffusion target remains the original healthy image.
At evaluation time the observed mammogram is the condition. The absolute
reconstruction error is used as the anomaly map.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.optimization import get_scheduler
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.inbreast_dataset import (  # noqa: E402
    get_dataloaders_inbreast,
    get_datasets_inbreast,
)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


# Convert command-line values such as 'True', 'false', '1' or '0' into
# a real Python boolean. argparse does not handle these strings reliably by itself.
def str_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).lower().strip()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


# Define all command-line arguments accepted by the script. Some arguments are
# kept for compatibility with the original Colab notebook even when the YAML file
# already contains the same value. Explicit command-line values have priority.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h_config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "inbreast_debug.yaml",
    )
    parser.add_argument("--modality", default="INBREAST")
    parser.add_argument("--fold", default="0")
    parser.add_argument("--datasets_dir", type=Path, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--center", type=str_to_bool, default=None)
    parser.add_argument("--normal_split", type=float, default=None)
    parser.add_argument("--eval", nargs="?", const=True, default=False, type=str_to_bool)

    # Notebook-compatible runtime controls used by the training loop.
    parser.add_argument("--val_steps", type=int, default=10)
    parser.add_argument("--log_frequency", type=int, default=1)
    return parser.parse_args()


# Read the YAML experiment configuration and convert it into nested dictionaries.
def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


# Build the small configuration object expected by inbreast_dataset.py.
# Values passed from the notebook override values stored in the YAML file.
def dataset_namespace(config: Mapping[str, Any], args: argparse.Namespace):
    dset = config["dataset"]
    loader = config["dataloader"]
    train_cfg = config["training"]
    return argparse.Namespace(
        datasets_dir=str(args.datasets_dir or dset["datasets_dir"]),
        image_size=args.image_size or int(dset["image_size"]),
        center=args.center if args.center is not None else bool(dset["center"]),
        normal_split=(
            args.normal_split
            if args.normal_split is not None
            else float(dset["normal_split"])
        ),
        seed=int(train_cfg["seed"]),
        train_batch_size=int(loader["train_batch_size"]),
        validation_batch_size=int(loader["validation_batch_size"]),
        dataloader_num_workers=int(loader["num_workers"]),
    )


# Fix random generators so that data splits, corruptions and sampled diffusion
# timesteps are reproducible as far as the selected hardware permits.
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Model and EMA
# -----------------------------------------------------------------------------


# Construct the denoising U-Net from the YAML configuration. The model receives
# two grayscale channels: the current noisy sample and the conditioning image.
# It returns one channel containing the predicted diffusion noise.
def create_model(config: Mapping[str, Any]) -> UNet2DModel:
    """Create a U-Net that receives [noisy target, condition] by channels."""
    model_cfg = dict(config["model"])
    for key in ("down_block_types", "up_block_types", "block_out_channels"):
        model_cfg[key] = tuple(model_cfg[key])

    channels = int(config["dataset"]["image_channels"])
    # The input has two channels because the noisy diffusion sample and the
    # conditioning mammogram are concatenated channel-wise. The model predicts
    # one noise channel because the original mammograms are grayscale.
    model_cfg.update(
        sample_size=int(config["dataset"]["image_size"]),
        in_channels=2 * channels,
        out_channels=channels,
    )
    return UNet2DModel(**model_cfg)


# Create the DDPM scheduler. It defines how noise is added during training and
# how one reverse-diffusion step is computed during reconstruction.
def create_scheduler(config: Mapping[str, Any]) -> DDPMScheduler:
    return DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="linear",
        prediction_type=str(config["diffusion"]["prediction_type"]),
    )


# EMA keeps a smoothed copy of model parameters. It changes more slowly than the
# current training model and is commonly more stable for diffusion evaluation.
class EMA:
    """Small exponential moving average helper for model parameters."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    parameter.detach(), alpha=1.0 - self.decay
                )

    def save_state(self, path: Path) -> None:
        torch.save(
            {
                "decay": self.decay,
                "shadow": {k: v.cpu() for k, v in self.shadow.items()},
            },
            path,
        )

    def load_state(self, path: Path, device: torch.device) -> None:
        state = torch.load(path, map_location="cpu")
        self.decay = float(state["decay"])
        self.shadow = {k: v.to(device) for k, v in state["shadow"].items()}

    def copy_to(self, model: torch.nn.Module) -> Dict[str, Tensor]:
        """Copy EMA weights to model and return a backup for later restore."""
        backup = {}
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                backup[name] = parameter.detach().clone()
                parameter.data.copy_(self.shadow[name].data)
        return backup

    @staticmethod
    def restore(model: torch.nn.Module, backup: Mapping[str, Tensor]) -> None:
        for name, parameter in model.named_parameters():
            if name in backup:
                parameter.data.copy_(backup[name].data)


# -----------------------------------------------------------------------------
# Synthetic restoration tasks
# -----------------------------------------------------------------------------


# Select a random point inside the breast foreground. This prevents synthetic
# changes from being placed mainly in the black background of a mammogram.
def foreground_center(image: Tensor, threshold: float) -> Tuple[int, int]:
    # image.mean(0) removes the channel dimension. Pixels above the threshold
    # are treated as breast foreground candidates.
    positions = (image.mean(0) > threshold).nonzero(as_tuple=False)
    height, width = image.shape[-2:]
    if len(positions) == 0:
        return random.randrange(height), random.randrange(width)
    y, x = positions[random.randrange(len(positions))]
    return int(y), int(x)


# Create a soft elliptical region around the selected point. The small average
# pooling operation smooths the border so the corruption is not perfectly sharp.
def ellipse_mask(
    image: Tensor, center_y: int, center_x: int, radius_y: int, radius_x: int
) -> Tensor:
    height, width = image.shape[-2:]
    y = torch.arange(height, device=image.device, dtype=image.dtype).view(-1, 1)
    x = torch.arange(width, device=image.device, dtype=image.dtype).view(1, -1)
    distance = ((y - center_y) / radius_y) ** 2 + ((x - center_x) / radius_x) ** 2
    mask = (distance <= 1).to(image.dtype)[None, None]
    # Smoothing produces a gradual transition at the artificial region border.
    return F.avg_pool2d(mask, 7, stride=1, padding=3)[0, 0]


# Apply either local Gaussian noise or a local brightness change inside the
# elliptical region. This creates an artificial abnormal-looking area.
def local_intensity_or_noise(
    image: Tensor, center_y: int, center_x: int, scale: float, use_noise: bool
) -> Tensor:
    height, width = image.shape[-2:]
    radius_y = random.randint(max(4, height // 24), max(5, height // 8))
    radius_x = random.randint(max(4, width // 24), max(5, width // 8))
    mask = ellipse_mask(image, center_y, center_x, radius_y, radius_x)[None]

    if use_noise:
        change = torch.randn_like(image) * scale
    else:
        sign = -1.0 if random.random() < 0.5 else 1.0
        change = torch.ones_like(image) * sign * scale
    return (image + change * mask).clamp(-1, 1)


# Create a spatial deformation by building a sampling grid and moving pixels
# either away from the selected point (source) or towards it (sink).
def radial_deformation(
    image: Tensor,
    center_y: int,
    center_x: int,
    distance_pixels: float,
    source: bool,
) -> Tensor:
    """Push content away from a point (source) or pull it in (sink)."""
    height, width = image.shape[-2:]
    y = torch.linspace(-1, 1, height, device=image.device, dtype=image.dtype)
    x = torch.linspace(-1, 1, width, device=image.device, dtype=image.dtype)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")

    cy = 2 * center_y / max(height - 1, 1) - 1
    cx = 2 * center_x / max(width - 1, 1) - 1
    dy, dx = grid_y - cy, grid_x - cx
    radius = torch.sqrt(dx.square() + dy.square() + 1e-8)
    influence = torch.exp(-radius.square() / (2 * random.uniform(0.12, 0.30) ** 2))

    amount = 2 * distance_pixels / max(height, width)
    direction = 1.0 if source else -1.0
    shift_x = direction * amount * influence * dx / radius
    shift_y = direction * amount * influence * dy / radius
    grid = torch.stack([grid_x + shift_x, grid_y + shift_y], dim=-1)[None]

    # grid_sample creates the deformed image by reading pixels from the newly
    # calculated coordinates. Border padding avoids empty black holes.
    return F.grid_sample(
        image[None], grid, mode="bilinear", padding_mode="border", align_corners=True
    )[0]


# Create the conditioning batch used during training. The original healthy image
# remains the clean diffusion target, while a corrupted copy is given as context.
def corrupt_healthy_batch(
    images: Tensor, config: Mapping[str, Any], centered: bool
) -> Tensor:
    """Create conditions while keeping the original images as clean targets."""
    cfg = config["restoration_tasks"]
    tasks = [int(task) for task in cfg["train_task_ids"]]
    if bool(cfg.get("use_noise_task", False)):
        tasks.append(1)

    threshold = float(cfg.get("use_threshold", 0.225))
    if centered:
        threshold = threshold * 2 - 1

    result = []
    for image in images:
        if random.random() < float(cfg.get("p_no_aug", 0.0)):
            result.append(image.clone())
            continue

        y, x = foreground_center(image, threshold)
        task = random.choice(tasks)
        if task == 1:
            corrupted = local_intensity_or_noise(
                image, y, x, float(cfg["intensity_task_scale"]), use_noise=True
            )
        elif task in {2, 3}:
            corrupted = radial_deformation(
                image,
                y,
                x,
                random.uniform(float(cfg["min_push_dist"]), float(cfg["max_push_dist"])),
                source=(task == 3),
            )
        elif task == 4:
            corrupted = local_intensity_or_noise(
                image, y, x, float(cfg["intensity_task_scale"]), use_noise=False
            )
        else:
            raise ValueError(f"Unsupported restoration task: {task}")
        result.append(corrupted)

    return torch.stack(result)


# -----------------------------------------------------------------------------
# Diffusion loss and reconstruction
# -----------------------------------------------------------------------------


# Compute the signal-to-noise ratio for each sampled diffusion timestep. It is
# used to reduce the dominance of extremely easy or difficult noise levels.
def compute_snr(scheduler: DDPMScheduler, timesteps: Tensor) -> Tensor:
    alphas = scheduler.alphas_cumprod.to(timesteps.device)[timesteps]
    return alphas / (1 - alphas)


# Perform one DDPM training objective calculation:
# 1. sample Gaussian noise and a random timestep,
# 2. add noise to the clean healthy image,
# 3. concatenate the noisy image with its corrupted condition,
# 4. ask the U-Net to predict the added noise,
# 5. compare predicted and true noise with MSE, optionally SNR-weighted.
def diffusion_loss(
    model: UNet2DModel,
    scheduler: DDPMScheduler,
    clean: Tensor,
    condition: Tensor,
    config: Mapping[str, Any],
) -> Tensor:
    """Train the U-Net to predict the noise added to the clean target."""
    diff_cfg = config["diffusion"]
    # This is the exact random noise that the network will later be asked to
    # predict. Keeping it allows construction of the supervised DDPM target.
    noise = torch.randn_like(clean)

    offset = float(diff_cfg.get("noise_offset", 0.0))
    if offset > 0:
        noise += offset * torch.randn(
            clean.shape[0], clean.shape[1], 1, 1,
            device=clean.device, dtype=clean.dtype,
        )

    input_noise = noise

    timesteps = torch.randint(
        0, scheduler.config.num_train_timesteps, (clean.shape[0],),
        device=clean.device, dtype=torch.long,
    )

    # Forward diffusion: create x_t from the clean target x_0 at the sampled t.
    noisy_clean = scheduler.add_noise(clean, input_noise, timesteps)

    if scheduler.config.prediction_type == "epsilon":
        target = noise
    elif scheduler.config.prediction_type == "v_prediction":
        target = scheduler.get_velocity(clean, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type: {scheduler.config.prediction_type}")

    # Concatenation gives the U-Net both pieces of information at once:
    # channel 0 = noisy target, channel 1 = corrupted conditioning image.
    prediction = model(torch.cat([noisy_clean, condition], dim=1), timesteps).sample
    gamma = config["optimizer"].get("snr_gamma")
    if gamma is None:
        return F.mse_loss(prediction.float(), target.float())

    snr = compute_snr(scheduler, timesteps)
    weights = torch.minimum(snr, torch.full_like(snr, float(gamma))) / snr
    loss = F.mse_loss(prediction.float(), target.float(), reduction="none")
    return (loss.mean((1, 2, 3)) * weights).mean()


# Reconstruct a pseudo-healthy image. Reverse diffusion starts from random noise
# and is conditioned on the observed mammogram at every denoising step. Gradients
# are disabled because this function is used only for validation/evaluation.
@torch.no_grad()
def reconstruct(
    model: UNet2DModel,
    scheduler: DDPMScheduler,
    images: Tensor,
    config: Mapping[str, Any],
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Generate pseudo-healthy images by reverse diffusion conditioned on input."""
    steps = int(config["diffusion"]["validation_timesteps"])
    guidance = float(config["diffusion"].get("validation_guidance", 0.0))
    # Reverse diffusion begins from pure Gaussian noise with the same shape as
    # the observed image.
    sample = torch.randn_like(images, device=device)

    try:
        scheduler.set_timesteps(steps, device=device)
    except TypeError:  # older Diffusers versions
        scheduler.set_timesteps(steps)

    model.eval()
    for timestep in scheduler.timesteps:
        if guidance > 0:
            sample_in = torch.cat([sample, sample])
            condition = torch.cat([torch.zeros_like(images), images])
        else:
            sample_in, condition = sample, images

        sample_in = scheduler.scale_model_input(sample_in, timestep)
        predicted_noise = model(torch.cat([sample_in, condition], 1), timestep).sample

        if guidance > 0:
            uncond, cond = predicted_noise.chunk(2)
            predicted_noise = uncond + guidance * (cond - uncond)

        # One reverse DDPM step transforms x_t into an estimate of x_(t-1).
        sample = scheduler.step(predicted_noise, timestep, sample).prev_sample

    # Regions that the model reconstructs differently from the input receive a
    # high anomaly value. For grayscale data, the channel mean keeps shape Bx1xHxW.
    anomaly_map = (sample - images).abs().mean(1, keepdim=True)
    anomaly_score = anomaly_map.mean((1, 2, 3))
    return sample, anomaly_map, anomaly_score


# -----------------------------------------------------------------------------
# Checkpoints
# -----------------------------------------------------------------------------


CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)$")


# Extract the numerical step from a directory name such as 'checkpoint-10'.
def checkpoint_step(path: Path) -> int:
    match = CHECKPOINT_RE.fullmatch(path.name)
    if not match:
        raise ValueError(f"Invalid checkpoint name: {path.name}")
    return int(match.group(1))


# Return all valid checkpoint directories sorted by their training step.
def checkpoints(run_dir: Path) -> List[Path]:
    if not run_dir.exists():
        return []
    return sorted(
        [p for p in run_dir.iterdir() if p.is_dir() and CHECKPOINT_RE.fullmatch(p.name)],
        key=checkpoint_step,
    )


# Return the most recent checkpoint, or None when training has not saved one.
def latest_checkpoint(run_dir: Path) -> Optional[Path]:
    found = checkpoints(run_dir)
    return found[-1] if found else None


# Save everything required either for evaluation or for continuing training:
# current U-Net weights, EMA weights, optimizer, LR scheduler and AMP scaler.
def save_checkpoint(
    run_dir: Path,
    step: int,
    model: UNet2DModel,
    ema: Optional[EMA],
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    scaler: Optional[torch.cuda.amp.GradScaler],
    config: Mapping[str, Any],
) -> None:
    folder = run_dir / f"checkpoint-{step}"
    folder.mkdir(parents=True, exist_ok=True)
    # Diffusers writes model configuration and weights into the 'unet' folder.
    model.save_pretrained(folder / "unet", safe_serialization=True)

    if ema:
        backup = ema.copy_to(model)
        # Temporarily copy EMA parameters into the model only for serialization.
        model.save_pretrained(folder / "unet_ema", safe_serialization=True)
        EMA.restore(model, backup)
        ema.save_state(folder / "ema_state.pt")

    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler else None,
        },
        folder / "trainer_state.pt",
    )

    limit = int(config["checkpointing"]["checkpoints_total_limit"])
    found = checkpoints(run_dir)
    while limit > 0 and len(found) > limit:
        shutil.rmtree(found.pop(0))
    print(f"Saved checkpoint: {folder}")


# Restore the trainable model and training state from a checkpoint.
def load_for_training(
    folder: Path, device: torch.device, use_ema: bool
) -> Tuple[UNet2DModel, Optional[EMA], Dict[str, Any]]:
    model = UNet2DModel.from_pretrained(folder / "unet").to(device)
    ema = EMA(model) if use_ema else None
    if ema and (folder / "ema_state.pt").exists():
        ema.load_state(folder / "ema_state.pt", device)
    state_path = folder / "trainer_state.pt"
    state = torch.load(state_path, map_location="cpu") if state_path.exists() else {}
    return model, ema, state


# Prefer EMA weights for evaluation because they are smoother. Fall back to the
# ordinary U-Net weights if the EMA directory is unavailable.
def load_for_evaluation(folder: Path, device: torch.device) -> UNet2DModel:
    model_dir = folder / "unet_ema" if (folder / "unet_ema").exists() else folder / "unet"
    return UNet2DModel.from_pretrained(model_dir).to(device).eval()


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------


# Evaluate the same denoising objective on the healthy validation subset without
# updating parameters. The model is returned to training mode afterwards.
def validation_loss(
    model, scheduler, loader, config, data_cfg, device, use_amp
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for clean in loader:
            clean = clean.to(device, non_blocking=True)
            # The clean image is the desired output; its synthetic corruption is
            # the condition that teaches restoration of healthy anatomy.
            condition = corrupt_healthy_batch(clean, config, data_cfg.center)
            losses.append(float(loss.cpu()))
    model.train()
    return float(np.mean(losses)) if losses else math.nan


# Main optimization loop. It loads data, creates or restores the model, computes
# the DDPM loss, applies gradient clipping, updates EMA weights, validates and
# periodically saves checkpoints.
def train(config, args, data_cfg, run_dir: Path, device: torch.device) -> None:
    train_loader, val_loader = get_dataloaders_inbreast(data_cfg, train=True)
    train_cfg, opt_cfg = config["training"], config["optimizer"]
    resume = latest_checkpoint(run_dir) if train_cfg.get("resume_from_checkpoint") == "latest" else None

    if resume:
        print(f"Resuming from: {resume}")
        model, ema, state = load_for_training(
            resume, device, bool(train_cfg.get("use_ema", False))
        )
        global_step = int(state.get("step", checkpoint_step(resume)))
    else:
        model = create_model(config).to(device)
        ema = EMA(model) if bool(train_cfg.get("use_ema", False)) else None
        state, global_step = {}, 0

    if bool(train_cfg.get("gradient_checkpointing", False)):
        model.enable_gradient_checkpointing()
    if bool(train_cfg.get("allow_tf32", False)) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(opt_cfg["learning_rate"]),
        betas=(float(opt_cfg["adam_beta1"]), float(opt_cfg["adam_beta2"])),
        weight_decay=float(opt_cfg["adam_weight_decay"]),
        eps=float(opt_cfg["adam_epsilon"]),
    )
    max_steps = int(train_cfg["max_train_steps"])
    lr_scheduler = get_scheduler(
        str(opt_cfg["lr_scheduler"]), optimizer=optimizer,
        num_warmup_steps=int(opt_cfg["lr_warmup_steps"]),
        num_training_steps=max_steps,
    )

    scheduler = create_scheduler(config)
    accumulation = int(train_cfg["gradient_accumulation_steps"])
    checkpoint_every = int(config["checkpointing"]["checkpointing_steps"])
    data_iterator = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(total=max_steps, initial=global_step, desc="Training")

    while global_step < max_steps:
        step_loss = 0.0
        for _ in range(accumulation):
            try:
                clean = next(data_iterator)
            except StopIteration:
                data_iterator = iter(train_loader)
                clean = next(data_iterator)

            clean = clean.to(device, non_blocking=True)
            condition = corrupt_healthy_batch(clean, config, data_cfg.center)
            with autocast_context(device, use_amp):
                loss = diffusion_loss(model, scheduler, clean, condition, config)
                backprop_loss = loss / accumulation

            if scaler:
                scaler.scale(backprop_loss).backward()
            else:
                backprop_loss.backward()
            step_loss += float(loss.detach().cpu()) / accumulation

        if scaler:
            scaler.unscale_(optimizer)
        # Gradient clipping limits unstable parameter updates.
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(opt_cfg["max_grad_norm"]))

        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        lr_scheduler.step()
        global_step += 1
        if ema:
            # Update the smoothed evaluation copy after the optimizer step.
            ema.update(model)

        progress.update(1)
        progress.set_postfix(loss=f"{step_loss:.5f}")
        if global_step % max(1, args.log_frequency) == 0:
            print(f"step={global_step} loss={step_loss:.6f}")
        if global_step % max(1, args.val_steps) == 0:
            value = validation_loss(model, scheduler, val_loader, config, data_cfg, device, use_amp)
            print(f"validation_loss={value:.6f}")
        if global_step % checkpoint_every == 0 or global_step == max_steps:
            save_checkpoint(run_dir, global_step, model, ema, optimizer, lr_scheduler, scaler, config)

    progress.close()


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------


# Select samples across the whole class list rather than simply taking the first
# few files. This gives a more distributed mini-evaluation subset.
def pick_evenly(indices: Sequence[int], count: int) -> List[int]:
    if len(indices) <= count:
        return list(indices)
    positions = np.linspace(0, len(indices) - 1, count, dtype=int)
    return [int(indices[position]) for position in positions]


# Build the mini-evaluation subset. In balanced mode, approximately half of the
# selected samples are anomalous and half are healthy.
def evaluation_indices(dataset: Dataset, size: int, balanced: bool) -> List[int]:
    if not balanced:
        return list(range(min(size, len(dataset))))
    labels = list(getattr(dataset, "labels"))
    anomalous = [i for i, label in enumerate(labels) if int(label) == 1]
    healthy = [i for i, label in enumerate(labels) if int(label) == 0]
    anomalous_count = min(len(anomalous), size // 2)
    healthy_count = min(len(healthy), size - anomalous_count)
    return pick_evenly(anomalous, anomalous_count) + pick_evenly(healthy, healthy_count)

# Compute sample-level metrics from one score per image and pixel-level metrics
# from the full anomaly maps and segmentation masks.
def metrics(labels: Tensor, scores: Tensor, masks: Tensor, maps: Tensor, config):
    labels_np, scores_np = labels.numpy(), scores.numpy()
    result = {}
    if config["evaluation"].get("compute_sample_metrics", True):
        result["sample_average_precision"] = float(average_precision_score(labels_np, scores_np))
        result["sample_auroc"] = safe_auc(labels_np, scores_np)
    if config["evaluation"].get("compute_pixel_metrics", True):
        masks_np, maps_np = masks.numpy().reshape(-1), maps.numpy().reshape(-1)
        result["pixel_average_precision"] = float(average_precision_score(masks_np, maps_np))
        result["pixel_auroc"] = safe_auc(masks_np, maps_np)
        result["optimal_dice"], result["optimal_dice_threshold"] = best_dice(masks_np, maps_np)
    return result


# Convert a tensor into a 2-D NumPy image suitable for Matplotlib. 
# When the dataset used [-1, 1], reverse that normalization back to [0, 1].
def display_image(image: Tensor, centered: bool) -> np.ndarray:
    image = image.float().cpu()
    if centered:
        image = image / 2 + 0.5
    return image[0].clamp(0, 1).numpy()

# Save one four-panel figure per sample: input, pseudo-healthy reconstruction,
# predicted anomaly map and ground-truth lesion mask.
def save_visualizations(folder, inputs, reconstructions, maps, masks, labels, paths, centered):
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(len(inputs)):
        figure, axes = plt.subplots(1, 4, figsize=(12, 3))
        axes[0].imshow(display_image(inputs[i], centered), cmap="gray")
        axes[1].imshow(display_image(reconstructions[i], centered), cmap="gray")
        axes[2].imshow(maps[i, 0].numpy(), cmap="hot")
        axes[3].imshow(masks[i, 0].numpy(), cmap="gray")
        for axis, title in zip(axes, ["Input", "Reconstruction", "Anomaly map", "Ground truth"]):
            axis.set_title(title)
            axis.axis("off")
        figure.suptitle(f"{Path(paths[i]).stem} | label={int(labels[i])}")
        figure.tight_layout()
        figure.savefig(folder / f"sample_{i:02d}_label_{int(labels[i])}.png", dpi=150)
        plt.close(figure)

# Run the complete evaluation pipeline: load the newest checkpoint, choose a mini
# subset, reconstruct each image, create anomaly maps/scores, compute metrics and
# save tensors, CSV data, JSON metrics and visualizations.
def evaluate(config, data_cfg, run_dir: Path, device: torch.device, modality: str) -> None:
    checkpoint = latest_checkpoint(run_dir)
    if checkpoint is None:
        raise FileNotFoundError(f"No checkpoint found in: {run_dir}")

    model = load_for_evaluation(checkpoint, device)
    scheduler = create_scheduler(config)
    dataset = get_datasets_inbreast(data_cfg, train=False)
    eval_cfg = config["evaluation"]
    indices = evaluation_indices(
        dataset, int(eval_cfg["mini_eval_size"]), bool(eval_cfg["balanced_mini_eval"])
    )
    # Subset keeps original dataset behavior while exposing only chosen samples.
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset, batch_size=data_cfg.validation_batch_size, shuffle=False,
        num_workers=data_cfg.dataloader_num_workers, pin_memory=(device.type == "cuda"),
    )

    selected_labels = [int(dataset.labels[i]) for i in indices]
    print(
        f"Evaluation subset: {len(indices)} samples "
        f"({sum(selected_labels)} anomalous, {len(indices)-sum(selected_labels)} healthy)"
    )

    inputs, reconstructions, maps, scores, masks, labels, paths = [], [], [], [], [], [], []
    for images, batch_masks, batch_labels, batch_paths in tqdm(loader, desc="Evaluation"):
        images = images.to(device, non_blocking=True)
        # Each input is converted into a pseudo-healthy reconstruction. The pixel
        # difference gives a map, and its mean gives one score for the full image.
        restored, anomaly_maps, anomaly_scores = reconstruct(model, scheduler, images, config, device)
        inputs.append(images.cpu())
        reconstructions.append(restored.cpu())
        maps.append(anomaly_maps.cpu())
        scores.append(anomaly_scores.cpu())
        masks.append(batch_masks.cpu())
        labels.append(batch_labels.cpu())
        paths.extend(list(batch_paths))

    inputs, reconstructions = torch.cat(inputs), torch.cat(reconstructions)
    maps, scores = torch.cat(maps), torch.cat(scores)
    masks, labels = torch.cat(masks), torch.cat(labels).long()

    step = checkpoint_step(checkpoint)
    output = run_dir / f"eval_{modality}_checkpoint-{step}"
    output.mkdir(parents=True, exist_ok=True)
    torch.save(maps, output / "anomaly_maps.pt")
    torch.save(scores, output / "anomaly_scores.pt")
    torch.save(labels, output / "labels.pt")
    torch.save(masks, output / "masks.pt")
    torch.save(reconstructions, output / "reconstructions.pt")

    pd.DataFrame(
        {"path": paths, "label": labels.tolist(), "anomaly_score": scores.tolist()}
    ).to_csv(output / "balanced_mini_eval_scores.csv", index=False)

    result = metrics(labels, scores, masks, maps, config)
    with (output / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    if config["results"].get("save_visualizations", True):
        save_visualizations(
            output / "visualizations", inputs, reconstructions, maps,
            masks, labels, paths, data_cfg.center,
        )

    for name, value in result.items():
        print(f"{name}: {value:.6f}")
    print(f"Saved results to: {output}")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


# Entry point: read configuration, prepare deterministic execution, choose the
# available device and dispatch either training or evaluation.
def main() -> None:
    args = parse_args()
    if args.modality.upper() != "INBREAST":
        raise ValueError("This trainer currently supports only INBREAST.")

    config = load_config(args.h_config)
    data_cfg = dataset_namespace(config, args)
    seed_everything(data_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(config["checkpointing"]["output_dir"]) / f"INBREAST_fold{args.fold}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Config: {args.h_config}")
    print(f"Dataset: {data_cfg.datasets_dir}/INBreast")
    print(f"Run directory: {run_dir}")
    print(f"Device: {device}")

    # The same script supports both workflows; --eval switches off optimization
    # and runs reconstruction-based anomaly evaluation instead.
    if args.eval:
        evaluate(config, data_cfg, run_dir, device, args.modality.upper())
    else:
        train(config, args, data_cfg, run_dir, device)


if __name__ == "__main__":
    main()
