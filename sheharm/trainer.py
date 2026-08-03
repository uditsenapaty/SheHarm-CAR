"""Joint training loop.

Paper Table `tab:hyperparameters`: AdamW, 1e-5 for pretrained encoders and 5e-4 for newly
initialised layers, weight decay 0.01, batch 32, up to 15 epochs, mixed precision, early
stopping with patience 5. The retained checkpoint is the one with the highest mean
validation F1 across target identification, harmfulness, harm category, and rationale
generation.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from .evaluate import evaluate


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def enable_determinism(seed: int) -> None:
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class WeightEMA:
    """Exponential moving average of the trainable weights, evaluated instead of the raw ones.

    The paper does not specify an averaging scheme; this only changes *which* weights are
    scored, never the objective, the schedule, or the checkpoint-selection rule.

    The decay must fit the run. This model trains for ~1.3k steps, so a decay of 0.999 -
    an effective window of ~1000 steps - leaves 27% of the averaged weights sitting on the
    random initialisation and makes every metric worse. The decay is additionally warmed up
    as min(decay, (1+t)/(10+t)) so the first evaluations are not averages over noise.
    """

    # Pretrained bulk: fine-tuned at 1e-5, so it barely moves and averaging it buys almost
    # nothing while costing a full fp32 copy on the accelerator and another on the host.
    # Averaging only the newly-initialised reasoning layers keeps the benefit at ~10% of the
    # memory. On a 15 GB host the full version gets the process SIGKILLed mid-epoch.
    PRETRAINED_PREFIXES = ("encoder.vision.", "encoder.text.", "rationale_decoder.decoder.",
                           "rationale_decoder.lm_head.")

    def __init__(self, model: nn.Module, decay: float = 0.99):
        self.decay = decay
        self.step = 0
        self.shadow = {
            name: parameter.detach().clone().float()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and not name.startswith(self.PRETRAINED_PREFIXES)
        }
        self.backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.step += 1
        decay = min(self.decay, (1.0 + self.step) / (10.0 + self.step))
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(decay).add_(parameter.detach().float(), alpha=1.0 - decay)

    @torch.no_grad()
    def apply_to(self, model: nn.Module) -> None:
        # The backup is parked on CPU: holding a second full copy on the accelerator would
        # add ~800 MB for this model and can push a 16 GB card into OOM.
        self.backup = {}
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = parameter.detach().to("cpu", copy=True)
                parameter.copy_(self.shadow[name].to(parameter.dtype))

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if name in self.backup:
                parameter.copy_(self.backup[name].to(parameter.device))
        self.backup = {}


def balanced_sampler(dataset, power: float = 1.0, seed: int = 42, include_target: bool = True):
    """Sample inversely to joint label frequency.

    Every headline metric is a macro average, so a rare class counts as much as `Non-Harm`.
    The paper fixes the losses but says nothing about the sampler, so mini-batch composition
    is a free lever: the objective stays exactly -sum log p.

    The target is folded into the key because it is the most skewed of the three - one
    concept covers over half the corpus - and Joint-F1 is a conjunction, so it is capped by
    whichever component is weakest. Target frequency is damped by a square root so that
    balancing it does not overwhelm the harmfulness and category balance.
    """
    from collections import Counter

    from torch.utils.data import WeightedRandomSampler

    frame = dataset.frame
    keys = (frame["harmfulness"].astype(str) + "|" + frame["harm_category"].astype(str)).tolist()
    counts = Counter(keys)
    weights = [(1.0 / counts[key]) ** power for key in keys]
    if include_target and "target_concept" in frame.columns:
        target_keys = frame["target_concept"].astype(str).tolist()
        target_counts = Counter(target_keys)
        weights = [w * (1.0 / target_counts[k]) ** (power * 0.5) for w, k in zip(weights, target_keys)]
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights, num_samples=len(frame), replacement=True, generator=generator)


@dataclass
class TrainConfig:
    epochs: int = 15
    patience: int = 5
    batch_size: int = 32
    encoder_lr: float = 1e-5
    new_lr: float = 5e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    num_workers: int = 2
    amp: bool = True
    seed: int = 42
    output_dir: str = "runs/default"
    selection_metrics: list[str] = field(default_factory=lambda: ["target_f1", "harm_f1", "category_f1", "bertscore"])
    eval_bertscore_during_training: bool = False
    log_every: int = 50
    # Paper-silent knobs (see configs/paper_literal.yaml to disable all of them).
    balanced_sampling: bool = True
    balance_power: float = 1.0
    balance_include_target: bool = True
    ema_decay: float = 0.99    # 0 disables EMA; must fit the run length
    lr_schedule: str = "cosine"   # paper-silent: it fixes the optimizer, not the decay shape


def build_optimizer(model: nn.Module, config: TrainConfig):
    encoder_params, new_params = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("encoder.vision.") or name.startswith("encoder.text."):
            encoder_params.append(parameter)
        else:
            new_params.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": config.encoder_lr},
            {"params": new_params, "lr": config.new_lr},
        ],
        weight_decay=config.weight_decay,
    )


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: (value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value)
        for key, value in batch.items()
    }


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, config: TrainConfig, ema=None) -> dict:
    model.train()
    running, component_totals = [], {}
    for step, batch in enumerate(loader):
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=config.amp and device.type == "cuda"):
            output = model(**batch)
            loss = output.total_loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if ema is not None:
            ema.update(model)

        running.append(loss.item())
        for name, value in output.losses.items():
            component_totals.setdefault(name, []).append(float(value.detach()))
        if config.log_every and step % config.log_every == 0:
            print(f"    step {step:4d}/{len(loader)} loss={loss.item():.4f}", flush=True)

    summary = {"loss": float(np.mean(running)) if running else 0.0}
    summary.update({f"loss_{name}": float(np.mean(values)) for name, values in component_totals.items()})
    return summary


def fit(model, train_dataset, dev_dataset, tokenizer, config: TrainConfig, device: torch.device) -> dict:
    enable_determinism(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    sampler = balanced_sampler(
        train_dataset, config.balance_power, config.seed, config.balance_include_target
    ) if config.balanced_sampling else None
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=(sampler is None), sampler=sampler, generator=generator,
        num_workers=config.num_workers, pin_memory=(device.type == "cuda"), drop_last=False,
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=(device.type == "cuda"),
    )

    optimizer = build_optimizer(model, config)
    total_steps = max(1, len(train_loader) * config.epochs)
    warmup_steps = int(total_steps * config.warmup_ratio)
    make_schedule = (
        get_cosine_schedule_with_warmup if config.lr_schedule == "cosine" else get_linear_schedule_with_warmup
    )
    scheduler = make_schedule(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_model.pt"
    best_score, bad_epochs, history = -1.0, 0, []

    ema = WeightEMA(model, config.ema_decay) if config.ema_decay else None

    for epoch in range(1, config.epochs + 1):
        train_summary = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, config, ema)
        if ema is not None:
            ema.apply_to(model)
        dev_metrics = evaluate(
            model, dev_loader, tokenizer, device,
            compute_bertscore=config.eval_bertscore_during_training,
            compute_counterfactuals=False,
        )
        tracked = [dev_metrics[name] for name in config.selection_metrics
                   if name in dev_metrics and not np.isnan(dev_metrics[name])]
        mean_f1 = float(np.mean(tracked)) if tracked else 0.0

        record = {"epoch": epoch, **train_summary, **{f"dev_{k}": v for k, v in dev_metrics.items()}, "dev_mean_f1": mean_f1}
        history.append(record)
        print(
            f"epoch {epoch:02d} | train_loss={train_summary['loss']:.4f} | "
            f"tgt={dev_metrics.get('target_f1', float('nan')):.2f} "
            f"harm={dev_metrics.get('harm_f1', float('nan')):.2f} "
            f"cat={dev_metrics.get('category_f1', float('nan')):.2f} | mean={mean_f1:.2f}",
            flush=True,
        )

        if mean_f1 > best_score:
            best_score, bad_epochs = mean_f1, 0
            # Saved while the EMA weights are applied, so the checkpoint is what was scored.
            torch.save(model.state_dict(), best_path)
        else:
            bad_epochs += 1
        if ema is not None:
            ema.restore(model)
        if bad_epochs >= config.patience:
            print(f"early stopping after epoch {epoch}", flush=True)
            break

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "train_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    return {"best_dev_mean_f1": best_score, "history": history, "checkpoint": str(best_path)}
