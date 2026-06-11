"""Training loop. One ``train`` drives every variant; only the
:class:`~sdrnn.model.SDRNNConfig` changes. Loss is
``task_loss + spatial_regularization``.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import torch

from sdrnn.model import SDRNN, SDRNNConfig


@dataclass
class TrainConfig:
    steps: int = 3000
    batch_size: int = 128
    lr: float = 1e-3
    eval_every: int = 200
    eval_batch: int = 512
    grad_clip: float = 1.0
    device: str = "auto"          # auto | cpu | mps | cuda
    seed: int = 0
    log: bool = True


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class TrainResult:
    config: dict
    train_config: dict
    history: List[dict] = field(default_factory=list)
    final_accuracy: float = 0.0
    final_loss: float = 0.0
    wall_seconds: float = 0.0


def train(
    model_config: SDRNNConfig,
    task,
    train_config: Optional[TrainConfig] = None,
) -> tuple[SDRNN, TrainResult]:
    """Train one network on one task. Returns the model and a result record.

    ``task`` is any object from :mod:`sdrnn.tasks` (duck-typed:
    ``generate``/``loss``/``accuracy`` and ``input_size``/``output_size``).
    """
    tc = train_config or TrainConfig()
    device = resolve_device(tc.device)

    # Keep model dims in sync with the task to avoid silent shape mismatches.
    model_config.input_size = task.input_size
    model_config.output_size = task.output_size

    torch.manual_seed(tc.seed)
    data_gen = torch.Generator().manual_seed(tc.seed + 1)

    model = SDRNN(model_config).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tc.lr)

    result = TrainResult(
        config=_config_dict(model_config),
        train_config=asdict(tc),
    )
    start = time.time()

    for step in range(1, tc.steps + 1):
        model.train()
        inputs, targets, mask = task.generate(tc.batch_size, generator=data_gen)
        inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)

        outputs = model(inputs)
        task_loss = task.loss(outputs, targets, mask)
        reg = model.spatial_regularization()
        loss = task_loss + reg

        opt.zero_grad()
        loss.backward()
        if tc.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        opt.step()

        if step % tc.eval_every == 0 or step == tc.steps:
            acc, eval_loss = evaluate(model, task, tc, device)
            rec = {
                "step": step,
                "task_loss": float(task_loss.item()),
                "reg": float(reg.item()),
                "eval_loss": eval_loss,
                "eval_acc": acc,
            }
            result.history.append(rec)
            if tc.log:
                print(
                    f"[{model_config.describe()}] step {step:>5} "
                    f"task_loss {task_loss.item():.4f} reg {reg.item():.4f} "
                    f"eval_acc {acc:.3f}",
                    flush=True,
                )

    acc, eval_loss = evaluate(model, task, tc, device)
    result.final_accuracy = acc
    result.final_loss = eval_loss
    result.wall_seconds = time.time() - start
    return model, result


@torch.no_grad()
def evaluate(model: SDRNN, task, tc: TrainConfig, device) -> tuple[float, float]:
    model.eval()
    gen = torch.Generator().manual_seed(12345)  # fixed eval set for comparability
    inputs, targets, mask = task.generate(tc.eval_batch, generator=gen)
    inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
    outputs = model(inputs)
    loss = task.loss(outputs, targets, mask)
    acc = task.accuracy(outputs, targets, mask)
    return float(acc), float(loss.item())


def _config_dict(cfg: SDRNNConfig) -> dict:
    """asdict, minus the non-serializable init_coords tensor."""
    d = asdict(cfg)
    d.pop("init_coords", None)
    return d
