"""Training tasks.

* :class:`MemoryProTask` - delayed cued-response working memory, the seRNN-style
  reproduction task. Structural metrics are measured on networks trained here.
* :class:`DelayedParityTask` / :class:`DelayedCopyTask` - tasks where temporal
  lag is informative, used to test whether delays help function.

Each task exposes ``input_size``/``output_size``, ``generate(batch, generator)``
-> (inputs, targets, mask), and ``loss``/``accuracy``. ``mask`` (batch, T) marks
the scored timesteps (typically the response window).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class MemoryProTask:
    """Delayed cued-response working-memory task.

    Trial structure (per timestep, one-hot cue channels + 1 go channel):

        [ fixation ][   cue   ][   delay   ][ response ]
          cue=off     cue=on      cue=off     go=on, cue=off

    The network must output the class of the cue that was shown, but only during
    the response window (after the delay), forcing it to *hold* the cue.
    """

    n_choices: int = 2
    cue_steps: int = 2
    delay_steps: int = 5
    response_steps: int = 2
    fixation_steps: int = 1
    noise: float = 0.0      # std of Gaussian input noise; >0 makes memory non-trivial

    @property
    def input_size(self) -> int:
        return self.n_choices + 1  # cue channels + go channel

    @property
    def output_size(self) -> int:
        return self.n_choices

    @property
    def seq_len(self) -> int:
        return self.fixation_steps + self.cue_steps + self.delay_steps + self.response_steps

    def generate(
        self, batch: int, generator: Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        T = self.seq_len
        go_ch = self.n_choices
        inputs = torch.zeros(batch, T, self.input_size)
        targets = torch.zeros(batch, T, dtype=torch.long)
        mask = torch.zeros(batch, T)

        cls = torch.randint(0, self.n_choices, (batch,), generator=generator)

        cue_start = self.fixation_steps
        cue_end = cue_start + self.cue_steps
        resp_start = cue_end + self.delay_steps
        resp_end = resp_start + self.response_steps

        rows = torch.arange(batch)
        # Cue: light up the class channel during the cue window.
        inputs[rows[:, None], torch.arange(cue_start, cue_end)[None, :], cls[:, None]] = 1.0
        # Go signal during response window.
        inputs[:, resp_start:resp_end, go_ch] = 1.0
        # Target = the cued class during the response window; mask selects it.
        targets[:, resp_start:resp_end] = cls[:, None]
        mask[:, resp_start:resp_end] = 1.0

        if self.noise > 0:
            # Noise on every channel/timestep forces the net to integrate the
            # brief cue robustly and hold it through the delay rather than
            # latching a single clean input - this is the pressure that makes
            # structured recurrent solutions (and emergent topology) necessary.
            inputs = inputs + self.noise * torch.randn(
                inputs.shape, generator=generator
            )
        return inputs, targets, mask

    def loss(self, outputs, targets, mask) -> torch.Tensor:
        return _masked_ce(outputs, targets, mask)

    def accuracy(self, outputs, targets, mask) -> float:
        return _masked_accuracy(outputs, targets, mask)


@dataclass
class DelayedParityTask:
    """Predict the parity of the input stream up to ``lag`` steps ago.

    A pulse train arrives on one channel; the target at time ``t`` is the parity
    (XOR/sum mod 2) of the pulses seen up to ``t - lag``. The fixed temporal
    offset is exactly the kind of structure where physically delaying signals
    might be functionally useful - the delay test task.
    """

    lag: int = 3
    seq_len: int = 16
    p_pulse: float = 0.5

    @property
    def input_size(self) -> int:
        return 1

    @property
    def output_size(self) -> int:
        return 2  # parity: even / odd

    def generate(
        self, batch: int, generator: Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        T = self.seq_len
        pulses = (torch.rand(batch, T, generator=generator) < self.p_pulse).float()
        inputs = pulses.unsqueeze(-1)  # (batch, T, 1)

        cumulative_parity = torch.cumsum(pulses, dim=1).long() % 2  # parity up to t
        targets = torch.zeros(batch, T, dtype=torch.long)
        mask = torch.zeros(batch, T)
        if self.lag < T:
            # Target at t reflects parity as of (t - lag); score once warmed up.
            targets[:, self.lag :] = cumulative_parity[:, : T - self.lag]
            mask[:, self.lag :] = 1.0
        return inputs, targets, mask

    def loss(self, outputs, targets, mask) -> torch.Tensor:
        return _masked_ce(outputs, targets, mask)

    def accuracy(self, outputs, targets, mask) -> float:
        return _masked_accuracy(outputs, targets, mask)


@dataclass
class DelayedCopyTask:
    """Output the input symbol from ``lag`` steps ago - a *learnable* lag task.

    Each timestep presents a one-hot symbol (``n_symbols`` classes); the target
    at time t is the symbol presented at t-lag. The network holds each symbol for
    ``lag`` steps and releases it on time. Unlike delayed-parity (XOR-hard, sat
    at chance), copy is learnable, so a no-delay-vs-delay accuracy comparison is
    possible. A delay line of length ``lag`` implements exactly this task, so it
    is the cleanest probe of whether transmission delays help function.
    """

    n_symbols: int = 4
    lag: int = 4
    seq_len: int = 20
    noise: float = 0.0

    @property
    def input_size(self) -> int:
        return self.n_symbols

    @property
    def output_size(self) -> int:
        return self.n_symbols

    def generate(
        self, batch: int, generator: Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        T = self.seq_len
        syms = torch.randint(0, self.n_symbols, (batch, T), generator=generator)
        inputs = F.one_hot(syms, self.n_symbols).float()
        if self.noise > 0:
            inputs = inputs + self.noise * torch.randn(inputs.shape, generator=generator)
        targets = torch.zeros(batch, T, dtype=torch.long)
        mask = torch.zeros(batch, T)
        if self.lag < T:
            targets[:, self.lag :] = syms[:, : T - self.lag]  # symbol from lag steps ago
            mask[:, self.lag :] = 1.0                          # score once warmed up
        return inputs, targets, mask

    def loss(self, outputs, targets, mask) -> torch.Tensor:
        return _masked_ce(outputs, targets, mask)

    def accuracy(self, outputs, targets, mask) -> float:
        return _masked_accuracy(outputs, targets, mask)


# -- shared masked objectives ---------------------------------------------
def _masked_ce(outputs: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Cross-entropy averaged over masked (scored) timesteps only."""
    b, T, c = outputs.shape
    logits = outputs.reshape(b * T, c)
    tgt = targets.reshape(b * T)
    per_step = F.cross_entropy(logits, tgt, reduction="none").reshape(b, T)
    denom = mask.sum().clamp_min(1.0)
    return (per_step * mask).sum() / denom


@torch.no_grad()
def _masked_accuracy(outputs: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> float:
    pred = outputs.argmax(dim=-1)
    correct = ((pred == targets).float() * mask).sum()
    denom = mask.sum().clamp_min(1.0)
    return (correct / denom).item()
