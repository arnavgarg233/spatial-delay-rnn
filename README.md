# Spatially-Embedded Delay-Coupled RNNs

**A conduction-time wiring economy, its allocation law, and the velocity inverse.**

Spatially-embedded RNNs (seRNN; Achterberg et al., *Nat. Mach. Intell.* 2023) place neurons in physical space and penalize long connections, and brain-like structure emerges. This repository adds the missing physical ingredient - **finite signal speed**: every recurrent edge carries a transmission delay proportional to physical distance (`τ_ij = round(d_ij / v)`), turning the network into a system of delay differential equations with spatially-structured delays.

| Pillar | Result | Code |
|---|---|---|
| **Economy** | at matched accuracy, weight is placed to minimize delay-weighted travel time `Σ\|W\|·τ`, below a histogram-matched shuffled-delay control | [`scripts/economy/`](scripts/economy/) |
| **Law** | the same allocation law `Saving(s) = B₀·s` holds across RNNs, Kuramoto oscillators, reservoirs, and spiking networks | [`scripts/law/`](scripts/law/) |
| **Inverse** | conduction velocity is recoverable from activity alone - on foreign generators and on real cortex | [`scripts/inverse/`](scripts/inverse/) |

Where seRNN realizes Cajal's **space** law (minimize wire length), this realizes his **time** law (minimize travel time). Each pillar has its own runners and reproducibility path.

## Headline results

**Economy - conduction cost at matched accuracy.** Conduction cost is the `|W|`-weighted mean edge delay `τ̄ = Σ|W|·τ / Σ|W|`; the shuffled-delay control keeps the identical delay histogram but scrambles which edge carries which delay.

| Condition (matched accuracy) | weighted-mean delay τ̄ ↓ |
|---|---|
| no-delay | 8.46 |
| shuffled (histogram-matched null) | 7.62 |
| **distance** | **6.32** |

Distance beats the shuffled null on **12/12 seeds** (paired *t* = **−64.9**). Saving is monotone in delay length (corr = **+0.96**), **+17 % stronger in 3D** than 2D, persists across **N = 48 → 1024**, and holds in a distance-delayed **transformer** block.

**Law - the allocation principle.** `min Σ|W|·τ` is the KKT / least-action / optimal-transport solution of a delayed-LQR problem, giving the linear law `Saving(s) = B₀·s` (saving linear in delay spread). The same law reappears in **Kuramoto oscillators, echo-state reservoirs, and spiking LIF networks** - a law of delay-coupled computation, via the Hardy-Littlewood-Pólya rearrangement.

**Inverse - recovering conduction velocity from activity.** Velocity is recoverable from a network's own activity, non-tautologically - on generators it was never fit to, and on real cortex:

| Generator | velocity recovery vs. null |
|---|---|
| in-silico RNN | **0 % error**; sharp bracketed residual minimum at the planted velocity |
| neural-field PDE *(foreign)* | 18 % error; beats an AC-preserving null, monotone in true speed (24/24 sweeps) |
| spiking network *(foreign)* | 18 % error; null *z* ≈ 10 |
| real cortex - Utah array | v̂ ≈ 0.80 m/s; **2.3×** below an autocorrelation-preserving null |
| real cortex - ECoG, 12 subjects | beats a smoothness-preserving cross-subject null (11/12 donors, *t* = 6.0) |

Positioning: the nearest prior work (Mészáros et al., arXiv:2511.01632, 2025) introduces distance-as-delay but reports only structural metrics - no causal control, no conduction-cost metric, no dose-response, and no velocity inverse. Those are the contributions here.

## Repository layout

```
spatial-delay-rnn/
├── src/sdrnn/            model, delays, geometry, tasks, training, graph metrics
│   ├── model.py          spatially-embedded delay-coupled RNN
│   ├── delays.py         distance→delay, delay buffer, shuffled / uniform controls
│   ├── geometry.py       neuron coordinates + distance matrix
│   ├── tasks.py          MemoryPro, DelayedCopy, DelayedMatch
│   ├── train.py          training loop + regularizers
│   └── graph_metrics.py  modularity, small-worldness, communicability
├── scripts/
│   ├── economy/          conduction-time economy: cost, dose-response, scaling, 3D/2D, transformer
│   ├── law/              the allocation law: optimal-control + Kuramoto / reservoir / spiking
│   ├── inverse/          velocity inverse: neural-field PDE, spiking, intracranial + ECoG
│   ├── controls/         energy-budget Pareto + analysis
│   ├── figures/          figure builders (panels a-d, schematic)
│   └── figures.py        regenerates the publication figures
├── results/              run outputs (.json), grouped economy / law / inverse / controls
├── figures/              publication figures (300 dpi) + assets/ (panels a-d)
├── tests/                unit tests
├── environment.yml       conda environment (PyTorch + MPS / CUDA)
└── LICENSE
```

## Installation

```bash
conda env create -f environment.yml
conda activate spatial-delay-rnn
```

Tested on macOS / Apple-silicon (MPS) and Linux CUDA. The communicability regularizer uses `matrix_exp`, which has no MPS kernel (CPU fallback via `PYTORCH_ENABLE_MPS_FALLBACK=1`); large-N runs (N ≳ 256) use the CUDA path in `scripts/economy/scaling_large_n.py`.

## Reproducing

```bash
# economy - the shuffled-control separation (distance < shuffled < no-delay at matched accuracy)
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/economy/conduction_cost.py --device mps
# economy - dose-response in delay length
python scripts/economy/dose_response.py --device mps
# law - the optimal-control / optimal-transport derivation
python scripts/law/optimal_control.py
# inverse - recover conduction velocity on a foreign neural-field PDE
python scripts/inverse/foreigninv_neural_field.py
# regenerate every publication figure from the saved results
python scripts/figures.py
```

## Citation

```bibtex
@article{spatialdelayrnn2026,
  title   = {A Conduction-Time Wiring Economy in Spatially-Embedded Delay-Coupled RNNs},
  author  = {[Anonymous for review]},
  journal = {Under review},
  year    = {2026}
}
```

Builds on seRNN (Achterberg et al., *Nature Machine Intelligence* 2023, s42256-023-00748-9).

## License

MIT - see [LICENSE](LICENSE).
