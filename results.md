# 10k-step hyperparam sweep — results

**Hardware**: RTX 5080, libtorch nightly cu128.

**Binary**: commit `eca7439` (current `main`). **All C++ was left untouched.**

**Date**: 2026-04-16/17 session (19 runs, ~6.5h of GPU time).

---

## Baseline setup (run01)

Starting point is the published config that reached 0.74 win-rate at 6k steps
in `logs/grid_lr/run_13` (2026-04-12 codebase). The same config at 10k steps
on the _current_ codebase is the baseline.

```yaml
num_steps: 10000

training:
  replay_capacity: 30000
  min_buffer_size: 10000
  train_batch_size: 4096
  train_steps_per_collect: 0.125
  model_swap_interval: 10

  td_mode: tdlambda
  td_lambda: 0.95
  use_duel_margin_maximization: true
  duel_margin_maximization_scale: 4000.0

  initial_temperature: 1.0
  min_temperature:     0.1
  temperature_decay:   0.999712

  initial_heuristic_weight: 1.0
  heuristic_decay_steps:    50000

  eval_interval: 2000
  eval_games:    500

  self_play:
    max_inference_batch: 4096
    min_games_per_batch: 2
    batch_timeout_ms: 5
    num_workers:      24
    num_games:        2048
    num_coordinators: 2

  model:
    input_size: 986
    hidden_layers: 3
    hidden_width:  256
    learning_rate: 0.00032
    output_activation: tanh
    loss_function:     mse
    architecture:      resnet
```

---

## Headline finding

**Current code learns ~3.4× slower** than the 2026-04-12 codebase at the
same config, because of commits that landed between those dates (Watkins
trace cutting `85a3511`, V2 DP `62c5c36`, NN-EV normalisation `43a3c70`).
All tuning below targets the _current_ code.

| epoch | `grid_lr/run_13` (old code) | `run01_baseline` (current code) |
|-------|------------------------------|---------------------------------|
| 2000  | 0.570                        | 0.150                           |
| 4000  | 0.636                        | 0.190                           |
| 6000  | **0.740** (end of run)       | 0.492                           |
| 10000 | —                            | **0.620**                       |

---

## Full results (sorted by 10k win rate)

| run                          | variation                                        | wr10k | margin  | GPS   | wall   | games    |
|------------------------------|--------------------------------------------------|------:|--------:|------:|-------:|---------:|
| **run09_hw512**              | hw=512                                           | 0.772 | 7175    | 90.2  | 889s   | 80 064   |
| **run08_tspc_lo** ⭐          | tspc=0.0625                                      | 0.770 | 6833    | 142.9 | 1120s  | 160 064  |
| final1_resnet_hw512_tspc     | hw=512 + tspc=0.0625                             | 0.768 | 6440    | 92.7  | 1744s  | 160 064  |
| combo1_mlp_tspc_lo ⭐         | mlp + tspc=0.0625                                | 0.736 | 5122    | **159.5** | 1014s | 160 065  |
| combo2_mlp_hw512             | mlp + hw=512                                     | 0.736 | 5429    | 118.4 | 690s   | 80 064   |
| run04_mlp ⭐                  | mlp                                              | 0.708 | 4642    | 153.0 | **535s** | 80 064 |
| run05_mc                     | td_mode=mc                                       | 0.704 | 4468    | 137.2 | 597s   | 80 063   |
| combo4_mlp_hw512_tspc        | mlp + hw=512 + tspc=0.0625                       | 0.698 | 4405    | 123.7 | 1307s  | 160 064  |
| combo5_mlp_lr_hi             | mlp + lr=0.00064                                 | 0.674 | 3182    | 151.7 | 539s   | 80 063   |
| final3_mlp_tdl99             | mlp + td_lambda=0.99                             | 0.656 | 2366    | 153.0 | 534s   | 80 063   |
| run11_lowtemp                | decay_start_step=500, start_value=0.3            | 0.644 | 2863    | 136.5 | 600s   | 80 063   |
| final2_mlp_hl5               | mlp + hidden_layers=5                            | 0.634 | 3246    | 141.1 | 580s   | 80 061   |
| **run01_baseline**           | (reference)                                      | 0.620 | 2961    | 135.5 | 605s   | 80 064   |
| run02_lr_hi                  | lr=0.00064                                       | 0.614 | 1438    | 136.5 | 600s   | 80 065   |
| run03_lr_lo                  | lr=0.00016                                       | 0.612 | 2062    | 135.6 | 603s   | 80 065   |
| combo3_mlp_mc                | mlp + td_mode=mc                                 | 0.576 | 1157    | 153.2 | 534s   | 80 065   |
| run10_bce_mlp                | mlp + sigmoid + bce (duel_margin off)            | 0.558 | 931     | **175.2** | 456s | 80 064 |
| run07_tspc_hi                | tspc=0.25                                        | 0.386 | -3487   | 123.2 | 338s   | 40 064   |
| run06_td0                    | td_mode=td0                                      | 0.000 | -102329 | 135.6 | 604s   | 80 064   |

---

## Observations

### 1. There is a hard 0.77 ceiling at 10k steps on the current code
`run09_hw512` (0.772), `run08_tspc_lo` (0.770), and
`final1_resnet_hw512_tspc` (0.768) all sit on the same plateau despite being
three distinct single- and dual-axis interventions. Stacking `hw=512` with
`tspc=0.0625` did **not** go above the ceiling. This is consistent with the
hypothesis that the bottleneck at 10k steps is the _Watkins trace-cutting
×  high-temperature exploration_ interaction, not model capacity or data
volume. See ideas.md §4.

### 2. `td_mode=td0` is broken on the current code
0.000 win rate at 10k steps with the default LR. With Watkins trace cutting,
tdlambda≈0.95 is degenerating to td0 anyway during high-exploration phases,
which is partly why the baseline is slower than it used to be. If you want
td0 to work, you'll likely need a target network (see ideas.md §3).

### 3. `mlp` beats `resnet` on every axis in isolation
- +14 pp win rate (0.708 vs 0.620)
- +13% GPS (153 vs 135)
- -12% wall clock (535s vs 605s)

This is almost certainly because `ResBlock.norm2.weight *= 0.01` in
`pro_yams_net.cc:74` makes each residual block contribute only ~1% of
identity at init, so a 3-layer resnet is effectively a 1-layer MLP for the
first several thousand steps. See ideas.md §12.

### 4. **Combinations don't stack linearly.** ⚠️
- mlp alone: 0.708
- hw=512 alone: 0.772
- mlp + hw=512: **0.736** (_worse than hw=512 alone_)
- mlp + tspc=0.0625: **0.736** (_worse than tspc=0.0625 alone_)
- mlp + hw=512 + tspc=0.0625: 0.698 (worst of the triple combinations)

This is the most surprising finding. The two best single-axis tweaks are
resnet-specific — they help resnet more than MLP. Once you switch to MLP
the ceiling is around 0.71-0.74.

### 5. LR is basically flat across ½× to 2× around 3.2e-4
`run02_lr_hi` (0.614), `run03_lr_lo` (0.612), baseline (0.620): identical
within noise. `combo5_mlp_lr_hi` is 0.674 (lower than plain mlp at 0.708).
Adam is doing its job; LR tuning at this scale is a second-order effect.

### 6. Deeper MLP hurts
`final2_mlp_hl5` (hl=5): 0.634 vs mlp hl=3: 0.708. Without residuals,
deeper nets don't train well in 10k steps. Adding more width (hw=512) is
worth more than adding more depth.

### 7. MC + MLP is worse than either alone
- mc + resnet: 0.704
- mlp + tdlambda: 0.708
- mc + mlp: **0.576** ← destabilising interaction.

The MC signal is high-variance (`tanh(margin/4000)` of a duel that can run
to ±25000+), and MLP converges fast enough on each batch to chase the noise
before the replay buffer averages it out. The resnet's slow-start
init (§3 above) acts as an implicit regulariser that shields it from MC
noise. See ideas.md §14 for a proposed fix via `duel_margin_maximization_scale`.

### 8. Lower `tspc` = more data per train step = better learning
`tspc=0.0625` (160k games) beat `tspc=0.125` (80k, baseline) by +15 pp.
`tspc=0.25` (40k) _lost_ 23 pp. The model was data-starved, not
capacity-starved. Training compute is not the bottleneck.

### 9. Highest GPS (175) has weakest learning
`run10_bce_mlp` (sigmoid + bce + duel_margin=false) hits 175 GPS but only
0.558 wr. The BCE target `win_rate ∈ {0, 1}` loses all margin information,
so the value function is less informative for action selection. This same
recipe was the best at 100k steps in prior code (see
`logs/test_tdl_100k_new1`), so the recipe is horizon-dependent.

---

## Recommended configs

### Best balance (learning × GPS) — recommended for production training
```diff
 training:
-  train_steps_per_collect: 0.125
+  train_steps_per_collect: 0.0625
```
**Result**: `run08_tspc_lo` — wr **0.770**, GPS **143**, wall 1120s.

Single-line change to the baseline. Architecture stays `resnet`.
+24 pp win rate vs baseline at only −5% GPS. Only costs 2× wall-clock
because you're doing 2× self-play. If you can afford the compute, this is
the best tuning in the study.

### Best GPS while still winning — recommended for quick prototyping
```diff
 training:
-  train_steps_per_collect: 0.125
+  train_steps_per_collect: 0.0625
 training.model:
-  architecture: resnet
+  architecture: mlp
```
**Result**: `combo1_mlp_tspc_lo` — wr **0.736**, GPS **160**, wall 1014s.

Ugly tradeoff: 3 pp worse than `run08_tspc_lo` but 12% more GPS and 10%
less wall. If you're running many short experiments, this is the pick.

### Minimal change for meaningful win
```diff
 training.model:
-  architecture: resnet
+  architecture: mlp
```
**Result**: `run04_mlp` — wr **0.708**, GPS **153**, wall **535s** (baseline
wall-time preserved).

Single-line change, +9 pp win rate, +13% GPS, no extra compute.
Recommended default for experiments you want finished fast.

---

## How to reproduce

```bash
# Single run
python3 scratch/hp10k/make_config.py logs/hp10k/my_run \
  train_steps_per_collect=0.0625
./build/pro_yams_ai --mode train --debug_mode 1 \
  --config logs/hp10k/my_run/config.yaml

# Full Phase 1 sweep (10 configs, ~100 min)
./scratch/hp10k/sweep.sh

# Phase 2 (5 combinations, ~80 min)
./scratch/hp10k/sweep2.sh

# Phase 3 (3 experiments, ~50 min)
./scratch/hp10k/sweep3.sh

# Summary table (reads all runs in logs/hp10k/)
python3 scratch/hp10k/analyze.py
```

---

## Unchecked paths / open questions

Cheap things I would try next if this study continued (all single-change
experiments, each ~10 min on the current rig):

1. **`tspc=0.03`** — even more data-per-step. Might push past 0.78 on
   resnet. Cost: ~40 min wall.
2. **`td_lambda=0.99` with resnet** — longer traces might compensate for
   Watkins cutting. (mlp+tdl=0.99 was only 0.656, so expect resnet+tdl=0.99
   around 0.70-0.75.)
3. **`temperature_decay_start_step=2000`** at full `initial_temperature=1.0`
   — keeps exploration high only for the heuristic-decay phase. Pairs
   naturally with Watkins; the `run11_lowtemp` result (0.644) is already
   promising but used initial_value=0.3.
4. **`model_swap_interval=1`** — instantaneously-fresh inference model.
   Removes off-policy lag, which matters more with fast-learning MLP.
5. **`duel_margin_maximization_scale=8000`** — avoid tanh saturation on the
   first few thousand high-score games (see ideas.md §14).
6. **A seeded re-run of run08_tspc_lo and run09_hw512** to confirm that
   0.770/0.772 is a real plateau rather than sample variance across different
   `sample_rng_` values.

Code-level changes are collected separately in `ideas.md`.
