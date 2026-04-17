# Ideas for code improvements — pro_yams_ai

_Generated during 10k-step hyperparam sweep experiments (2026-04-16)._
These are suggestions — none were applied. Each item lists the rationale so you
can judge whether it's worth building.

---

## 1. Learning-rate scheduler

`src/model/trainer.cc:24-27` constructs Adam with a single, constant LR.
With 10k-step budgets we see a clear tradeoff: high LR bootstraps faster but
ceiling is lower, low LR is slow early but more stable late. A schedule (e.g.
cosine decay, warmup+linear, or LR reduction on plateau) would let a single run
get fast early learning AND a clean late-training convergence.

**Suggested wire-up:**
```yaml
training.model:
  learning_rate: 0.00032
  lr_schedule: cosine          # or "constant" (default) | "warmup_linear"
  lr_min: 0.00005
  lr_warmup_steps: 500
```
Implement as a post-step hook in `TrainingLoop::run`, updating
`optimizer_->param_groups()[0].options()->lr = new_lr`.

---

## 2. Gradient clipping

No gradient clipping is currently present in `train_step`. When the replay
buffer churns quickly (low `replay_capacity`) and the target distribution
shifts, a single bad batch can produce a huge gradient. `torch::nn::utils::
clip_grad_norm_(model_->parameters(), 1.0)` before `optimizer_->step()` would
be trivial and protective.

**Why:** We saw loss jump from 0.18 → 0.32 between steps in some runs; likely
a symptom that would disappear with clipping.

---

## 3. Target network for TD bootstrapping

With `td_mode=td0` or `tdlambda` the bootstrap target is the CURRENT model's
prediction of the next state. Every `model_swap_interval=10` steps we refresh
the inference model from the trainer. This means targets are moving fast,
which is known-destabilising in Q-learning / TD families.

A target network (a slow-moving copy that updates only every e.g. 1000 steps
or via Polyak averaging) would stabilise TD learning. Especially relevant
since we already saw `td0` collapse at lr=1e-5 (grid_lr run_02).

---

## 4. Temperature-decayed Watkins trace cutting

`85a3511` added Watkins trace cutting when the chosen action is exploratory.
During early training the temperature is near 1.0, so nearly every action is
exploratory → traces are cut almost always → tdlambda behaves like td0.

This partially explains why lambda=0.95 is slow to bootstrap.

**Option A:** Treat an action as "exploratory" only if its sampled prob is
significantly below the greedy action's prob (e.g. `p_sampled < 0.5 * p_max`),
rather than any time `temperature > 0`. Softmax sampling at T=1 still picks
the greedy action most of the time.

**Option B:** Add a `lambda_schedule` that starts low (near 0) and grows to
its target as exploration cools, so the agent only relies on long traces once
its policy is stable.

---

## 5. Heuristic-blend annealing in eval runs

`run_evaluation` uses a raw `SolverConfig greedy_cfg`; the heuristic weight is
dropped entirely. But during self-play the heuristic linearly decays over
50k steps (`training_loop.cc:254`). So our eval always tests a weaker policy
than what the worker threads are actually playing.

At low training steps (say step 2000), 96% of the solver decisions during
self-play are still heuristic-blended, yet eval plays with pure NN. This makes
early eval win-rates look worse than they really are and adds noise to the
curve.

**Suggested:** expose a `eval_heuristic_weight` that either follows the same
decay schedule, or runs two evals (blended + pure) and logs both.

---

## 6. Prioritised replay (or recency-weighted sampling)

`ReplayBuffer` uses uniform random sampling. With a ring buffer of 30k samples
and 80k games contributing ~150k samples/sec, many batches contain stale data
before min_buffer_size is reached. Prioritised replay (higher weight for
samples with larger TD error) or a simple exponential-recency bias
(`p ∝ exp(-age/τ)`) would concentrate training on the data that carries real
learning signal.

**Light version:** bias the sampler toward the last 20% of the buffer without
full priority tracking — cheap and captures most of the benefit.

---

## 7. Log LR, heuristic_weight, and tdlambda to training_log.csv

`TrainingMetrics` (see `src/training/metrics.h`) logs `temperature`, `epsilon`,
`gps` but not `heuristic_weight`, `learning_rate`, nor the effective
`td_lambda` (if trace cutting were tracked). Adding these would make post-hoc
analysis much easier — right now we reconstruct them from the config.

---

## 8. Per-sample target-staleness metric

We have no visibility into how "fresh" the samples we're training on are.
Add a counter per sample of "batches_seen" or "age_in_steps_when_drawn" and
log the avg per training step. This would instantly flag the "replay_cap too
big for min_buffer" tuning mistake we saw in grid_lr runs 04-06.

---

## 9. Warmup eval with lower temperature

First few evals in a 10k run are essentially testing random policies (win rate
0.06 at step 2000). Those evals cost ~100s each and contribute no useful
signal for tuning. A `eval_start_step` parameter (skip evals before N) would
save 10-20% of wall clock on short sweeps.

---

## 10. Deterministic seed management

`sample_rng_(0xDEADBEEF12345678ULL)` is hard-coded. For hyperparam sweeps we
want each run to have a controllable seed so that variance across runs can be
distinguished from variance across seeds. Expose `training.seed` in
TrainingConfig.

---

## 11. Optimizer choice

Adam is a reasonable default, but for value-function regression (small net,
stable objectives) plain SGD with momentum often converges more smoothly and
to a flatter minimum. AdamW (with weight decay) would also be worth trying;
no weight decay is set currently.

---

## 12. Resnet init gamma is 0.01 — possibly too aggressive

`pro_yams_net.cc:74` sets `norm2->weight` to 0.01 for ReZero-style init. With
a 3-layer resnet this means block contributions start at 1% of identity.
Gradients through these blocks during early training are tiny. Try 0.1 (a
common initial value for ReZero/Fixup) and see if bootstrap speed improves.

---

## 13. Log GPU utilisation alongside GPS

We optimise for GPS and win-rate but have no visibility into whether the GPU
is the bottleneck or the CPU (assembly in coordinators takes ~60% of the loop
per `debug_coordinator.log`). A tiny `nvidia-smi dmon` snapshot every eval
interval, or even just `cudaMemGetInfo`, would be a cheap dashboard addition.

---

## 14. ~~No duel-margin clipping in MC mode~~  → verify

`use_duel_margin_maximization` maps duel margin through `tanh(margin/4000)`.
With some early games ending at duel=±25000+, this saturates tanh completely.
Check whether the distribution of targets is actually a healthy spread in
\[-1, 1\] at step 2000 or is bimodal at ±1. If bimodal, the MSE loss has no
gradient for these samples — increasing `duel_margin_maximization_scale` to
8000 or 10000 early on might help.
