# Known Bugs (Deferred Fixes)

Bugs identified but not yet fixed because the active trained model
depends on the current (buggy) behavior. Fix during the next training
run / model retrain.

---

## 1. DP solver ignores the SS-vs-LS constraint when bounding LS

**File:** `src/solver/dp_eval.cc` — `build_Sc()`

**Rule:** "Your LS must be strictly higher than the highest SS recorded by
anyone." Enforced at scoring time in `src/engine/scoring.cc:79-80` and at
candidate filtering in `src/solver/solver.cc:60-62`.

**Bug:** When computing the minimum threshold the DP must hit for LS,
`build_Sc()` only snaps to `ctx.golden_max[col][kRowLS]` and ignores
`ctx.golden_max[col][kRowSS]`. Because `max_SS` in a column can exceed
`max_LS` (placing SS does not check the global LS gmax, only the same
player's own LS — see `scoring.cc:67-69`), the DP underestimates the LS
threshold and overestimates the probability of hitting a valid LS. This
biases EV upward for LS-related rollouts in MCTS and the heuristic bot.

**Why deferred:** The current trained model has learned around this
biased EV signal. Fixing it now would shift the tensor / heuristic
distribution and degrade the live model until retrained.

**Fix (apply at next retrain):**

```cpp
if (board.cells[p][col][kRowLS] != kCellEmpty) {
    Sc_M[1] = -1;
} else {
    // LS must beat both existing LS and existing SS (rule enforced in
    // scoring.cc:79-80). Bound the min threshold by max(LS_gmax, SS_gmax+1).
    int min_ls = ctx.golden_max[col][kRowLS];
    int max_ss = ctx.golden_max[col][kRowSS];
    if (max_ss > 0 && max_ss + 1 > min_ls) {
        min_ls = max_ss + 1;
    }
    Sc_M[1] = snap_gmax(min_ls, kValsMid, 13);
    if (ctx.ss_scratched[p][col]) Sc_M[1] = 31;
    ++EM;
}
```

**Retrain checklist when fixing:**
- Regenerate any cached tensors / replay buffers.
- Re-baseline heuristic bot EV against MC rollouts (LS-heavy positions
  should now show lower predicted EV).
- Sanity-check that `compute_pc_data` in `src/engine/tensor.cc` (which
  also reasons about LS thresholds for `p_one`) is consistent with the
  DP path — it already applies `max(golden_min, max_ss + 1)` at
  `tensor.cc:217`, so the DP fix brings the two into agreement.
