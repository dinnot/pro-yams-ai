"""Per-input-feature L1 weight analysis for the first projection layer.

Loads checkpoints/mlp-256/selfplay3/checkpoint_step_300000.model, pulls out
hidden_blocks.0.weight (shape [256, 986]), and summarises which of the 986
input features have the smallest connected-weight magnitude. Layout follows
src/engine/tensor.cc / tensor.h.
"""

import sys
import torch

CKPT = "checkpoints/mlp-256/selfplay3/checkpoint_step_300000.model"

ROW_NAMES = ["1s", "2s", "3s", "4s", "5s", "6s",
             "SS", "LS", "FH", "K", "STR", "U8", "Y"]
COL_NAMES = ["Down", "Free", "Up", "Mid", "Turbo", "UpDown"]
PLAYER_NAMES = ["me", "opp"]

NROW = 13
NCOL = 6
NPLR = 2


def build_feature_index():
    """Return list[str] of length 986: human-readable name for each feature."""
    names = []

    # Group A (312) : cell occupancy + value
    # for pi in (me, opp): for col 0..5: for row 0..12: [is_filled, value/max]
    for pi in range(NPLR):
        for col in range(NCOL):
            for row in range(NROW):
                names.append(f"A.fill[{PLAYER_NAMES[pi]},{COL_NAMES[col]},{ROW_NAMES[row]}]")
                names.append(f"A.val [{PLAYER_NAMES[pi]},{COL_NAMES[col]},{ROW_NAMES[row]}]")

    # Group B.1 (24): upper_sum, e_raw per (player, col)
    for pi in range(NPLR):
        for col in range(NCOL):
            names.append(f"B1.upper_sum[{PLAYER_NAMES[pi]},{COL_NAMES[col]}]")
            names.append(f"B1.e_raw    [{PLAYER_NAMES[pi]},{COL_NAMES[col]}]")

    # Group B.2 (84): per-col duel features (14 each)
    b2_subnames = [
        "rem_me", "rem_opp",
        "margin_now", "margin_E",
        "crush_diff_now", "crush_diff_E",
        "pts_to_2x_now", "pts_to_3x_now", "pts_to_4x_now", "pts_to_5x_now",
        "pts_to_2x_E",   "pts_to_3x_E",   "pts_to_4x_E",   "pts_to_5x_E",
    ]
    for col in range(NCOL):
        for sub in b2_subnames:
            names.append(f"B2.{sub}[{COL_NAMES[col]}]")

    # Group C (156): p_one per (player, col, row)
    for pi in range(NPLR):
        for col in range(NCOL):
            for row in range(NROW):
                names.append(f"C.p_one[{PLAYER_NAMES[pi]},{COL_NAMES[col]},{ROW_NAMES[row]}]")

    # Group D (14): globals
    for col in range(NCOL):
        names.append(f"D.coef[{COL_NAMES[col]}]")
    names.append("D.cells_filled")
    names.append("D.total_duel_now")
    names.append("D.total_duel_E")
    names.append("D.my_min>opp_max")
    names.append("D.my_max<opp_min")
    names.append("D.phase>50")
    names.append("D.phase>100")
    names.append("D.phase>140")

    # Group E (216): per (player, col) × 18 (3 horizons × (5 upper-probs + 1 upper-ev))
    horizon_tags = ["Tmin", "Tmid", "Tmax"]
    upper_targets = [60, 70, 80, 90, 100]
    for pi in range(NPLR):
        for col in range(NCOL):
            for h in horizon_tags:
                for tgt in upper_targets:
                    names.append(f"E.P_upper>={tgt}@{h}[{PLAYER_NAMES[pi]},{COL_NAMES[col]}]")
                names.append(f"E.EV_upper@{h}[{PLAYER_NAMES[pi]},{COL_NAMES[col]}]")

    # Group F (180): per (player, col) × 15 (3 horizons × (p_mid, ev_mid, p_low, ev_low, p_clean))
    f_subs = ["p_mid", "ev_mid", "p_low", "ev_low", "p_clean"]
    for pi in range(NPLR):
        for col in range(NCOL):
            for h in horizon_tags:
                for sub in f_subs:
                    names.append(f"F.{sub}@{h}[{PLAYER_NAMES[pi]},{COL_NAMES[col]}]")

    assert len(names) == 986, f"got {len(names)} names"
    return names


def group_of(idx: int) -> str:
    if idx < 312:  return "A"
    if idx < 336:  return "B1"
    if idx < 420:  return "B2"
    if idx < 576:  return "C"
    if idx < 590:  return "D"
    if idx < 806:  return "E"
    return "F"


def subgroup_of(idx: int, name: str) -> str:
    """Coarser bucket for averaging — e.g. 'A.fill', 'A.val', 'B2.margin_now', ..."""
    if name.startswith("A.fill"): return "A.fill"
    if name.startswith("A.val"):  return "A.val"
    if name.startswith("B1."):    return name.split("[")[0].strip()
    if name.startswith("B2."):    return name.split("[")[0].strip()
    if name.startswith("C."):     return "C.p_one"
    if name.startswith("D."):     return name.split("[")[0].strip() if "[" in name else name
    if name.startswith("E."):
        # E.P_upper>=N@H[...] or E.EV_upper@H[...]
        return name.split("[")[0].strip()
    if name.startswith("F."):
        return name.split("[")[0].strip()
    return "?"


def main():
    print(f"Loading {CKPT} ...")
    m = torch.jit.load(CKPT, map_location="cpu")

    W = None
    for n, p in m.named_parameters():
        if n == "hidden_blocks.0.weight":
            W = p.detach()
            break
    if W is None:
        sys.exit("hidden_blocks.0.weight not found")

    assert W.shape == (256, 986), f"unexpected shape {tuple(W.shape)}"
    print(f"hidden_blocks.0.weight shape: {tuple(W.shape)}")

    # Per-input L1: sum over output dim (256), one number per input feature.
    l1 = W.abs().sum(dim=0)            # shape [986]
    assert l1.shape == (986,)

    names = build_feature_index()

    # ---- overall stats ----
    total = float(l1.sum())
    mean = float(l1.mean())
    std = float(l1.std())
    print(f"\nOverall: total={total:.2f}  mean={mean:.4f}  std={std:.4f}  "
          f"min={float(l1.min()):.4f}  max={float(l1.max()):.4f}")

    # ---- per-group totals & means ----
    print("\n=== Per top-level group ===")
    print(f"{'group':<6}{'count':>6}{'sum':>14}{'mean':>10}{'%total':>10}")
    group_sum = {}
    group_cnt = {}
    for i in range(986):
        g = group_of(i)
        group_sum[g] = group_sum.get(g, 0.0) + float(l1[i])
        group_cnt[g] = group_cnt.get(g, 0) + 1
    for g in ["A", "B1", "B2", "C", "D", "E", "F"]:
        s = group_sum[g]; c = group_cnt[g]
        print(f"{g:<6}{c:>6}{s:>14.2f}{s/c:>10.4f}{100*s/total:>9.2f}%")

    # ---- per-subgroup means (sorted ascending — smallest contributors first) ----
    print("\n=== Per subgroup, mean L1 (ascending — least-contributing first) ===")
    sub_sum = {}
    sub_cnt = {}
    for i, nm in enumerate(names):
        sg = subgroup_of(i, nm)
        sub_sum[sg] = sub_sum.get(sg, 0.0) + float(l1[i])
        sub_cnt[sg] = sub_cnt.get(sg, 0) + 1
    rows = []
    for sg in sub_sum:
        s = sub_sum[sg]; c = sub_cnt[sg]
        rows.append((s / c, sg, c, s))
    rows.sort()
    print(f"{'subgroup':<32}{'count':>7}{'mean':>10}{'sum':>14}")
    for mean_v, sg, c, s in rows:
        print(f"{sg:<32}{c:>7}{mean_v:>10.4f}{s:>14.2f}")

    # ---- 30 individual features with the smallest L1 ----
    print("\n=== 30 individual features with smallest L1 (least connected) ===")
    order = torch.argsort(l1).tolist()
    print(f"{'idx':>5} {'L1':>9}  group   name")
    for k in order[:30]:
        print(f"{k:>5} {float(l1[k]):>9.4f}  {group_of(k):<6} {names[k]}")

    # ---- 15 features with the largest L1 (for contrast) ----
    print("\n=== 15 individual features with largest L1 (most connected) ===")
    for k in order[-15:][::-1]:
        print(f"{k:>5} {float(l1[k]):>9.4f}  {group_of(k):<6} {names[k]}")

    # ---- bottom-10% counts per top-level group ----
    print("\n=== Where do the bottom-10% of features (98 of them) live? ===")
    cutoff = order[:98]
    bucket = {}
    for k in cutoff:
        bucket[group_of(k)] = bucket.get(group_of(k), 0) + 1
    for g in ["A", "B1", "B2", "C", "D", "E", "F"]:
        print(f"  {g:<3}: {bucket.get(g, 0):>3} / {group_cnt[g]}  "
              f"({100*bucket.get(g,0)/group_cnt[g]:.1f}% of group)")


if __name__ == "__main__":
    main()
