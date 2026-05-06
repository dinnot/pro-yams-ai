# Task 06: State Tensor Design (V2.1)

## Overview

Design and implement the feature-rich, normalized state observation tensor that the neural network evaluates. This tensor is the bridge between the game engine and the NN — it encodes everything the network needs to predict win probability from a board position.

The tensor is generated from a `BoardState` (no dice, no rolls — pure board position after a placement). It always represents the perspective of the player who just placed (whose opponent moves next).

> **V2.1 redesign (2026-05).** The tensor was redesigned to leverage the precomputed DP tables (`get_upper_prob`, `get_upper_ev`, `get_middle_prob`, `get_middle_ev`, `get_lower_prob`, `get_lower_ev`). Multi-turn PMF convolutions are no longer computed at tensor-generation time — they are O(1) lookups. The result is a denser 986-feature tensor with deeper game-theoretic structure (expected values, multi-horizon section forecasts, crush-level projections, signed duel margins).

## Prerequisites

- Task 02 completed (engine data structures, scoring, GameContext)
- Task 04 completed (precomputed tables, transitions)
- DP Tables module available (`src/solver/dp_tables.{h,cc}`) and cache populated at `cache/dp_tables/dp_v1.bin`

---

## 1. Probability & DP Precomputation

### 1.1 Single-Turn Achievement Probability (Group C)

For each row type and each minimum score threshold, precompute the probability of achieving a score ≥ threshold when dedicating a full turn (3 rolls with optimal holding toward that goal). This produces `prob_3rolls[row][threshold]` (normal columns) and `prob_2rolls[row][threshold]` (Turbo).

The Group C "1-turn non-scratch probability" features query these tables directly at `T = 1` — no compounding is performed at tensor generation time.

### 1.2 DP Tables (Groups B / E / F)

Six precomputed DP tables expose, in O(1):

| Function | Returns |
|----------|---------|
| `get_upper_prob(dp, v, Sc_U, T, R)` | P(upper sum increase ≥ R in T turns) |
| `get_upper_ev(dp, v, Sc_U, T, S_cur)` | E[upper points] including progressive bonus |
| `get_middle_prob(dp, v, Sc_M, T)` | P(no scratch on SS/LS in T turns) |
| `get_middle_ev(dp, v, Sc_M, T)` | E[middle points] |
| `get_lower_prob(dp, v, Sc_L, T)` | P(no lower-section scratch in T turns) |
| `get_lower_ev(dp, v, Sc_L, T)` | E[lower points] |

Indexed by `(T, variant, sc, [R | S])`. Variants: `FREE`, `TURBO`, `DOWN`, `UP`, `UPDOWN` (Mid uses `UPDOWN` at runtime).

The state encoders snap a per-cell `golden_max` constraint to the nearest legal Sc constraint value:

```cpp
constexpr int8_t kVals1s[]  = {-1, 0, 3, 4, 5};
constexpr int8_t kVals2s[]  = {-1, 0, 6, 8, 10};
constexpr int8_t kVals3s[]  = {-1, 0, 12, 15};
constexpr int8_t kVals4s[]  = {-1, 0, 16, 20};
constexpr int8_t kVals5s[]  = {-1, 0, 20, 25};
constexpr int8_t kVals6s[]  = {-1, 0, 24, 30};
constexpr int8_t kValsMid[] = {-1, 0, 21..31};                         // 13 entries
constexpr int8_t kValsFH[]  = {-1, 0, 27, 28, 30..48, 50};             // 24 entries
constexpr int8_t kValsK[]   = {-1, 0, 38, 42, 46, 50, 54};
constexpr int8_t kValsQ[]   = {-1, 0, 50};
constexpr int8_t kValsU8[]  = {-1, 0, 65, 70, 75};
constexpr int8_t kValsY[]   = {-1, 0, 80, 85, 90, 95, 100};
```

`Sc[i] = -1` indicates the cell is already filled. `Sc_M[i] = 31` is the forced-scratch sentinel applied when SS or LS scratch destroys the partner.

### 1.3 Horizon Allocator

When a column has `EU + EM + EL` empty cells and the queried horizon is `T` turns, the horizon is split proportionally among the three sections:

```cpp
int Ecol = EU + EM + EL;
int TU = Ecol > 0 ? (T * EU) / Ecol : 0;
int TM = Ecol > 0 ? (T * EM) / Ecol : 0;
int TL = T - TU - TM;
```

Three horizons are queried per (player, column): `T_min = EU + EM + EL`, `T_max = 78 - cells_filled(player)`, `T_mid = (T_min + T_max) / 2`.

---

## 2. Tensor Feature Design

### 2.1 Perspective Convention

- **"Me"** = the player who just placed (player index 0 in the tensor)
- **"Opponent"** = the player who moves next (player index 1 in the tensor)

The caller is responsible for passing the correct `player` parameter.

### 2.2 Layout Summary (986 features)

| Group | Features | Description |
|-------|----------|-------------|
| A | 312 | Cell values (is_filled, score / row_max) — 2 features × 2 players × 6 cols × 13 rows |
| B | 108 | Per-player upper/E_raw + my-perspective duel features per column |
| C | 156 | 1-turn non-scratch probability per cell |
| D |  14 | Global aggregates and phase flags |
| E | 216 | Upper-section P/EV at three horizons per (player, col) |
| F | 180 | Mid/Low P/EV + P_clean at three horizons per (player, col) |
| **Total** | **986** | |

---

### Group A — Cell Values (312)

Iterate `Player ∈ {me, opp} → Col ∈ [0,5] → Row ∈ [0,12]`. Two features per cell:

| # | Feature | Empty | Filled |
|---|---------|-------|--------|
| 1 | `is_filled` | 0.0 | 1.0 |
| 2 | `score / kMaxScorePerRow[row]` | 0.0 | normalized cell score |

`kMaxScorePerRow`: `{5, 10, 15, 20, 25, 30, 29, 30, 50, 54, 50, 75, 100}`.

Count: 2 × 6 × 13 × 2 = **312**.

---

### Group B — Per-player + My-perspective duel (108)

**B.1 — per-player × per-column (24 features):** for each player (me, opp) and each column:

| # | Feature | Normalization |
|---|---------|--------------|
| 1 | `upper_sum` | `min(1, x / 100)` |
| 2 | `E_raw = get_E_raw(p, col, EU+EM+EL)` | `min(1, x / 500)` |

`get_E_raw` apportions the column's empty-cell horizon and sums `get_upper_ev + get_middle_ev + get_lower_ev` plus already-filled lower-section scores.

**B.2 — my-perspective × per-column (84 features):** 14 features per column:

| # | Feature | Notes |
|---|---------|-------|
| 1 | `cells_rem_me / 13` | empty cells in this column for me |
| 2 | `cells_rem_opp / 13` | empty cells in this column for opp |
| 3 | `duel_now_signed` | `tanh((my_raw − opp_raw) × coeff × active_crush / 15000)` |
| 4 | `E_duel_signed` | `tanh((my_E − opp_E) × coeff × active_E_crush / 15000)` |
| 5 | `crush_level` | `(crush_my − crush_opp) / 5` |
| 6 | `E_crush_level` | same with expected raw scores |
| 7–10 | `pts_to_Nx(N, my_raw, opp_raw, scale)` | `N=2,3,4,5`, scales `500, 750, 1000, 1250` |
| 11–14 | `pts_to_Nx(N, my_E, opp_E, scale)` | same scales using expected scores |

`pts_to_Nx(N, me, opp, scale)`:

```cpp
inline float pts_to_Nx(int N, float me, float opp, float scale) {
    if (me >= opp) return  std::max(0.0f, N * opp + 1.0f - me) / scale;
    else           return -std::max(0.0f, N * me + 1.0f - opp) / scale;
}
```

`active_crush = max(crush_multiplier(my, opp), crush_multiplier(opp, my))`.

Count: 24 + 84 = **108**.

---

### Group C — 1-turn Non-scratch Probability (156)

For each player → column → row:

```
if cell is filled:                     out = 1.0
elif row==SS and ls_scratched[p][col]: out = 0.0
elif row==LS and ss_scratched[p][col]: out = 0.0
else:
    threshold = clamped Golden Rule minimum (with SS<LS / LS>SS adjustments)
    if threshold > kMaxScorePerRow[row]: out = 0.0
    elif col == kColTurbo:
        out = prob_2rolls_compound[row][threshold][1]
    else:
        out = prob_3rolls_compound[row][threshold][1]
```

Note: `prob_*_compound[row][threshold][1] == prob_*[row][threshold]` (single-turn probability — no compounding). Compounding happens implicitly inside the DP tables for Groups E/F.

Count: 2 × 6 × 13 = **156**.

---

### Group D — Global Aggregates (14)

| # | Features | Notes |
|---|----------|-------|
| 1 | 6 × `coefficients[col] / 18` | column coefficient row |
| 2 | `cells_filled / 156` | game progress |
| 3 | `tanh(Σ duel_now / 80000)` | sum of unscaled per-col duel margins, then tanh |
| 4 | `tanh(Σ E_duel / 100000)` | same for expected scores |
| 5 | `game_math_won` | `1.0` if `Σ my_raw_min > Σ opp_potential` else 0 |
| 6 | `game_math_lost` | `1.0` if `Σ my_potential < Σ opp_raw_min` else 0 |
| 7 | `is_midgame` | `cells_filled > 50` |
| 8 | `is_endgame` | `cells_filled > 100` |
| 9 | `is_final` | `cells_filled > 140` |

`my_raw_min` = sum of `compute_column_raw_score` (filled cells + realized upper bonus). `my_potential` = sum of `compute_column_potential_score` (max-fill empties + max upper bonus).

Count: 6 + 1 + 1 + 1 + 1 + 1 + 3 = **14**.

---

### Group E — Upper-section P/EV at three horizons (216)

For each player → column → `T ∈ {T_min, T_mid, T_max}`:

| # | Feature | Notes |
|---|---------|-------|
| 1–5 | `get_upper_prob(dp, v, Sc_U, TU, R)` for `R = max(0, target − cur_upper)` and `target ∈ {60, 70, 80, 90, 100}` | five bonus tiers |
| 6 | `clamp01(get_upper_ev(dp, v, Sc_U, TU, cur_upper) / 100)` | expected upper points |

Count: 2 × 6 × 3 × 6 = **216**.

---

### Group F — Mid/Low P/EV + P_clean at three horizons (180)

For each player → column → `T ∈ {T_min, T_mid, T_max}`:

| # | Feature | Notes |
|---|---------|-------|
| 1 | `get_middle_prob(dp, v, Sc_M, TM)` | P(no SS/LS scratch) |
| 2 | `get_middle_ev / 60` | expected middle points |
| 3 | `get_lower_prob(dp, v, Sc_L, TL)` | P(no lower scratch) |
| 4 | `get_lower_ev / 250` | expected lower points |
| 5 | `P_clean = lower_has_scratch ? 0 : P_upper(R=60−cur) × P_mid × P_low` | clean column probability |

Count: 2 × 6 × 3 × 5 = **180**.

---

## 3. Tensor Generation Function

```cpp
// src/engine/tensor.h

constexpr int kTensorSize = 986;

constexpr int kMaxScorePerRow[kNumRows] = {
    5, 10, 15, 20, 25, 30,   // 1s..6s
    29, 30,                  // SS, LS
    50, 54, 50, 75, 100      // FH, K, STR, U8, Y
};

void generate_tensor(const BoardState& board, const GameContext& ctx,
                     int player, const PrecomputedTables& tables,
                     float* out);
```

`PrecomputedTables` carries a `DPTables dp_tables` member; `init_precomputed_tables` auto-loads the cache from `$DP_TABLES_CACHE`, `cache/dp_tables/dp_v1.bin`, or `../cache/dp_tables/dp_v1.bin`. If no cache is found, DP-dependent features fall back to safe defaults (0).

**Implementation outline:**

```
int idx = 0;

// Group A (312): is_filled, score / row_max  (per p, col, row)

// Group B (108):
//   B.1 per-player × col: upper_sum/100, E_raw/500
//   B.2 my-perspective × col: cells_rem, duel_now_signed, E_duel_signed,
//        crush_level, E_crush_level, 4× pts_to_Nx(current), 4× pts_to_Nx(expected)

// Group C (156): 1-turn non-scratch P (filled→1.0, forced scratch→0.0,
//                Turbo→prob_2rolls_compound[..][1], else prob_3rolls_compound[..][1])

// Group D (14): coeffs, progress, tanh(Σdn/80k), tanh(ΣdE/100k),
//               game_math_won, game_math_lost, 3 phase flags

// Group E (216): for (p, col, T∈{T_min,T_mid,T_max}):
//   build_Sc → apportion → 5× get_upper_prob (R=tier−cur_upper) + get_upper_ev/100

// Group F (180): for (p, col, T∈{T_min,T_mid,T_max}):
//   middle_prob, middle_ev/60, lower_prob, lower_ev/250,
//   P_clean = lower_has_scratch ? 0 : P_upper(60-cur) * P_mid * P_low

assert(idx == kTensorSize);
```

### 3.1 Helper Functions

```cpp
// In tensor.cc anonymous namespace:
int8_t snap_gmax(int gmax, const int8_t* vals, int count);
Variant get_variant(int col);
void build_Sc(int p, int col, const BoardState&, const GameContext&,
              int8_t Sc_U[6], int8_t Sc_M[2], int8_t Sc_L[5],
              int& EU, int& EM, int& EL);
void apportion_turns(int T, int EU, int EM, int EL,
                     int& TU, int& TM, int& TL);
float get_E_raw(int p, int col, int T, const BoardState&, const GameContext&,
                const DPTables&);
float pts_to_Nx(int N, float me, float opp, float scale);

// Public API:
int compute_column_raw_score      (const BoardState&, const GameContext&, int p, int col);
int compute_column_potential_score(const BoardState&, const GameContext&, int p, int col);
int compute_total_potential       (const BoardState&, int p);
int count_empty_cells             (const BoardState&, int p, int col);
int count_filled_cells            (const BoardState&, int p);
int sum_all_filled                (const BoardState&, int p);
```

---

## 4. Performance Considerations

### 4.1 Why the V2.1 redesign

The V1 tensor compounded probabilities at tensor-generation time using `pow()`. Multi-turn PMF convolutions for upper/middle/lower sections would have added ms-scale latency per tensor — crippling for the GPU coordinator pipeline.

The DP tables condense all of that into O(1) lookups (~2 GB on disk, ~1 s load, ~48 s compute one-off). Tensor generation now performs only:
- A handful of `build_Sc` constructions per column (cheap, no allocation),
- `kDPNumVariants` × O(1) array indexes per query,
- Standard normalisation arithmetic.

### 4.2 Afterstate batching

`generate_tensor_batch` clones the BoardState/GameContext per request and patches them with the candidate placement. The DP tables themselves are read-only and shared across all requests.

```cpp
void generate_tensor_batch(const BoardState& board, const GameContext& ctx,
                            int player,
                            const AfterstateRequest* requests, int request_count,
                            const PrecomputedTables& tables,
                            float* out);
```

For each request: clone, apply placement (cell write + cells_filled++), patch `golden_max`, `upper_sum`, scratch flags / mutual destruction, then `generate_tensor` into `out + i * kTensorSize`.

---

## 5. File Organization

```
src/engine/
├── tensor.h        # kTensorSize=986, kMaxScorePerRow, generate_tensor, generate_tensor_batch
├── tensor.cc       # V2.1 implementation (Groups A/B/C/D/E/F + helpers)
└── probability_tables.h / probability_tables.cc

src/solver/
├── dp_tables.{h,cc}                # Query API + persistence
├── dp_tables_compute.cc            # One-off DP compute
└── precomputed_tables.{h,cc}       # Embeds DPTables; auto-loads cache
```

The engine library has no libtorch dependency — it produces raw float arrays. The model library (Task 07) wraps these into libtorch tensors.

---

## 6. Unit Tests

### 6.1 Probability table tests (`tests/engine/probability_table_test.cc`)

Sanity checks on `prob_3rolls` / `prob_2rolls`: monotonicity in threshold, correct boundary conditions, Turbo ≤ Normal.

### 6.2 Tensor generation tests (`tests/engine/tensor_test.cc`)

V2.1 layout offsets:

```cpp
constexpr int kGroupAStart = 0,   kGroupASize = 312;
constexpr int kGroupBStart = 312, kGroupBSize = 108;
constexpr int kGroupCStart = 420, kGroupCSize = 156;
constexpr int kGroupDStart = 576, kGroupDSize = 14;
constexpr int kGroupEStart = 590, kGroupESize = 216;
constexpr int kGroupFStart = 806, kGroupFSize = 180;
```

Test cases:

- `kTensorSize == 986` and group offsets sum to 986.
- Empty board: Group A all zero; Group C all > 0; phase flags zero; game progress = 0.
- All features in `[-1, 1]` (signed tanh / crush-diff features may be negative); no NaNs.
- Single placement at (Free, 6s, 18): Group A `(1.0, 18/30)` at the right slot; Group C probability = 1.0 at that cell; game progress = 1/156.
- Perspective flipping: Group A halves swap between `tensor0` and `tensor1`.
- SS/LS forced scratch: paired probability = 0.0.
- Coefficients normalised to `coefficients[col] / 18`.
- Mid-game smoke test: no crashes, all features finite and in range.

### 6.3 Batch generation tests (`tests/engine/tensor_batch_test.cc`)

- Batch matches per-request `generate_tensor` on a manually patched clone.
- Single-request batch matches single-call output.
- Group A `is_filled` is 1.0 at the placed cell for each request (offset uses 2 features per cell).
- All features in `[-1, 1]`; no NaNs.

---

## 7. Benchmarks

Add to `tests/benchmarks/tensor_bench.cc`:

- **BM_GenerateTensor:** single tensor generation (mid-game).
- **BM_GenerateTensorBatch:** batch of 100, 500, 1000 afterstates.
- **BM_DPLookup:** isolated `get_upper_prob` / `get_upper_ev` query throughput.
- **BM_PrecomputedTablesInit:** one-time init cost (with and without DP cache).

---

## 8. Definition of Done

This task is complete when:

1. `kTensorSize == 986` and all V2.1 groups produce the expected feature counts.
2. `generate_tensor` queries the DP tables for Groups B/E/F and falls back to zero when the DP cache is absent.
3. All features lie in `[-1, 1]` (signed groups) or `[0, 1]` (probabilities, magnitudes); no NaNs.
4. Perspective flipping works correctly (swapping player produces the expected Group A swap).
5. `generate_tensor_batch` matches per-request `generate_tensor` to within float tolerance.
6. SS/LS edge cases (mutual destruction, forced scratch, ordering constraints) handled.
7. All unit tests pass; CTest green for `tensor_tests` (and the rest of the suite).
8. `kTensorSize` is referenced symbolically everywhere (no leftover hardcoded literals in source — only in YAML configs).
9. `ModelConfig::input_size` defaults to 986, matching `kTensorSize`.
10. The DP cache auto-load probes `$DP_TABLES_CACHE`, `cache/dp_tables/dp_v1.bin`, and `../cache/dp_tables/dp_v1.bin`.
