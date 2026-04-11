# Pro Yams AI — High-Level Design Document

## 1. Project Overview

### Goal
Build a super-human level AI for the Pro Yams (Competitive Yams) game using a TD-gammon-style approach with a custom C++ engine and libtorch.

### Core Philosophy
No black-box frameworks. Every component is purpose-built for Pro Yams, fully understood, and manually optimized. The developer makes every architectural decision.

### Hardware Target
- **CPU:** AMD Ryzen 9900X (24 threads)
- **GPU:** NVIDIA RTX 5080
- **RAM:** 64 GB
- **OS:** Ubuntu 25.10
- **Crucial Context for AI:** Standard releases of PyTorch/libtorch 2.4.x do not natively support the sm_120 architecture yet. Expect compatibility warnings or "no kernel image" errors if standard GPU tensor operations are invoked without nightly builds. Handle these gracefully.

---

## 2. Game Rules — Pro Yams (Competitive Yams)

Pro Yams is a high-stakes, competitive variation of Yahtzee (Yams) designed for 1v1 play. It introduces strict column constraints, a "King of the Hill" scoring requirement, and a dueling system where point differences are multiplied based on performance. The game rewards strategic risk-taking, long-term planning across multiple columns, and the ability to read and react to an opponent's board.

### 2.1 The Basics

- **Players:** 2 (1v1)
- **Dice:** 5 standard six-sided dice
- **Turns:** Players alternate turns. On your turn, you roll all 5 dice, then may reroll any subset of them up to 2 more times (for a total of 3 rolls per turn). After your final roll, you must place a score in exactly one available cell on your board.
- **Game length:** Each player has 78 turns total (13 rows × 6 columns). The game ends when both players have filled all 78 cells.
- **Objective:** Win the overall "Duel" by outscoring your opponent across columns.

### 2.2 The Board

The board consists of **13 rows** and **6 columns**, giving each player 78 cells to fill over the course of the game. Each cell is the intersection of one row (which defines *what* you need to roll) and one column (which defines *when* you can fill it).

#### 2.2.1 The Rows

Rows define the scoring categories. Each row has specific dice requirements and a score formula.

**Upper Section (Rows 1–6: "Numbers")**

These rows score the sum of dice showing a specific face value.

| Row | Target | Score | Example |
|-----|--------|-------|---------|
| 1s | Ones | Sum of all 1s | Three 1s = 3 |
| 2s | Twos | Sum of all 2s | Four 2s = 8 |
| 3s | Threes | Sum of all 3s | Two 3s = 6 |
| 4s | Fours | Sum of all 4s | Five 4s = 20 |
| 5s | Fives | Sum of all 5s | Three 5s = 15 |
| 6s | Sixes | Sum of all 6s | Four 6s = 24 |

**Lower Section (Rows 7–13: "Combinations")**

| Row | Name | Dice Requirement | Score Formula | Min | Max |
|-----|------|-----------------|---------------|-----|-----|
| SS | Small Sum | Sum of all dice ≥ 20 | Sum of all 5 dice | 20 | 29 |
| LS | Large Sum | Sum of all dice ≥ 20 | Sum of all 5 dice | 20 | 30 |
| FH | Full House | 3 of a kind + 2 of a kind | 20 + sum of all 5 dice | 25 | 50 |
| K | Four of a Kind | 4 identical dice | 30 + (face value × 4) | 34 | 54 |
| STR | Straight | 1-2-3-4-5 or 2-3-4-5-6 | 45 (small) or 50 (large) | 45 | 50 |
| 8 | Under Eight | Sum of all dice ≤ 8 | 60 + 5 × (8 − sum) | 60 | 75 |
| Y | Yams | 5 identical dice | 75 + 5 × (face value − 1) | 75 | 100 |

Note: For Four of a Kind (K), only the 4 matching dice contribute to the score formula. The 5th die is ignored. For example, four 6s and a 3 scores 30 + (6 × 4) = 54, not 30 + 27.

**Scratching:** If you cannot meet the requirements for a row (or choose not to), you may "scratch" it by writing 0 in that cell. This is sometimes strategically necessary but has consequences (see Clean Column Bonus and SS/LS rules below).

#### 2.2.2 The Golden Rule: Competitive Scoring

This is the rule that makes Pro Yams fundamentally different from regular Yahtzee. To place a score in any cell, your score must be **greater than or equal to** the highest score currently in that same row **within the same column** across all players' boards.

Example: If your opponent scored 24 in their Free column's "6s" row, you must score at least 24 (four 6s) to place a score in your own Free column's "6s" cell. If you only have three 6s (= 18), you cannot place a non-zero score in that cell — you must either scratch it (0) or choose a different cell. However, your opponent's 24 in the Free column has no effect on your "6s" cell in the Down column or any other column.

All rules in Pro Yams are column-scoped: the Golden Rule, the SS/LS interlock, the upper section bonus, and the clean column bonus are all evaluated independently per column. Columns never affect each other.

#### 2.2.3 The SS/LS Interlock

Small Sum (SS) and Large Sum (LS) have a strict, interconnected relationship within each column:

1. **Base requirement:** Both require a minimum sum of 20 to score points.
2. **SS < LS ordering:** Within the same column, your SS must be strictly less than your LS (if both are filled with non-zero scores).
3. **SS maximum:** SS can never exceed 29 (since LS maximum is 30, and SS must be strictly less than LS).
4. **LS floor (column-scoped):** Within a given column, your LS must be strictly higher than the highest SS recorded by anyone in **that same column** (not across the entire board). For example, if your opponent scored SS = 25 in the Free column, then any LS placed in the Free column by either player must be > 25. An SS of 25 in the Free column has no effect on LS requirements in the Up column.
5. **Mutual destruction:** If you scratch (0) your SS in a column and LS is already filled in that column, then LS is automatically scratched too (and vice versa). If the paired cell is not yet filled, it is not immediately scratched, but it can only be scratched in the future — it can never receive a non-zero score.

#### 2.2.4 The Columns

Each column restricts the **order** in which its 13 rows can be filled:

| Column | Name | Constraint |
|--------|------|-----------|
| Down | Descending | Must fill strictly top to bottom: 1s first, then 2s, ..., ending with Yams |
| Free | Free | Can fill in any order |
| Up | Ascending | Must fill strictly bottom to top: Yams first, then Under 8, ..., ending with 1s |
| Mid | Middle-Out | Must start from one of the two middle rows (6s or SS) and expand outward to adjacent cells |
| Turbo | x2 Turbo | Can fill in any order, BUT limited to 2 rolls per turn instead of 3 |
| UpDown | Connected | First score can go anywhere. All subsequent scores must be placed immediately above or below an existing filled cell in that column |

#### 2.2.5 Column Coefficients

At the start of each game, the six values **{8, 10, 12, 14, 16, 18}** are randomly shuffled and assigned one to each column. Both players share the same coefficient assignment. These coefficients multiply the final duel score difference for each column, making high-coefficient columns worth fighting harder for.

### 2.3 Bonuses

#### 2.3.1 Upper Section Bonus

For each column, the raw sum of scores in rows 1–6 earns a progressive bonus:

| Upper Sum | Bonus |
|-----------|-------|
| < 60 | +0 |
| 60–69 | +30 |
| 70–79 | +50 |
| 80–89 | +100 |
| 90–99 | +200 |
| ≥ 100 | +500 |

This bonus is calculated per column, per player.

#### 2.3.2 The Clean Column Bonus

A player earns the Clean Column bonus for a column if both conditions are met:

1. The upper section sum (rows 1–6) in that column is **≥ 60**.
2. **None** of the lower section cells (SS, LS, FH, K, STR, Under 8, Yams) in that column contain a 0 (scratch).

Note: A 0 in an upper row (1s–6s) is acceptable as long as the total upper sum reaches 60+. The clean column bonus value depends on the duel context (see below).

### 2.4 Winning the Game: The Duel

The game is not won by raw total score. Instead, players fight a **duel** on each of the 6 columns, and the overall winner is determined by the sum of all duel results.

#### 2.4.1 Duel Calculation (per column)

**Step 1 — Raw Scores:** For each column, calculate each player's raw score: sum of all 13 cells + upper section bonus. (Do not include the clean column bonus yet.)

**Step 2 — Crush Multiplier:** Compare your raw score to your opponent's:

| Condition | Multiplier |
|-----------|-----------|
| Your raw ≥ 5× opponent's (or opponent = 0 and you > 0) | 5× |
| Your raw ≥ 4× opponent's | 4× |
| Your raw ≥ 3× opponent's | 3× |
| Your raw ≥ 2× opponent's | 2× |
| Otherwise | 1× (no crush) |

The highest applicable multiplier is used. The crush is checked in both directions — either player can crush the other.

**Step 3 — Apply Clean Column Bonus:** If a player earned the Clean Column bonus, it is added to their score. The bonus value depends on the crush context:

- If a crush multiplier (2×–5×) is active in this column: clean bonus = **100 points**
- If no crush (1×): clean bonus = **200 points**

**Step 4 — Final Duel Points:**
1. Difference = (Your adjusted score − Opponent's adjusted score)
2. Duel points = Difference × Crush multiplier × Column coefficient

#### 2.4.2 Game Result

The **final score** is the sum of all 6 column duel points. A positive total means you win; negative means you lose; zero is a draw (extremely rare). For AI training purposes: win = 1.0, loss = 0.0, draw = 0.5.

---

## 3. Framework & Infrastructure

### Language & Dependencies
- **C++** (standard: C++17 or C++20) for the entire codebase
- **libtorch** (PyTorch C++ frontend) for neural network inference and training
- **CMake** as the build system (native libtorch/CUDA integration)
- No OpenSpiel, no Python in the critical path

### Project Structure
```
pro_yams_ai/
├── CMakeLists.txt
├── config/                  # Hyperparameter configs (YAML/JSON)
├── src/
│   ├── engine/              # Game state, rules, legal moves, scoring
│   ├── solver/              # Expectimax dice solver
│   ├── model/               # Neural network definition, inference, training
│   ├── self_play/           # Async game orchestration, data collection
│   ├── training/            # Training loop, replay buffer, TD targets
│   ├── heuristic/           # Heuristic bot for bootstrapping & evaluation
│   └── main.cpp             # Entry point, mode selection (train/eval/play)
├── checkpoints/             # Saved models and training state
└── logs/                    # Training metrics, evaluation results
```

### Development & Training
- Native compilation on Ubuntu 25.10, fast compile-run-debug cycles.
- Direct use of libtorch and CUDA for training.

---

## 4. Development Requirements

### Code Quality
- All code must be well-documented with clear comments explaining the *why*, not just the *what*. Public interfaces (classes, functions) should have doc comments describing purpose, parameters, return values, and any non-obvious behavior.
- Code should follow consistent naming conventions and be organized into logical modules with clear boundaries.

### Testing
- Unit tests should exist for most of the codebase, with particular emphasis on:
  - Game engine: scoring calculations, legal move generation, Golden Rule enforcement, SS/LS interlock, column constraints, duel computation
  - Solver: correctness of dynamic programming layers, transition probability math, hold mask evaluation
  - Tensor generation: correct feature computation, normalization, perspective flipping
- Tests should be runnable via CMake/CTest and executable as part of a standard build-and-test workflow.
- Test coverage should be comprehensive enough to catch rule implementation bugs early — these are the hardest to diagnose once training is underway.

### Performance Discipline
- **Performance is paramount.** All possible optimizations should be made during development, not deferred. The engine and solver are on the critical path (called millions of times during training), and every microsecond matters at scale.
- Performance should be continuously profiled, preferably through benchmarking tests that run as part of the test suite. Any code change or optimization must be validated with before/after profiling data to confirm it actually improves (or at minimum does not regress) performance.
- Testability must not come at the cost of production performance:
  - Prefer compile-time polymorphism (templates) over runtime polymorphism (virtual functions) in hot paths
  - Use dependency injection or template parameters for swapping implementations (e.g., NN vs heuristic win probability function) without virtual dispatch overhead
  - Test-only code (mocks, extra logging, assertions) should be behind compile flags or in separate test-only translation units, never in production builds
- Profile early and often. Performance regressions should be caught before they compound.

---

## 5. Architecture Overview

### The TD-Gammon Approach
The AI learns a **state value function** V(s) that predicts the probability of winning from a given board position (output in [0, 1]). There is no policy network. Decisions are made by evaluating all possible afterstates and selecting the best one.

### Key Principle: Perspective Consistency
V(s) always evaluates from the perspective of the player who is **not** about to move (i.e., the player who just placed a score). When comparing V(s) from the current player's perspective against V(s') from the opponent's perspective, use **1 - V(s')** to align perspectives.

### Component Interaction
```
┌─────────────┐     observations      ┌──────────────┐
│  Self-Play   │ ──────────────────►  │  Inference    │
│  (N games    │ ◄──────────────────  │  Thread       │
│   on CPU)    │     V(s) values      │  (GPU batch)  │
└──────┬───────┘                      └──────┬────────┘
       │ transitions                         │ shared model
       ▼                                     │
┌─────────────┐     gradients         ┌──────┴────────┐
│  Replay     │ ──────────────────►  │  Training     │
│  Buffer     │                       │  Thread       │
└─────────────┘                       │  (GPU)        │
                                      └───────────────┘
```

---

## 6. Game Engine

### Responsibilities
- Board state representation (13 rows × 6 columns × 2 players)
- Legal move generation respecting all column constraints (Down, Free, Up, Mid, Turbo, UpDown)
- Score calculation for all row types
- Golden Rule enforcement (score must ≥ highest existing score in that row)
- SS/LS interlock logic (mutual destruction, ordering constraints)
- Column coefficient assignment (random shuffle of {8, 10, 12, 14, 16, 18})
- Duel score computation at game end (crush multipliers, clean column bonuses)
- Terminal state detection and win/loss/draw determination

### Design Constraints
- Zero ML dependencies — pure C++ with no libtorch includes
- Optimized for millions of calls per second
- The engine determines win/loss (not score margin). A positive final duel score = win (1.0), negative = loss (0.0), zero = draw (0.5).

### Column Coefficients
Both players share the same coefficient assignment. Coefficients are randomly shuffled at game start and remain fixed for the duration of the game. The assignment is part of the game state and must be included in the observation tensor.

### Turn Structure
- Starting player is chosen randomly each game
- Each player gets 13 turns per column × 6 columns = 78 total placements
- Each turn: up to 3 rolls (1 initial + 2 rerolls), then mandatory placement

---

## 7. Expectimax Solver

### Role
Given current dice, rolls remaining, and access to a win probability function, the solver determines the optimal action: which dice to hold and reroll, or whether to stop and score immediately.

### Algorithm (Dynamic Programming, Bottom-Up)

**Layer 0 — No rerolls left:**
For each of the 252 sorted dice states, enumerate all legal cell placements. For each placement, compute the resulting board state (cell filled, dice cleared) and call the win probability function. Store the best win probability as V0[dice_state].

**Layer 1 — 1 reroll left:**
For each of the 252 dice states:
- **Stop value:** V0[current_dice] (score with current dice now)
- **Reroll value:** For each of the 32 hold masks, compute the probability-weighted expected value over all resulting dice states using V0
- V1[dice_state] = max(stop value, best reroll value)

**Layer 2 — 2 rerolls left:**
Same structure as Layer 1, but referencing V1 for reroll values and V0 for stop values.

### Win Probability Function
In the full system, this is a neural network inference call. During heuristic play, it is replaced by: score × column_coefficient (no NN needed).

### Precomputed Tables (from existing solver)
- 252 sorted dice states with ID mapping
- 32 hold masks per state
- Transition probability tables (held dice → probability distribution over resulting states)
- These are computed once at startup and reused for all solver calls

### Exploration
Softmax with temperature is applied at two levels:
1. **Placement selection:** Instead of argmax over V(afterstate), sample proportionally with temperature
2. **Hold mask selection:** Instead of argmax over hold mask EVs, sample proportionally with temperature

Temperature anneals over training (high early → low later).

---

## 8. Neural Network

### Architecture
- **Type:** Feedforward Neural Network (MLP)
- **Hidden activations:** ReLU
- **Output activation:** Sigmoid (win probability in [0, 1])
- **Hidden layers:** 2–3 (configurable hyperparameter)
- **Hidden width:** 256–512 neurons per layer (configurable hyperparameter)
- **Framework:** libtorch

### Input: State Observation Tensor
Feature-rich, normalized encoding of the board position. Includes both players' boards (current player always first), precomputed features to aid learning. All values normalized to roughly [0, 1] range.

Key feature categories (detailed design TBD in implementation planning):
- Both players' cell values (normalized by max possible score per row)
- Column coefficients (normalized)
- Upper section sums per column per player (normalized)
- Clean column eligibility flags per column per player
- Cells remaining per column per player (normalized)
- Row-level "king of the hill" maximums (the Golden Rule threshold per row)
- SS/LS scratch status per column per player
- SS/LS dependency state (is LS filled? what constrains SS?)
- Game progress (fraction of cells filled)

### Output
Single scalar: P(current player wins) ∈ [0, 1]

### Perspective Convention
The tensor is always constructed with the evaluating player's data first, opponent's data second. V(s) is always called after a placement, representing "I just placed, opponent moves next, what's my win probability?"

---

## 9. Training Pipeline

### Overview
Self-play and training run simultaneously. Self-play generates game data using the current frozen model. Training consumes data from the replay buffer and periodically swaps updated weights into the inference model.

### Self-Play (Asynchronous Architecture)
- N games run concurrently on CPU threads
- Each game progresses independently through turns
- When a game needs NN evaluation (solver computing afterstate values), it pushes observation tensors onto a shared queue and yields
- A dedicated **inference thread** collects observations from the queue, batches them into a single tensor, runs a GPU forward pass, and returns results to the requesting games
- Batch collection uses a size threshold or timeout (whichever comes first) to balance latency vs throughput
- Completed game trajectories (sequences of states and the terminal outcome) are pushed to the replay buffer

### Replay Buffer
- **Bounded buffer** holding the last M game trajectories (or K transitions)
- New games are appended, oldest are evicted when capacity is reached
- Buffer size is a configurable hyperparameter (target: 10–50× training batch size)
- Stores: (board_state_tensor, td_target) pairs

### TD Target Computation
Configurable mode (hyperparameter):
- **TD(0):** Target = 1 - V(s') where s' is the next state from opponent's perspective using the current network
- **TD(λ):** Blended target using exponential decay over future states. Lambda is a hyperparameter.
- **Monte Carlo:** Target = terminal game outcome (1.0 for win, 0.0 for loss, 0.5 for draw)

All three modes are supported. Short training runs determine which converges best.

### Training Loop
- Sample mini-batch from replay buffer
- Compute loss: MSE between V(s) and TD target
- Backprop and update weights (optimizer: Adam, learning rate configurable)
- Periodically copy updated weights to the inference model (every K training steps)

### Model Swap
The inference thread holds a frozen copy of the model. Periodically (configurable interval), training exports new weights and the inference thread atomically swaps to the updated model. Games in progress continue seamlessly — they just start getting slightly different evaluations.

---

## 10. Heuristic Bot (Evaluation Benchmark)

### Heuristic Bot
A fast, NN-free bot that uses the same solver logic but replaces the win probability function with: **score × column_coefficient**. This captures the fundamental insight that high scores in high-coefficient columns are valuable.

**Properties:**
- Beats random play 100% of the time by a large margin
- Loses to competent human players 100% of the time
- Requires zero NN inference — runs at maximum engine speed
- Used exclusively as an **evaluation benchmark** to measure training progress

### No Supervised Bootstrapping
The system starts from randomly initialized weights and goes straight to self-play. The feature-rich 809-dimension tensor (with precomputed probabilities, duel advantages, clean column eligibility, etc.) provides enough structure for the network to learn from scratch without teacher guidance. This avoids the risk of the network over-fitting to heuristic play patterns and getting stuck in local optima that are hard to escape. This follows the TD-gammon approach, which achieved world-class play starting from random weights.

---

## 11. Evaluation & Progress Tracking

### Primary Metric
Win rate against the heuristic bot, measured periodically during training. As the NN improves, this should approach and eventually reach ~100%.

### Secondary Metrics
- Training loss over time
- Average V(s) at game start (should stabilize near 0.5 for balanced play)
- Win rate as first player vs second player (should be roughly equal)
- Temperature value at current training stage

### Evaluation Protocol
Periodically pause training, run N evaluation games (NN vs heuristic bot, greedy play with no exploration), record win rate, log results. Resume training.

---

## 12. Checkpointing & Resumability

### What Gets Saved
- Model weights (libtorch serialization)
- Optimizer state (Adam moments)
- Replay buffer contents
- Training iteration count
- Current temperature / epsilon values
- Configuration snapshot (hyperparameters used for this run)

### Checkpoint Frequency
Configurable (e.g., every K training steps or every N minutes). Keep the last few checkpoints to allow rollback if training degrades.

### Resume
Load checkpoint, reconstruct all state, continue training seamlessly. Self-play restarts fresh games (no need to save in-progress games).

---

## 13. Configuration Management

All hyperparameters in a single config file (YAML or JSON), loadable at startup without recompilation:

```yaml
# Network
hidden_layers: 3
hidden_width: 256
learning_rate: 0.001
optimizer: adam

# Training
td_mode: mc                # mc recommended (targets never go stale in replay buffer)
td_lambda: 0.7             # only used if td_mode = td_lambda
training_batch_size: 256
replay_buffer_size: 2000000 # ~6.5GB. Large buffer prevents policy oscillation.
model_swap_interval: 100   # training steps between model swaps

# Self-Play
num_concurrent_games: 512
inference_batch_size: 512
inference_timeout_ms: 5

# Exploration
placement_temperature: 1.0
hold_temperature: 1.0
temperature_decay: 0.999
min_temperature: 0.05

# Evaluation
eval_interval: 1000        # training steps between evaluations
eval_games: 200

# Checkpointing
checkpoint_interval: 5000  # training steps
max_checkpoints: 5
```

---


---

## 15. Hyperparameter Tuning Strategy

### Philosophy
Rather than committing to a single configuration and hoping it works, the system is designed so that all key parameters are configurable without recompilation. Training runs should be short and diagnostic before committing to long runs. The goal is to find a configuration that shows clear learning signal within 1–2 hours, then scale it up.

### Phase 1: Architecture Search (1–2 hours per run)

Run short self-play training experiments varying the network architecture. Each run starts from random weights.

**Configurations to try:**

| Run | Layers | Width | Learning Rate | Notes |
|-----|--------|-------|---------------|-------|
| A1 | 2 | 256 | 0.001 | Baseline small |
| A2 | 2 | 512 | 0.001 | Wider |
| A3 | 3 | 256 | 0.001 | Deeper |
| A4 | 3 | 512 | 0.001 | Deeper + wider |
| A5 | 3 | 256 | 0.0003 | Lower LR |
| A6 | 3 | 256 | 0.003 | Higher LR |

**What to measure:** Training loss convergence speed and win rate against the heuristic bot. Pick the configuration that achieves the highest heuristic bot win rate fastest.

**Duration:** ~1–2 hours each. Total: ~6–12 hours.

### Phase 2: TD Mode Comparison (2–4 hours per run)

Using the best architecture from Phase 1, compare TD learning modes with actual self-play. MC is the recommended starting point because ground-truth game outcomes never go stale in the replay buffer, avoiding value divergence from off-policy bootstrapping.

| Run | TD Mode | Lambda | Buffer Size | Notes |
|-----|---------|--------|-------------|-------|
| T1 | MC | — | 2,000,000 | Recommended baseline (stable, no stale targets) |
| T2 | TD(0) | — | 50,000 | Small buffer, fast cycling (targets stay fresh) |
| T3 | TD(λ) | 0.5 | 1,000,000 | Balanced buffer and target blending |
| T4 | TD(λ) | 0.8 | 2,000,000 | Large buffer, mostly MC-like targets |

**What to measure:** Win rate against heuristic bot at 1-hour and 2-hour marks. Also track training loss stability — volatile loss suggests the TD mode or buffer configuration is problematic.

**Duration:** ~2–4 hours each. Total: ~8–16 hours.

### Phase 3: Exploration Tuning (2–4 hours per run)

Using the best architecture and TD mode, tune exploration parameters:

| Run | Placement Temp | Hold Temp | Decay | Notes |
|-----|---------------|-----------|-------|-------|
| E1 | 1.0 | 1.0 | 0.999 | High exploration |
| E2 | 0.5 | 0.5 | 0.999 | Moderate |
| E3 | 1.0 | 0.3 | 0.999 | Explore placements, not holds |
| E4 | 1.0 | 1.0 | 0.9995 | Slower decay |

**What to measure:** Win rate against heuristic bot over time. Too little exploration = fast initial improvement but plateau. Too much = slow convergence but potentially higher ceiling.

**Duration:** ~3 hours each. Total: ~12 hours.

### Phase 4: Full Training Run (remaining hours)

With the best configuration from Phases 1–3, commit to a long training run with the remaining budget (~60–70 hours). Monitor win rate against heuristic bot, looking for:

- **By hour 5:** Should consistently beat heuristic bot >80%
- **By hour 20:** Should beat heuristic bot >95%
- **By hour 50+:** Evaluate against human play patterns; look for strategic behaviors like column sacrifice, crush targeting, and SS/LS manipulation

If learning plateaus, consider: increasing network size, adjusting replay buffer size, or lowering minimum temperature to reduce exploration.

### Tuning Infrastructure

All runs should log to structured output (CSV or JSON) so results can be compared easily. Each run should checkpoint at regular intervals so it can be resumed if promising. The evaluation harness (periodic games against heuristic bot) must run identically across all experiments for fair comparison.

---

## 16. Next Steps — Detailed Implementation Plan

This document covers the high-level architecture. The detailed implementation plan is documented as a series of sequential development tasks, each in its own file. Each task builds on the previous ones, and the complete set produces a fully working system ready for training.

| Task | File | Description |
|------|------|-------------|
| 01 | `01_project_scaffolding_cmake.md` | Directory structure, CMake build system, gtest/gbench, libtorch integration |
| 02 | `02_game_engine_core.md` | Board state, scoring, legal moves, Golden Rule, SS/LS interlock, duel computation |
| 03 | `03_game_flow.md` | Game loop, dice rolling, hold/reroll, Turbo enforcement, RNG |
| 04 | `04_solver_precomputed_tables.md` | Dice states, hold masks, transition probabilities, score tables |
| 05 | `05_solver_heuristic.md` | Expectimax solver (two-step API), heuristic bot, exploration |
| 06 | `06_state_tensor.md` | 809-feature tensor design, normalization, probability precomputation |
| 07 | `07_neural_network.md` | libtorch MLP, inference engine, trainer, checkpoints |
| 08 | `08_solver_nn_async.md` | NN integration, async self-play architecture (workers + coordinator + queues) |
| 09 | `09_replay_buffer_training.md` | Replay buffer, training loop, model swap, exploration decay, logging |
| 10 | `10_evaluation_system.md` | Periodic NN vs heuristic benchmark games, win rate tracking |
| 11 | `11_configuration_management.md` | YAML config, CLI overrides, wired main.cpp, signal handling |
| 12 | `12_ui.md` | Web UI: bot vs bot viewer, human vs bot play, training dashboard |
| 13 | `14_mc_rollout_bot.md` | MC rollout-enhanced bot with time-budgeted search (nice-to-have) |
