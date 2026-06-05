# **Competitive Yams (Pro Yams) — Definitive Game Rules**

"Competitive Yams" is a high-stakes, cutthroat variation of Yahtzee (Yams) designed exclusively for 1v1 or 2v2 play. It introduces strict column constraints, a "King of the Hill" scoring requirement, and an aggressive dueling system where point differences are multiplied based on performance.

## **1. The Basics**

* **Players:** 2 (1v1) or 4 (2v2).
* **Dice:** 5 standard six-sided dice.
* **Turns:** On your turn, you have up to **3 rolls** (one initial roll and 2 re-rolls). You may hold any number of dice between rolls.
* **Objective:** Score the most points by winning "Duels" on specific columns against your opponents.

## **2. The Board**

The game sheet consists of **13 Rows** and **6 Columns**. In 2v2, every player has their own sheet.

### **The Rows & Scoring**

| Row | Name | Target | Score Calculation |
| :---- | :---- | :---- | :---- |
| **1-6** | Numbers | Specific Dice | Sum of that number (e.g., three 5s = 15) |
| **SS** | Small Sum | Sum of all dice | Sum of all 5 dice (Min. 20 required) |
| **LS** | Large Sum | Sum of all dice | Sum of all 5 dice (Min. 20 required) |
| **FH** | Full House | 3 of a kind + 2 of a kind | 20 + Sum of all 5 dice |
| **K** | 4 of a Kind | 4 of a kind | 30 + (Face Value of the 4 matching dice × 4). The 5th die is ignored. |
| **STR** | Straight | 1-2-3-4-5 (Small) or 2-3-4-5-6 (Large) | 45 (Small) / 50 (Large) |
| **8** | Less than 8 | Sum of all dice ≤ 8 | 60 + [5 × (8 - Sum)] |
| **Y** | Yams | 5 of a kind | 75 + [5 × (Face Value - 1)] |

### **The Golden Rule: Competitive Scoring**

To validate a score in *any* box, your score must be **greater than or equal to the highest score currently recorded in that specific box by anyone else in the game** — including your teammate in 2v2 play.

* *Example (1v1):* If your opponent scored 24 in their *Free* column's "6" box, you must score at least 24 (four 6s) to fill your "6" box in the *Free* column.
* *Example (2v2):* If your teammate has already filled their *Free* "6" box with 24, you also need ≥ 24 to fill your own *Free* "6" box. Your teammate's high score raises the bar for you just as an opponent's would.

If your dice fail the threshold, you scratch the box (score 0).

### **The SS / LS Interlock**

Small Sum (SS) and Large Sum (LS) share a strict, interconnected relationship:

* **Base Requirement:** Both require a minimum sum of 20 to score points.
* **SS Constraints:** Your SS must be **strictly lower** than your own LS (if your LS is already filled). It cannot exceed 29 (since LS max is 30).
* **LS Constraints:** Your LS must be **strictly higher** than the highest SS recorded by anyone on the board (opponents and, in 2v2, your teammate).
* **Mutual Destruction:** If you scratch (0) your SS in a column, your LS in that column is automatically scratched. If you scratch your LS, your SS is automatically scratched.

### **Scratching (Scoring 0)**

Scratching a cell means placing a score of 0 in it. Scratching is **not a choice** or a special placement type. It simply means you placed your final dice in a cell where the computed score evaluates to 0 (either because your dice do not match the cell's criteria, or because your score fails to meet the Golden Rule threshold).

You **cannot voluntarily choose to score 0** in a cell if your current dice yield a valid score greater than 0 that beats the competitive threshold. For example, you cannot choose to scratch a Full House cell if your final dice represent a valid Full House for that cell.

### **The Lucky Yams (First-Roll Bonus)**

If your **very first roll** of a turn is a Yams (5 dice of the same face), you may write the **maximum possible score** in any *one* cell you are currently allowed to fill — **independent of the dice faces**. The value written is that row's theoretical maximum (e.g. **100** in Yams, **30** in the 6s box, **50** in Full House, **54** in 4-of-a-Kind, **75** in Less-than-8), still subject to the Golden Rule and the SS/LS interlock (so a Small Sum is capped just below an already-filled Large Sum, etc.). In 2v2 the Golden Rule still includes your teammate's scores.

* The bonus only applies on the **initial roll**. You keep the normal option to re-roll, but **re-rolling forfeits the bonus** (you then play out the turn under the usual rules).
* This is the **default** rule. It can be disabled in configuration for AI training runs that prefer the simpler engine behaviour (where a first-roll Yams scores normally, like any other roll).

### **The Columns**

At the start of the game, each of the 6 columns is randomly assigned a unique **Coefficient** (Multiplier) from the set: **8, 10, 12, 14, 16, 18**. This coefficient multiplies the final point difference won in that column, making high-coefficient columns the primary strategic targets.

Each column restricts **how** you can fill its rows:

* **Down:** Must be filled strictly from top to bottom (starts at 1, ends at Yams).
* **Free:** Can be filled in any order.
* **Up:** Must be filled strictly from bottom to top (starts at Yams, ends at 1).
* **Mid:** Must start from the middle (either 6 or Small Sum) and fill adjacent cells outwards. *Adjacency wraps around: if you fill Row 1 (1s), Row 13 (Yams) becomes adjacent, and vice versa.*
* **x2 (Turbo):** Can be filled in any order, but you are limited to **2 rolls** maximum per turn instead of 3.
* **Up/Down (Connected):** Your first score can be placed anywhere in this column. All subsequent scores must be placed immediately above or below an existing score. *Adjacency wraps around as in Mid.*

**Maximum Cell Availability:** At any given point during your turn, there are at most **32 available cells** to choose from:

* **Free:** 13 cells (all available until filled)
* **x2 (Turbo):** 13 cells (all available until filled)
* **Down:** 1 cell (next empty cell top-to-bottom)
* **Up:** 1 cell (next empty cell bottom-to-top)
* **Mid:** at most 2 cells (empty cells adjacent to the filled middle section)
* **Up/Down:** at most 2 cells (empty cells adjacent to the current filled block)

## **3. Bonuses**

### **Upper Section Bonus (Rows 1-6)**

The bonus is progressive based on the raw sum of your 1s through 6s in a specific column:

1. **< 60**: +0
2. **60 - 69**: +30
3. **70 - 79**: +50
4. **80 - 89**: +100
5. **90 - 99**: +200
6. **≥ 100**: +500

### **The Clean Column Bonus**

You achieve a Clean Column if you complete a column without any failed scores in the critical sections. Specifically, you earn this bonus if:

* Your raw score in the Upper Section (Rows 1-6) is **≥ 60**. (You may have a 0 in an individual upper row, provided the total sum still reaches 60+.)
* **NONE** of the middle or lower cells (SS, LS, FH, K, STR, 8, Y) contain a 0.

The numeric *value* of the bonus is not fixed — it depends on whether a Crush Multiplier is active in the duel where it is being applied (see Step 3 below). The Clean Column Bonus is **NOT** included in the "Raw Score" used to determine the Crush Multiplier.

## **4. Winning the Game (The Duel)**

The game is not won by raw total score, but by winning **Duels** on each column against your opponents.

### **The Duel Calculation**

For every column, you perform a direct comparison against an opponent:

**Step 1 — Calculate Raw Scores**
Raw Score = Sum of all boxes + Upper Section Bonus.
*Do not include the Clean Column Bonus here.*

**Step 2 — Determine the Crush Multiplier**
Compare your Raw Score to your opponent's Raw Score:

* If YourRaw ≥ 2 × OpponentRaw → **2x**
* If YourRaw ≥ 3 × OpponentRaw → **3x**
* If YourRaw ≥ 4 × OpponentRaw → **4x**
* If YourRaw ≥ 5 × OpponentRaw → **5x**
* *(If OpponentRaw is exactly 0 and yours is > 0, the multiplier is 5x.)*

**Step 3 — Apply Clean Column Bonuses**
If a player achieved the Clean Column Bonus, it is added to their score before the difference is calculated. The bonus *value* depends on whether a Crush Multiplier was triggered in Step 2:

* If there **IS** a Crush Multiplier (2x–5x) active → Clean Column Bonus = **+100**.
* If there is **NO** Crush Multiplier (1x) → Clean Column Bonus = **+200**.

**Step 4 — Calculate Final Duel Points**
* Difference = (Your Adjusted Score − Opponent's Adjusted Score)
* **DuelPoints = Difference × Crush Multiplier × Column Coefficient**

**Example Calculations (1v1):**

* **Scenario A (Crush Multiplier):** You have a clean column with Raw 430. Opponent has 200. Since 430 / 200 ≥ 2 → **2x** multiplier. Because a multiplier is in your favor, your clean column bonus is **+100**. Duel Points before the coefficient: ((430 + 100) − 200) × 2.
* **Scenario B (No Crush Multiplier):** You have a clean column with Raw 430. Opponent has 250. Since 430 / 250 < 2 → **no** multiplier (1x). Bonus is **+200**. Duel Points before the coefficient: ((430 + 200) − 250) × 1.

## **5. Game Modes**

### **1v1 Scoring**

You play 6 duels (one for each column) against your single opponent.

* **Final Score:** The sum of all 6 Duel Points.
* **Winner:** A positive total score means you win; a negative score means your opponent wins.

### **2v2 Scoring (Team Play)**

Players sit in a circle: **A** and **C** form Team 1; **B** and **D** form Team 2. Seating order around the table is **A → B → C → D**, so each player's two neighbors are always opponents (members of the other team).

**Turn Order:** Players take their turns in clockwise order: A, then B, then C, then D, then back to A, and so on.

**Duels Against Both Neighbors:** Each player duels both of their immediate neighbors (Left and Right). Every column therefore triggers **two duels per player**:

* Player A duels Player B (right neighbor) and Player D (left neighbor).
* Player B duels Player C and Player A.
* Player C duels Player D and Player B.
* Player D duels Player A and Player C.

**Per-Pairing Independence:** The Crush Multiplier and the Clean Column Bonus value are computed **independently for each pairing**. The same clean column can be worth +100 in one of your duels (because you crushed that specific neighbor) and +200 in the other duel (because you didn't crush the other neighbor). Likewise, your raw score can produce different crush multipliers against each neighbor.

**Per-Player Column Points:** A player's points for a single column equal the **sum of their duel points against both neighbors** for that column.
* Example: Player A's points in the *Down* column = DuelPoints(A vs B in Down) + DuelPoints(A vs D in Down).

**Team Score:** Sum each teammate's total duel points across all 6 columns:
* Team 1 total = (Player A's total duel points) + (Player C's total duel points).
* Team 2 total = (Player B's total duel points) + (Player D's total duel points).

**Winner:** The team with the highest combined total score wins.
