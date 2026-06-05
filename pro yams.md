# **Competitive Yams (Pro Yams) \- Game Rules**

"Competitive Yams" is a high-stakes, cutthroat variation of Yahtzee (Yams) designed for 1v1 play. It introduces strict column constraints, a "King of the Hill" scoring requirement, and an aggressive dueling system.

## **1\. The Basics**

* **Players:** 2 (1v1).  
* **Dice:** 5 standard six-sided dice.  
* **Turns:** On your turn, you have up to **3 rolls** (one initial roll and 2 re-rolls). You may hold any number of dice between rolls.  
* **Objective:** Score the most points by winning "Duels" on specific columns against your opponent.

## **2\. The Board**

The game sheet consists of **13 Rows** and **6 Columns**.

### **The Rows & Scoring**

First, understand the scoring categories available in every column.

| Row | Name | Target | Score Calculation |
| :---- | :---- | :---- | :---- |
| **1-6** | Numbers | Specific Dice | Sum of that number (e.g., three 5s \= 15\) |
| **SS** | Small Sum | Sum of all dice | Sum of all 5 dice |
| **LS** | Large Sum | Sum of all dice | Sum of all 5 dice |
| **FH** | Full House | 3 of one kind \+ 2 of another | 20 \+ Sum of all dice |
| **K** | 4 of a Kind | 4 of a kind | 30 \+ Sum of all dice |
| **STR** | Straight | 1-2-3-4-5 (Small) or 2-3-4-5-6 (Large) | 45 (Small) / 50 (Large) |
| **8** | Less than 8 | Sum of all dice ≤ 8 | 60 \+ 5 × (8 \- Sum) |
| **Y** | Yams | 5 of a kind | 75 \+ 5 × (Face Value \- 1\) |

#### **Special Row Constraints**

* **Competitive Scoring (The Golden Rule)**: To validate a score in *any* box, your score must be **greater than or equal to** the highest score currently recorded in that specific box by your opponent.  
  * *Example:* If your opponent scored 24 in their *Free* column's "6" box, you must score at least 24 (four 6s) to fill your "6" box in the *Free* column. If you only roll three 6s (18), you score **0**.  
* **SS / LS Relationship**:  
  * **SS (Small Sum)**: Must be at least 20\. Must be strictly lower than your **LS** (if LS is already filled). If you scratched (0) your LS in this column, you must scratch SS as well (cannot put any score \> 0 in there). If you scratch LS after putting a score in SS, SS will become scratched as well.  
  * **LS (Large Sum)**: Must be at least 20\. Must be strictly higher than the highest **SS** recorded by anyone. If you scratched (0) your SS in this column, you must scratch LS as well (cannot put any score \> 0 in there). If you scratch SS after putting a score in LS, LS will become scratched as well.  
* **Scratching (Scoring 0\)**: Scratching a cell means placing a score of 0 in it. Importantly, scratching is **not a choice** or a special placement type. It simply means you placed your final dice in a cell where the computed score evaluates to 0 (either because your dice do not match the cell's criteria, or because your score fails to meet the Competitive Scoring threshold). You **cannot voluntarily choose to score 0** in a cell if your current dice yield a valid score greater than 0 that beats the competitive threshold. For example, you cannot choose to scratch a Full House cell if your final dice represent a valid Full House for that cell. Similarly, you cannot choose to score 0 in a "6s" cell if you have rolled two 6s and no one has recorded a score higher than 12 in that box.

#### **The Lucky Yams (First-Roll Bonus)**

If your **very first roll** of a turn is a Yams (5 dice of the same face), you may write the **maximum possible score** in any *one* cell you are currently allowed to fill — **independent of the dice faces**. The value written is that row's theoretical maximum (e.g. **100** in Yams, **30** in the 6s box, **50** in Full House, **54** in 4-of-a-Kind, **75** in Less-than-8), still subject to the Golden Rule and the SS/LS interlock (so a Small Sum is capped just below an already-filled Large Sum, etc.).

* The bonus only applies on the **initial roll**. You keep the normal option to re-roll, but **re-rolling forfeits the bonus** (you then play out the turn under the usual rules).
* This is the **default** rule. It can be disabled in configuration for AI training runs that prefer the simpler engine behaviour (where a first-roll Yams scores normally, like any other roll).

### **The Columns**

Each column determines **how** you are allowed to fill the rows above.

**Coefficients:** At the start of the game, each of the 6 columns is randomly assigned a unique **Coefficient** (Multiplier) from the following set: **8, 10, 12, 14, 16, 18**. This coefficient multiplies the point difference won in that column, making high-coefficient columns the primary strategic targets.

* **Down**: Must be filled strictly from top to bottom (starts at 1, ends at Yams).  
* **Free**: Can be filled in any order.  
* **Up**: Must be filled strictly from bottom to top (starts at Yams, ends at 1).  
* **Mid**: Must start from the middle (6 or Small Sum) and fill adjacent cells outwards (up or down). *Note: Adjacency wraps around the board. If you fill Row 1 (1s), Row 13 (Yams) becomes adjacent and available, and vice versa.*  
* **x2 (Turbo)**: Can be filled in any order, but you are limited to **2 rolls** maximum instead of 3\.  
* **Up/Down (Connected)**: You may place your first score anywhere in this column. All subsequent scores in this column must be placed immediately above or below an existing score (adjacent). *Note: Adjacency wraps around the board. If you fill Row 1 (1s), Row 13 (Yams) becomes adjacent and available, and vice versa.*

**Maximum Cell Availability:**

Because of the strict placement rules governing the columns, you will never have access to the entire board at once. At any given point during your turn, there are at most **32 available cells** to choose from:

* **Free:** 13 cells (all available until filled)  
* **x2 (Turbo):** 13 cells (all available until filled)  
* **Down:** 1 cell (the next available empty cell top-to-bottom)  
* **Up:** 1 cell (the next available empty cell bottom-to-top)  
* **Mid:** At most 2 cells (the empty cells immediately adjacent to the filled middle section)  
* **Up/Down:** At most 2 cells (the empty cells immediately adjacent to the current filled block)

## **3\. Bonuses**

### **Upper Section Bonus (Rows 1-6)**

Unlike standard Yahtzee, the bonus is progressive based on the sum of your 1s through 6s in a column:

1. **\< 60**: \+0  
2. **60 \- 69**: \+30  
3. **70 \- 79**: \+50  
4. **80 \- 89**: \+100  
5. **90 \- 99**: \+200  
6. **≥ 100**: \+500

### **Clean Column Bonus**

If you complete a column without scratching any box (no 0s), you receive a **Virtual Bonus** for that column. The value of this bonus depends dynamically on whether you have achieved a Crush Multiplier against your opponent in that specific column:

* If there is **no Crush Multiplier** in your favor (your Raw Score is less than 2× the opponent's), the bonus is **\+200 points**.  
* If there **is a Crush Multiplier** in your favor (your Raw Score is 2× or more than the opponent's), the bonus is **\+100 points**.

*Important:* This bonus is added to your score difference, but it is **NOT** included in the "Raw Score" used to determine the Crush Multiplier (see below).

## **4\. Winning the Game (The Duel)**

The game is not won by raw total score, but by winning **Duels** on each column against your opponent.

### **The Duel Calculation**

For every column, you perform a comparison against your opponent:

1. **Calculate Raw Scores**: Sum of boxes \+ Upper Bonus (Rows 1-6). *Do not include Clean Column Bonus yet.* 2\. **Determine Crush Multiplier**: Compare your Raw Score to your opponent's Raw Score.  
   * If YourRaw ≥ 2 × OpponentRaw → **2x** \* If YourRaw ≥ 3 × OpponentRaw → **3x** \* ...up to **5x** \* *(Note: If OpponentRaw is 0, multiplier is 5x).* 3\. **Calculate Final Difference**:  
   * Add your Clean Column Bonus (either \+100 or \+200, depending on the multiplier found in Step 2\) to your Raw Score, if applicable.  
   * Diff \= ((YourRaw \+ CleanColumnBonus) \- OpponentRaw)  
2. **Apply Coefficient & Multiplier**: DuelPoints \= Diff × ColumnCoefficient × CrushMultiplier

**Example Calculations:**

* **Scenario A (Crush Multiplier):** You have a clean column with a Raw Score of 430\. The opponent has a score of 200\. Since 430 / 200 ≥ 2, you have a **2x** crush multiplier. Because you have a multiplier in your favor, your clean column bonus is **\+100**. Your Duel Points before the column coefficient would be: ((430 \+ 100\) \- 200\) × 2\.  
* **Scenario B (No Crush Multiplier):** You have a clean column with a Raw Score of 430\. The opponent has 250\. Since 430 / 250 \< 2, there is **no** crush multiplier (1x). Because there is no multiplier in your favor, your clean column bonus is **\+200**. Your Duel Points before the column coefficient would be: ((430 \+ 200\) \- 250\) × 1\.

### **Final Scoring**

You play 6 duels (one per column) against your single opponent.

* **Final Score:** The sum of all 6 Duel Points.  
* **Winner:** The player with a positive net score wins the game.