# **Competitive Yams (Pro Yams) \- Game Rules**

"Competitive Yams" is a high-stakes, cutthroat variation of Yahtzee (Yams) designed for 1v1, 1v1v1, or 2v2 play. It introduces strict column constraints, a "King of the Hill" scoring requirement, and an aggressive dueling system.

## **1\. The Basics**

* **Players:** 2 (1v1), 3 (1v1v1), or 4 (2v2).  
* **Dice:** 5 standard six-sided dice.  
* **Turns:** On your turn, you have up to **3 rolls**(one initial roll and 2 re-rolls). You may hold any number of dice between rolls.  
* **Objective:** Score the most points by winning "Duels" on specific columns against your neighbors.

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

* **Competitive Scoring (The Golden Rule)**: To validate a score in *any* box, your score must be **greater than or equal to** the highest score currently recorded in that specific box by **anyone** in the game (teammate or opponent).  
  * *Example:* If your opponent scored 24 in their *Free* column's "6" box, you must score at least 24 (four 6s) to fill your "6" box in the *Free* column. If you only roll three 6s (18), you score **0**.  
* **SS / LS Relationship**:  
  * **SS (Small Sum)**: Must be at least 20\. Must be strictly lower than your **LS** (if LS is already filled). If you scratched (0) your LS in this column, you must scratch SS as well (cannot put any score \> 0 in there). If you scratch LS after putting a score in SS, SS will become scratched as well.  
  * **LS (Large Sum)**: Must be at least 20\. Must be strictly higher than the highest **SS** recorded by anyone. If you scratched (0) your SS in this column, you must scratch LS as well (cannot put any score \> 0 in there). If you scratch SS after putting a score in LS, LS will become scratched as well.  
  * 

#### **"Sec" (Dry) Bonus**

If you roll a **Yams** (5 identical dice) on your **very first roll** of a turn, you trigger the "Sec" bonus. This acts as a Joker, allowing you to fill **any** available cell in the column you are playing with its **maximum possible score**:

1. **Numbers 1-6:** 5 × \[Number\] (e.g., 25 for 5s).  
2. **SS/LS:** Maximum possible sum (typically 29/30).  
3. **Full House:** 50 points.  
4. **4 of a Kind:** 54 points.  
5. **Straight:** 50 points.  
6. **Less than 8:** 75 points.  
7. **Yams:** 100 points.

### **The Columns**

Each column determines **how** you are allowed to fill the rows above.

**Coefficients:** At the start of the game, each of the 6 columns is randomly assigned a unique **Coefficient** (Multiplier) from the following set: **8, 10, 12, 14, 16, 18**. This coefficient multiplies the point difference won in that column, making high-coefficient columns the primary strategic targets.

* **Down**: Must be filled strictly from top to bottom (starts at 1, ends at Yams).  
* **Free**: Can be filled in any order.  
* **Up**: Must be filled strictly from bottom to top (starts at Yams, ends at 1).  
* **Mid**: Must start from the middle (6 or Small Sum) and fill adjacent cells outwards (up or down).  
* **x2 (Turbo)**: Can be filled in any order, but you are limited to **2 rolls** maximum instead of 3\.  
* **Up/Down (Connected)**: You may place your first score anywhere in this column. All subsequent scores in this column must be placed immediately above or below an existing score (adjacent).

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

If you complete a column without scratching any box (no 0s), you receive a **Virtual Bonus** of \+200 points for that column.

*Important:* This bonus is added to your score difference, but it is **NOT** included in the "Raw Score" used to determine the Crush Multiplier (see below).

## **4\. Winning the Game (The Duel)**

The game is not won by raw total score, but by winning **Duels** on each column against your immediate neighbors.

### **The Duel Calculation**

For every column, you perform a comparison against an opponent:

1. **Calculate Raw Scores**: Sum of boxes \+ Upper Bonus (Rows 1-6). *Do not include Clean Column Bonus yet.*  
2. **Determine Crush Multiplier**: Compare your Raw Score to your opponent's Raw Score.  
   * If YourRaw ≥ 2 × OpponentRaw → **2x**  
   * If YourRaw ≥ 3 × OpponentRaw → **3x**  
   * ...up to **5x**  
   * *(Note: If OpponentRaw is 0, multiplier is 5x).*  
3. Calculate Final Difference: Diff \= (YourRaw \- OpponentRaw)  
   If you have the Clean Column Bonus, add 200 to this difference.  
4. **Apply Coefficient & Multiplier**: DuelPoints \= Diff × ColumnCoefficient × CrushMultiplier

### **Game Modes**

#### **1v1 Scoring**

You play 6 duels (one per column) against your single opponent.

* **Final Score:** Sum of all 6 Duel Points.  
* **Winner:** Positive score wins.

#### **1v1v1 Scoring (Triple Threat)**

Players effectively sit in a triangle. You play duels against **both** of your neighbors (Left and Right).

* For every column, you calculate DuelPoints against your Left Neighbor and separate DuelPoints against your Right Neighbor.  
* **Final Score:** Sum of all duels against both neighbors.  
* **Winner:** Player with the highest total score.

#### **2v2 Scoring**

Players sit in a circle (A and C are team 1; B and D are team 2).

* You calculate duels against **both** neighbors (Left and Right), who are always members of the opposing team.  
* **Team Score:** Sum of Player A's total \+ Player C's total.  
* **Winner:** Team with the highest total score.