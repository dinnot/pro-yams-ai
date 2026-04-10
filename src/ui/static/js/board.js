// Board rendering for Pro Yams UI.

const COLUMN_NAMES = ['Down', 'Free', 'Up', 'Mid', 'Turbo', 'UpDown'];
const COLUMN_KEYS = ['down', 'free', 'up', 'mid', 'turbo', 'updown'];
const ROW_NAMES = ['1s', '2s', '3s', '4s', '5s', '6s', 'SS', 'LS', 'FH', 'K', 'STR', 'U8', 'Y'];

const Board = {
    // Render a player's board into a grid container.
    // boardData: object like { "down": { "1s": -1, "2s": 4, ... }, ... }
    // coefficients: array of 6 integers
    // legalPlacements: array of { column, row, score } (only for human turns)
    // onCellClick: callback(column, row) for clicking legal cells
    render(containerId, boardData, coefficients, legalPlacements, onCellClick) {
        const container = document.getElementById(containerId);
        if (!container) return;
        container.innerHTML = '';

        // Build a set of legal (col, row) for quick lookup.
        const legalSet = new Set();
        const legalScores = {};
        if (legalPlacements) {
            for (const p of legalPlacements) {
                const key = `${p.column},${p.row}`;
                legalSet.add(key);
                legalScores[key] = p.score;
            }
        }

        // Top-left empty cell.
        const corner = document.createElement('div');
        corner.className = 'row-label';
        container.appendChild(corner);

        // Column headers with coefficient on a second line.
        for (let c = 0; c < 6; c++) {
            const hdr = document.createElement('div');
            hdr.className = 'col-header';
            if (coefficients) {
                hdr.innerHTML = `${COLUMN_NAMES[c]}<br><span class="col-coeff">x${coefficients[c]}</span>`;
            } else {
                hdr.textContent = COLUMN_NAMES[c];
            }
            container.appendChild(hdr);
        }

        // Rows.
        for (let r = 0; r < 13; r++) {
            // Row label.
            const label = document.createElement('div');
            label.className = 'row-label';
            label.textContent = ROW_NAMES[r];
            container.appendChild(label);

            // Cells.
            for (let c = 0; c < 6; c++) {
                const cell = document.createElement('div');
                cell.className = 'board-cell';
                const colKey = COLUMN_KEYS[c];
                const rowKey = ROW_NAMES[r];
                const value = boardData && boardData[colKey] ? boardData[colKey][rowKey] : -1;

                if (value === -1) {
                    cell.classList.add('empty');
                    const key = `${c},${r}`;
                    if (legalSet.has(key)) {
                        cell.classList.remove('empty');
                        cell.classList.add('legal');
                        const score = legalScores[key];
                        cell.textContent = score !== undefined ? score : '';
                        cell.title = `${COLUMN_NAMES[c]} / ${ROW_NAMES[r]}: score ${score}`;
                        cell.addEventListener('click', () => {
                            if (onCellClick) onCellClick(c, r);
                        });
                    } else {
                        cell.textContent = '';
                    }
                } else if (value === 0) {
                    cell.classList.add('scratched');
                    cell.textContent = '0';
                } else {
                    cell.classList.add('scored');
                    cell.textContent = value;
                }

                container.appendChild(cell);
            }

            // Upper section bonus row after row 6 (index 5).
            if (r === 5) {
                const bonusLabel = document.createElement('div');
                bonusLabel.className = 'row-label';
                bonusLabel.textContent = 'Bonus';
                bonusLabel.style.color = '#ffc107';
                bonusLabel.style.fontSize = '0.7rem';
                container.appendChild(bonusLabel);

                for (let c = 0; c < 6; c++) {
                    const bonusCell = document.createElement('div');
                    bonusCell.className = 'board-cell';
                    bonusCell.style.background = '#1a1a1a';
                    bonusCell.style.fontSize = '0.7rem';
                    bonusCell.style.color = '#ffc107';
                    const colKey = COLUMN_KEYS[c];

                    // Compute upper sum.
                    let upperSum = 0;
                    let allFilled = true;
                    for (let ur = 0; ur < 6; ur++) {
                        const v = boardData && boardData[colKey] ? boardData[colKey][ROW_NAMES[ur]] : -1;
                        if (v === -1) { allFilled = false; }
                        if (v > 0) upperSum += v;
                    }

                    const bonus = Board.computeBonus(upperSum);
                    if (allFilled) {
                        bonusCell.textContent = `${upperSum}/${bonus}`;
                    } else {
                        bonusCell.textContent = `${upperSum}`;
                    }
                    container.appendChild(bonusCell);
                }
            }
        }
    },

    computeBonus(upperSum) {
        if (upperSum >= 100) return 500;
        if (upperSum >= 90) return 200;
        if (upperSum >= 80) return 100;
        if (upperSum >= 70) return 50;
        if (upperSum >= 60) return 30;
        return 0;
    }
};
