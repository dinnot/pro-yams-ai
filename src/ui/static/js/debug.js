// Debug panel rendering for Pro Yams UI.
// Shows hold and placement evaluations captured from bot turns.

const Debug = {
    /// Build a collapsed debug panel DOM element for one turn's debug data.
    /// debugData: { hold_evals: [[{mask, held_flags, expected_value},...], ...],
    ///              placement_evals: [{column, row, column_name, row_name, score, eval_value},...] }
    buildPanel(debugData) {
        const panel = document.createElement('div');
        panel.className = 'debug-panel';

        // --- Hold evaluations ---
        if (debugData.hold_evals && debugData.hold_evals.length > 0) {
            debugData.hold_evals.forEach((stepCandidates, stepIdx) => {
                const stepTitle = document.createElement('div');
                stepTitle.className = 'debug-section-title';
                stepTitle.textContent = `Reroll ${stepIdx + 1} — hold candidates (top 8)`;
                panel.appendChild(stepTitle);

                const tbl = Debug.buildTable(
                    ['Rank', 'Hold (dice)', 'Expected Value'],
                    stepCandidates.slice(0, 8).map((cand, rank) => [
                        rank + 1,
                        Debug.renderHoldMask(cand.held_flags),
                        cand.expected_value.toFixed(4)
                    ]),
                    0  // highlight rank-1 row (index 0)
                );
                panel.appendChild(tbl);
            });
        }

        // --- Placement evaluations ---
        if (debugData.placement_evals && debugData.placement_evals.length > 0) {
            const plTitle = document.createElement('div');
            plTitle.className = 'debug-section-title';
            plTitle.textContent = `Placement candidates (top 10)`;
            panel.appendChild(plTitle);

            const tbl = Debug.buildTable(
                ['Rank', 'Column', 'Row', 'Score', 'Bot Eval'],
                debugData.placement_evals.slice(0, 10).map((cand, rank) => [
                    rank + 1,
                    cand.column_name || cand.column,
                    cand.row_name    || cand.row,
                    cand.score,
                    cand.eval_value.toFixed(4)
                ]),
                0  // highlight rank-1 row
            );
            panel.appendChild(tbl);
        }

        return panel;
    },

    /// Render a hold mask as a 5-die string: held dice are shown in a highlight span.
    /// held_flags: array of 5 booleans.
    renderHoldMask(held_flags) {
        if (!held_flags) return '?????';
        return held_flags.map((held, i) =>
            held
                ? `<span class="debug-held-die">h${i + 1}</span>`
                : `<span class="debug-free-die">_${i + 1}</span>`
        ).join(' ');
    },

    /// Build a simple HTML table.
    /// headers: string[]
    /// rows: (string|number)[][]
    /// highlightRow: index of row to highlight (or -1 for none)
    buildTable(headers, rows, highlightRow) {
        const tbl = document.createElement('table');
        tbl.className = 'debug-table';

        const thead = tbl.createTHead();
        const hr = thead.insertRow();
        for (const h of headers) {
            const th = document.createElement('th');
            th.textContent = h;
            hr.appendChild(th);
        }

        const tbody = tbl.createTBody();
        rows.forEach((row, ri) => {
            const tr = tbody.insertRow();
            if (ri === highlightRow) tr.classList.add('debug-best-row');
            for (const cell of row) {
                const td = tr.insertCell();
                if (typeof cell === 'string' && cell.includes('<span')) {
                    td.innerHTML = cell;
                } else {
                    td.textContent = String(cell);
                }
            }
        });

        return tbl;
    }
};
