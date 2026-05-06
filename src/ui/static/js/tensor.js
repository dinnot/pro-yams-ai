// Tensor inspector — fetches the NN input tensor for the current board and
// renders it organized by structural groups (A–F) defined in src/engine/tensor.h.

const Tensor = {
    state: null,
    onlyNonzero: false,

    init() {
        document.getElementById('btn-show-tensor').addEventListener('click', () => Tensor.open());
        document.getElementById('btn-tensor-close').addEventListener('click', () => Tensor.close());
        document.getElementById('tensor-perspective').addEventListener('change', () => Tensor.refresh());
        document.getElementById('tensor-only-nonzero').addEventListener('change', (e) => {
            Tensor.onlyNonzero = e.target.checked;
            Tensor.render();
        });
        document.querySelector('#tensor-modal .tensor-modal-backdrop')
            .addEventListener('click', () => Tensor.close());
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') Tensor.close();
        });
    },

    async open() {
        if (!Game.sessionId) return;
        const cur = Game.state ? Game.state.current_player : 0;
        document.getElementById('tensor-perspective').value = String(cur ?? 0);
        document.getElementById('tensor-modal').style.display = 'block';
        await Tensor.refresh();
    },

    close() {
        document.getElementById('tensor-modal').style.display = 'none';
    },

    async refresh() {
        const sel = Number(document.getElementById('tensor-perspective').value);
        const data = await API.getTensor(Game.sessionId, sel);
        if (data.error) {
            document.getElementById('tensor-modal-body').innerHTML =
                `<div class="tensor-error">${data.error}</div>`;
            return;
        }
        Tensor.state = data;
        Tensor.render();
    },

    render() {
        const data = Tensor.state;
        if (!data) return;

        // Meta header.
        const meta = document.getElementById('tensor-meta');
        const nnPart = (data.nn_value !== undefined)
            ? ` · NN win prob (P${data.perspective}): <strong>${(data.nn_value * 100).toFixed(2)}%</strong>`
            : '';
        meta.innerHTML =
            `${data.size} features · perspective: Player ${data.perspective}${nnPart}`;

        // Body.
        const body = document.getElementById('tensor-modal-body');
        body.innerHTML = '';

        const renderers = {
            'A': Tensor.renderGroupA,
            'B': Tensor.renderGroupB,
            'C': Tensor.renderGroupC,
            'D': Tensor.renderGroupD,
            'E': Tensor.renderGroupE,
            'F': Tensor.renderGroupF,
        };

        for (const g of data.groups) {
            const sec = document.createElement('details');
            sec.className = 'tensor-group';
            sec.open = (g.name === 'A' || g.name === 'D');

            const summary = document.createElement('summary');
            summary.innerHTML =
                `<span class="tg-name">Group ${g.name}</span>` +
                `<span class="tg-range">[${g.start}…${g.start + g.size - 1}] · ${g.size} feats</span>` +
                `<span class="tg-desc">${g.description}</span>`;
            sec.appendChild(summary);

            const slice = data.values.slice(g.start, g.start + g.size);
            const renderer = renderers[g.name];
            const inner = renderer
                ? renderer(slice, data.perspective)
                : Tensor.renderRaw(slice, g.start);
            sec.appendChild(inner);

            body.appendChild(sec);
        }
    },

    // ---- value formatting helpers ----------------------------------------

    fmt(v) {
        if (v === 0) return '0';
        if (Math.abs(v) < 0.0005) return v.toExponential(1);
        return v.toFixed(3);
    },

    cellHtml(v, idx, label) {
        const isZero = Math.abs(v) < 1e-9;
        const cls = v > 0 ? 'tn-pos' : v < 0 ? 'tn-neg' : 'tn-zero';
        const hide = (Tensor.onlyNonzero && isZero) ? ' tn-hide' : '';
        const tt = (idx != null ? `idx ${idx}: ` : '') + (label || '') + ` = ${v}`;
        return `<span class="tn-cell ${cls}${hide}" title="${tt}">${isZero && Tensor.onlyNonzero ? '·' : Tensor.fmt(v)}</span>`;
    },

    perspectiveLabel(pi, persp) {
        // pi==0 is "me" (perspective player), pi==1 is opponent.
        const mePlayer = persp;
        const oppPlayer = 1 - persp;
        return pi === 0 ? `P${mePlayer} (me)` : `P${oppPlayer} (opp)`;
    },

    // ---- group renderers --------------------------------------------------

    // Group A: 312 = 2 players × 6 cols × 13 rows × 2 [present, normalized_value]
    renderGroupA(slice, persp) {
        const wrap = document.createElement('div');
        wrap.className = 'tg-body';
        let i = 0;

        for (let pi = 0; pi < 2; pi++) {
            const ph = document.createElement('div');
            ph.className = 'tg-subhead';
            ph.textContent = Tensor.perspectiveLabel(pi, persp) + ' — cell occupancy & normalized value';
            wrap.appendChild(ph);

            const tbl = document.createElement('table');
            tbl.className = 'tn-table';
            let html = '<thead><tr><th></th>';
            for (let c = 0; c < 6; c++) html += `<th colspan="2">${COLUMN_NAMES[c]}</th>`;
            html += '</tr><tr><th></th>';
            for (let c = 0; c < 6; c++) html += '<th>has</th><th>val</th>';
            html += '</tr></thead><tbody>';

            // Build per-row cells.
            const rowsHtml = [];
            for (let r = 0; r < 13; r++) rowsHtml.push(`<tr><th>${ROW_NAMES[r]}</th>`);

            for (let c = 0; c < 6; c++) {
                for (let r = 0; r < 13; r++) {
                    const present = slice[i++];
                    const value   = slice[i++];
                    const idx0 = (i - 2);
                    const idx1 = (i - 1);
                    rowsHtml[r] += `<td>${Tensor.cellHtml(present, idx0, `${COLUMN_NAMES[c]}/${ROW_NAMES[r]} present`)}</td>`;
                    rowsHtml[r] += `<td>${Tensor.cellHtml(value,   idx1, `${COLUMN_NAMES[c]}/${ROW_NAMES[r]} val`)}</td>`;
                }
            }
            for (let r = 0; r < 13; r++) rowsHtml[r] += '</tr>';

            html += rowsHtml.join('') + '</tbody>';
            tbl.innerHTML = html;
            wrap.appendChild(tbl);
        }
        return wrap;
    },

    // Group B: 108 = B.1 (24) + B.2 (84)
    renderGroupB(slice, persp) {
        const wrap = document.createElement('div');
        wrap.className = 'tg-body';
        let i = 0;

        // B.1: per-player x per-column [upper_sum/100, e_raw/500]
        const h1 = document.createElement('div');
        h1.className = 'tg-subhead';
        h1.textContent = 'B.1 — per-column upper_sum (norm) and expected raw score (norm)';
        wrap.appendChild(h1);

        const t1 = document.createElement('table');
        t1.className = 'tn-table';
        let html = '<thead><tr><th>Player</th><th>Feature</th>';
        for (let c = 0; c < 6; c++) html += `<th>${COLUMN_NAMES[c]}</th>`;
        html += '</tr></thead><tbody>';

        for (let pi = 0; pi < 2; pi++) {
            const usumRow = [];
            const erawRow = [];
            for (let c = 0; c < 6; c++) {
                usumRow.push(slice[i++]);
                erawRow.push(slice[i++]);
            }
            html += `<tr><th rowspan="2">${Tensor.perspectiveLabel(pi, persp)}</th>`;
            html += '<th>upper_sum</th>' +
                usumRow.map((v, c) => `<td>${Tensor.cellHtml(v, null, `${COLUMN_NAMES[c]} upper_sum`)}</td>`).join('') + '</tr>';
            html += '<tr><th>e_raw</th>' +
                erawRow.map((v, c) => `<td>${Tensor.cellHtml(v, null, `${COLUMN_NAMES[c]} e_raw`)}</td>`).join('') + '</tr>';
        }
        html += '</tbody>';
        t1.innerHTML = html;
        wrap.appendChild(t1);

        // B.2: per-column 14 features
        const labels = [
            'rem_me', 'rem_opp',
            'margin_now', 'margin_E',
            'd_crush_now', 'd_crush_E',
            'pts_2x_now', 'pts_3x_now', 'pts_4x_now', 'pts_5x_now',
            'pts_2x_E',   'pts_3x_E',   'pts_4x_E',   'pts_5x_E',
        ];
        const h2 = document.createElement('div');
        h2.className = 'tg-subhead';
        h2.textContent = 'B.2 — per-column duel/crush projections (my perspective)';
        wrap.appendChild(h2);

        const t2 = document.createElement('table');
        t2.className = 'tn-table';
        html = '<thead><tr><th>Feature</th>';
        for (let c = 0; c < 6; c++) html += `<th>${COLUMN_NAMES[c]}</th>`;
        html += '</tr></thead><tbody>';

        const cols = [];
        for (let c = 0; c < 6; c++) {
            const arr = [];
            for (let k = 0; k < 14; k++) arr.push(slice[i++]);
            cols.push(arr);
        }
        for (let k = 0; k < 14; k++) {
            html += `<tr><th>${labels[k]}</th>`;
            for (let c = 0; c < 6; c++) {
                html += `<td>${Tensor.cellHtml(cols[c][k], null, `${COLUMN_NAMES[c]} ${labels[k]}`)}</td>`;
            }
            html += '</tr>';
        }
        html += '</tbody>';
        t2.innerHTML = html;
        wrap.appendChild(t2);
        return wrap;
    },

    // Group C: 156 = 2 players × 6 cols × 13 rows  (1-turn non-scratch prob)
    renderGroupC(slice, persp) {
        const wrap = document.createElement('div');
        wrap.className = 'tg-body';
        let i = 0;
        for (let pi = 0; pi < 2; pi++) {
            const ph = document.createElement('div');
            ph.className = 'tg-subhead';
            ph.textContent = Tensor.perspectiveLabel(pi, persp) + ' — 1-turn non-scratch probability';
            wrap.appendChild(ph);

            const tbl = document.createElement('table');
            tbl.className = 'tn-table';
            let html = '<thead><tr><th></th>';
            for (let c = 0; c < 6; c++) html += `<th>${COLUMN_NAMES[c]}</th>`;
            html += '</tr></thead><tbody>';

            const block = [];
            for (let c = 0; c < 6; c++) {
                const colArr = [];
                for (let r = 0; r < 13; r++) colArr.push(slice[i++]);
                block.push(colArr);
            }
            for (let r = 0; r < 13; r++) {
                html += `<tr><th>${ROW_NAMES[r]}</th>`;
                for (let c = 0; c < 6; c++) {
                    html += `<td>${Tensor.cellHtml(block[c][r], null, `${COLUMN_NAMES[c]}/${ROW_NAMES[r]} P_one`)}</td>`;
                }
                html += '</tr>';
            }
            html += '</tbody>';
            tbl.innerHTML = html;
            wrap.appendChild(tbl);
        }
        return wrap;
    },

    // Group D: 14 globals
    renderGroupD(slice) {
        const wrap = document.createElement('div');
        wrap.className = 'tg-body';

        const h1 = document.createElement('div');
        h1.className = 'tg-subhead';
        h1.textContent = 'Column coefficients (normalized)';
        wrap.appendChild(h1);

        const t1 = document.createElement('table');
        t1.className = 'tn-table';
        let html = '<thead><tr>';
        for (let c = 0; c < 6; c++) html += `<th>${COLUMN_NAMES[c]}</th>`;
        html += '</tr></thead><tbody><tr>';
        for (let c = 0; c < 6; c++) html += `<td>${Tensor.cellHtml(slice[c], null, `${COLUMN_NAMES[c]} coeff`)}</td>`;
        html += '</tr></tbody>';
        t1.innerHTML = html;
        wrap.appendChild(t1);

        const labels = [
            'filled_frac', 'tanh(duel_now)', 'tanh(duel_E)',
            'dominance_my', 'dominance_opp',
            'phase>50', 'phase>100', 'phase>140',
        ];
        const h2 = document.createElement('div');
        h2.className = 'tg-subhead';
        h2.textContent = 'Global state and phase flags';
        wrap.appendChild(h2);

        const t2 = document.createElement('table');
        t2.className = 'tn-table';
        html = '<tbody>';
        for (let k = 0; k < labels.length; k++) {
            html += `<tr><th>${labels[k]}</th><td>${Tensor.cellHtml(slice[6 + k], null, labels[k])}</td></tr>`;
        }
        html += '</tbody>';
        t2.innerHTML = html;
        wrap.appendChild(t2);
        return wrap;
    },

    // Group E: 216 = 2 × 6 × 18 (3 horizons × {P60,P70,P80,P90,P100,EU/100})
    renderGroupE(slice, persp) {
        return Tensor.renderHorizonGroup(slice, persp,
            ['P60', 'P70', 'P80', 'P90', 'P100', 'EU'],
            'Upper section: target-bonus probabilities and expected upper score');
    },

    // Group F: 180 = 2 × 6 × 15 (3 horizons × {P_mid,EV_mid,P_low,EV_low,P_clean})
    renderGroupF(slice, persp) {
        return Tensor.renderHorizonGroup(slice, persp,
            ['P_mid', 'EV_mid', 'P_low', 'EV_low', 'P_clean'],
            'Middle/lower probabilities, EV, and clean-column probability');
    },

    renderHorizonGroup(slice, persp, featLabels, headline) {
        const wrap = document.createElement('div');
        wrap.className = 'tg-body';
        const horizons = ['T_min', 'T_mid', 'T_max'];
        const featsPerHorizon = featLabels.length;
        const featsPerCol = 3 * featsPerHorizon;
        let i = 0;

        const h0 = document.createElement('div');
        h0.className = 'tg-subhead';
        h0.textContent = headline;
        wrap.appendChild(h0);

        for (let pi = 0; pi < 2; pi++) {
            const ph = document.createElement('div');
            ph.className = 'tg-subhead tg-subhead-sm';
            ph.textContent = Tensor.perspectiveLabel(pi, persp);
            wrap.appendChild(ph);

            const tbl = document.createElement('table');
            tbl.className = 'tn-table';
            let html = '<thead><tr><th rowspan="2">Column</th>';
            for (const h of horizons) html += `<th colspan="${featsPerHorizon}">${h}</th>`;
            html += '</tr><tr>';
            for (let h = 0; h < 3; h++) {
                for (const f of featLabels) html += `<th>${f}</th>`;
            }
            html += '</tr></thead><tbody>';

            for (let c = 0; c < 6; c++) {
                html += `<tr><th>${COLUMN_NAMES[c]}</th>`;
                for (let k = 0; k < featsPerCol; k++) {
                    const v = slice[i++];
                    const horizon = horizons[Math.floor(k / featsPerHorizon)];
                    const feat    = featLabels[k % featsPerHorizon];
                    html += `<td>${Tensor.cellHtml(v, null, `${COLUMN_NAMES[c]} ${horizon}/${feat}`)}</td>`;
                }
                html += '</tr>';
            }
            html += '</tbody>';
            tbl.innerHTML = html;
            wrap.appendChild(tbl);
        }
        return wrap;
    },

    // Fallback: dense raw value grid
    renderRaw(slice, baseIdx) {
        const wrap = document.createElement('div');
        wrap.className = 'tg-body tg-raw';
        let html = '';
        for (let i = 0; i < slice.length; i++) {
            html += Tensor.cellHtml(slice[i], baseIdx + i, `feature ${baseIdx + i}`);
        }
        wrap.innerHTML = html;
        return wrap;
    },
};
