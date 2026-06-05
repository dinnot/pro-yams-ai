// Game view logic for Pro Yams UI.

const Game = {
    sessionId: null,
    state: null,
    options: null,
    holdMask: 0,
    // Boards the current session was loaded from (PY1 position), or null for a
    // fresh game. copy-position replays history on top of this.
    basePosition: null,
    autoPlayTimer: null,
    legalCollapsed: true,
    serverNumPlayers: 2,
    serverVariant: '1v1',

    async init() {
        document.getElementById('btn-new-game').addEventListener('click', () => Game.newGame());
        document.getElementById('btn-load-position').addEventListener('click', () => Game.loadPosition());
        document.getElementById('btn-step').addEventListener('click', () => Game.step());
        document.getElementById('btn-play-all').addEventListener('click', () => Game.playAll());
        document.getElementById('btn-reroll').addEventListener('click', () => Game.reroll());
        document.getElementById('legal-placements-toggle').addEventListener('click', () => {
            Game.legalCollapsed = !Game.legalCollapsed;
            Game.applyLegalCollapse();
        });

        // The server's variant is fixed at launch — query once so we know
        // whether to expose the P2/P3 dropdowns.
        try {
            const info = await API.getInfo();
            Game.serverNumPlayers = info.num_players || 2;
            Game.serverVariant    = info.game_variant || '1v1';
        } catch (_) { /* leave defaults */ }
        document.body.dataset.variant = Game.serverVariant;
        const show2v2 = Game.serverNumPlayers >= 4;
        for (const el of document.querySelectorAll('.sel-2v2-only')) {
            el.hidden = !show2v2;
        }
    },

    // Ensure board-pN / board-grid-N divs exist for each player. The HTML
    // ships with P0/P1; P2/P3 are created on demand when the JSON reports
    // a 2v2 game. Also toggles the boards-area layout via data-variant.
    ensureBoardContainers(numPlayers, variant) {
        const area = document.querySelector('.boards-area');
        if (!area) return;
        area.dataset.variant = variant;

        // Team assignment in 2v2: P0/P2 are Team 0, P1/P3 are Team 1.
        const teamOf = (p) => (p & 1) === 0 ? 0 : 1;

        for (let p = 0; p < numPlayers; ++p) {
            let container = document.getElementById(`board-p${p}`);
            if (!container) {
                container = document.createElement('div');
                container.id = `board-p${p}`;
                container.className = 'board-container';
                container.innerHTML =
                    `<h3>Player ${p} <span class="player-label" id="p${p}-type"></span></h3>` +
                    `<div id="board-grid-${p}" class="board-grid"></div>`;
                area.appendChild(container);
            }
            // Update team class for 2x2 shading.
            container.classList.remove('team-0', 'team-1');
            if (variant === '2v2') {
                container.classList.add(`team-${teamOf(p)}`);
            }
        }

        // Hide any leftover containers from a previous larger game.
        for (let p = numPlayers; p < 8; ++p) {
            const c = document.getElementById(`board-p${p}`);
            if (c) c.style.display = 'none';
        }
        for (let p = 0; p < numPlayers; ++p) {
            const c = document.getElementById(`board-p${p}`);
            if (c) c.style.display = '';
        }
    },

    async newGame() {
        Game.stopAutoPlay();
        Game.basePosition = null;
        const debugMode = document.getElementById('chk-debug').checked;
        const data = await API.newGame(Game.collectPlayerTypes(), undefined, debugMode);
        Game.sessionId = data.session_id;
        Game.state = data.game_state;
        Game.holdMask = 0;
        Game.updateUI();

        // Enable controls.
        document.getElementById('btn-step').disabled = false;
        document.getElementById('btn-play-all').disabled = false;
        document.getElementById('btn-show-tensor').disabled = false;

        // If first player is a bot, auto-advance until human or end.
        if (!Game.state.waiting_for_human && !Game.state.game_over) {
            // Leave it for user to click Step or Play All.
        }
    },

    // Player types for every seat the server expects (P2/P3 dropdowns exist
    // but are .hidden in 1v1). Shared by newGame() and loadPosition().
    collectPlayerTypes() {
        const playerTypes = [];
        for (let p = 0; p < Game.serverNumPlayers; p++) {
            const sel = document.getElementById(`sel-p${p}`);
            playerTypes.push(sel ? sel.value : 'heuristic_v2');
        }
        return playerTypes;
    },

    // Copy the board position after history turn `index` to the clipboard as a
    // PY1 notation string. `el` is the clicked icon, flashed ✓ on success.
    async copyPositionAt(index, el) {
        const gs = Game.state;
        if (!gs || !gs.history || !gs.history[index]) return;
        const variant = gs.game_variant ||
            ((gs.num_players === 4) ? '2v2' : '1v1');
        const pos = Position.fromHistory(gs.history, gs.coefficients, variant,
                                         index, Game.basePosition);
        const ok = await Position.copyToClipboard(Position.encode(pos));
        if (el) {
            const prev = el.textContent;
            el.textContent = ok ? '✓' : '✗';
            setTimeout(() => { el.textContent = prev; }, 900);
        }
    },

    // Start a new game from a pasted PY1 position string. Reuses the current
    // player-type selectors and debug toggle; dice are rolled fresh for the
    // player on move.
    async loadPosition() {
        const input = document.getElementById('position-input');
        const errEl = document.getElementById('position-error');
        const raw = (input.value || '').trim();
        const showErr = (msg) => { if (errEl) { errEl.textContent = msg; errEl.style.display = msg ? '' : 'none'; } };
        showErr('');
        if (!raw) { showErr('Paste a position string first.'); return; }

        let pos;
        try { pos = Position.decode(raw); }
        catch (e) { showErr('Invalid position: ' + e.message); return; }

        if (pos.variant !== Game.serverVariant) {
            showErr(`Position is ${pos.variant} but this server runs ${Game.serverVariant}.`);
            return;
        }

        Game.stopAutoPlay();
        const debugMode = document.getElementById('chk-debug').checked;
        const data = await API.newGame(Game.collectPlayerTypes(), undefined, debugMode, pos);
        if (data.error) { showErr(data.error); return; }
        // Remember the loaded board so copy-position can replay history on top
        // of it rather than starting from an empty board.
        Game.basePosition = pos.boards;
        Game.sessionId = data.session_id;
        Game.state = data.game_state;
        Game.holdMask = 0;
        Game.updateUI();

        document.getElementById('btn-step').disabled = false;
        document.getElementById('btn-play-all').disabled = false;
        document.getElementById('btn-show-tensor').disabled = false;
    },

    async step() {
        if (!Game.sessionId) return;
        if (Game.state && Game.state.game_over) return;

        const data = await API.step(Game.sessionId);
        if (data.error) { Game.setStatus(data.error); return; }
        Game.state = data;
        Game.holdMask = 0;
        Game.updateUI();
    },

    async playAll() {
        if (!Game.sessionId) return;
        const data = await API.playAll(Game.sessionId);
        if (data.error) { Game.setStatus(data.error); return; }
        Game.state = data;
        Game.holdMask = 0;
        Game.updateUI();
    },

    async reroll() {
        if (!Game.sessionId || !Game.state || !Game.state.waiting_for_human) return;
        const prevDice = Game.state.dice ? Game.state.dice.slice() : [];
        const prevMask = Game.holdMask;
        const data = await API.hold(Game.sessionId, prevMask);
        if (data.error) { Game.setStatus(data.error); return; }
        Game.state = data;
        Game.holdMask = Game.remapHoldMask(prevDice, prevMask, data.dice || []);
        Game.updateUI();
    },

    // After a reroll the dice are re-sorted, so bit positions shift.
    // Rebuild the holdMask by matching held values into the new sorted array.
    remapHoldMask(prevDice, prevMask, newDice) {
        const heldValues = prevDice.filter((_, i) => (prevMask >> i) & 1).sort((a, b) => a - b);
        if (heldValues.length === 0) return 0;
        let hi = 0;
        let newMask = 0;
        for (let i = 0; i < newDice.length && hi < heldValues.length; i++) {
            if (newDice[i] === heldValues[hi]) {
                newMask |= (1 << i);
                hi++;
            }
        }
        return newMask;
    },

    async placeScore(column, row) {
        if (!Game.sessionId || !Game.state || !Game.state.waiting_for_human) return;
        const data = await API.place(Game.sessionId, column, row);
        if (data.error) { Game.setStatus(data.error); return; }
        Game.state = data;
        Game.holdMask = 0;
        Game.options = null;
        Game.updateUI();
    },

    stopAutoPlay() {
        if (Game.autoPlayTimer) {
            clearInterval(Game.autoPlayTimer);
            Game.autoPlayTimer = null;
        }
    },

    // Mirrors compute_duel<Traits>() from duel.cc. Iterates every cross-team
    // pairing (1 in 1v1, 4 in 2v2). totalDuel is Team-0 minus Team-1 margin.
    computeDuel(gs) {
        if (!gs.boards || !gs.coefficients) return null;

        const upperBonus = (sum) => {
            if (sum >= 100) return 500;
            if (sum >= 90)  return 200;
            if (sum >= 80)  return 100;
            if (sum >= 70)  return 50;
            if (sum >= 60)  return 30;
            return 0;
        };
        const crushMult = (my, opp) => {
            if (opp === 0 && my > 0) return 5;
            if (opp > 0 && my >= 5 * opp) return 5;
            if (opp > 0 && my >= 4 * opp) return 4;
            if (opp > 0 && my >= 3 * opp) return 3;
            if (opp > 0 && my >= 2 * opp) return 2;
            return 1;
        };

        const numPlayers = gs.num_players ||
            (gs.player_types ? gs.player_types.length : 2);
        const team0 = numPlayers === 4 ? [0, 2] : [0];
        const team1 = numPlayers === 4 ? [1, 3] : [1];
        const UPPER = ROW_NAMES.slice(0, 6);
        const LOWER = ROW_NAMES.slice(6);
        const coeffs = gs.coefficients;

        let totalDuel = 0;
        const colData = [];

        for (let c = 0; c < 6; c++) {
            const colKey = COLUMN_KEYS[c];
            const coeff  = coeffs[c];

            const players = [];
            for (let p = 0; p < numPlayers; p++) {
                const b = gs.boards[`player${p}`];
                let upperSum = 0, cellsSum = 0, lowerHasScratch = false;
                for (const rn of UPPER) {
                    const v = b?.[colKey]?.[rn] ?? -1;
                    if (v > 0) { upperSum += v; cellsSum += v; }
                }
                for (const rn of LOWER) {
                    const v = b?.[colKey]?.[rn] ?? -1;
                    if (v === 0) lowerHasScratch = true;
                    if (v > 0) cellsSum += v;
                }
                const uBonus = upperBonus(upperSum);
                players.push({
                    seat: p,
                    upperSum, cellsSum, uBonus,
                    raw: cellsSum + uBonus,
                    lowerHasScratch,
                    isClean: upperSum >= 60 && !lowerHasScratch,
                    cBonus: 0,
                    adjusted: cellsSum + uBonus,
                });
            }

            const pairings = [];
            let colDuel = 0;
            let maxCrush = 1;
            for (const t0p of team0) {
                for (const t1p of team1) {
                    const a = players[t0p], b = players[t1p];
                    const crush0 = crushMult(a.raw, b.raw);
                    const crush1 = crushMult(b.raw, a.raw);
                    const activeCrush = Math.max(crush0, crush1);
                    if (activeCrush > maxCrush) maxCrush = activeCrush;
                    const cleanBonus = activeCrush > 1 ? 100 : 200;
                    const adjA = a.raw + (a.isClean ? cleanBonus : 0);
                    const adjB = b.raw + (b.isClean ? cleanBonus : 0);
                    const duelPts = (adjA - adjB) * activeCrush * coeff;
                    colDuel += duelPts;
                    pairings.push({ t0p, t1p, activeCrush, cleanBonus,
                                    adjA, adjB, duelPts });
                    if (a.isClean && cleanBonus > a.cBonus) {
                        a.cBonus = cleanBonus; a.adjusted = a.raw + cleanBonus;
                    }
                    if (b.isClean && cleanBonus > b.cBonus) {
                        b.cBonus = cleanBonus; b.adjusted = b.raw + cleanBonus;
                    }
                }
            }

            totalDuel += colDuel;
            colData.push({ c, colKey, coeff, players, pairings,
                           activeCrush: maxCrush, duelPts: colDuel });
        }

        return { totalDuel, colData };
    },

    renderScoreBreakdown(gs) {
        const panel = document.getElementById('score-breakdown');
        if (!gs.game_over || !gs.boards || !gs.coefficients) {
            panel.style.display = 'none';
            return null;
        }

        const { totalDuel, colData } = Game.computeDuel(gs);
        panel.style.display = 'block';

        const numPlayers = gs.num_players ||
            (gs.player_types ? gs.player_types.length : 2);
        const fmt = (n) => n > 0 ? `+${n}` : String(n);
        const bonusCell = (v) => v > 0 ? `<span class="bonus-val">+${v}</span>` : '<span style="color:#444">—</span>';

        let html;
        if (numPlayers === 4) {
            html = Game.renderBreakdown2v2(colData, totalDuel, fmt);
        } else {
            // 1v1: one row per column with full per-player stat columns.
            let header1 = `<tr><th rowspan="2">Column</th>`;
            for (let p = 0; p < numPlayers; p++) {
                header1 += `<th colspan="4" style="border-bottom:1px solid #0f3460">Player ${p}</th>`;
            }
            header1 += `<th rowspan="2">Crush</th><th rowspan="2">Duel pts</th></tr>`;
            let header2 = `<tr>`;
            for (let p = 0; p < numPlayers; p++) {
                header2 += `<th>cells</th><th>+upper</th><th>+clean</th><th>adj</th>`;
            }
            header2 += `</tr>`;

            html = `<h4>Score Breakdown</h4>
            <table class="score-table">
                <thead>${header1}${header2}</thead>
                <tbody>`;

            for (const col of colData) {
                const ptsCls = col.duelPts > 0 ? 'pts-pos' : col.duelPts < 0 ? 'pts-neg' : '';
                let row = `<tr><td class="col-name">${COLUMN_NAMES[col.c]}<span class="col-mult"> ×${col.coeff}</span></td>`;
                for (const pl of col.players) {
                    row += `<td>${pl.cellsSum}</td><td>${bonusCell(pl.uBonus)}</td>` +
                           `<td>${bonusCell(pl.cBonus)}</td><td class="adj-val">${pl.adjusted}</td>`;
                }
                row += `<td class="crush-val">×${col.activeCrush}</td>` +
                       `<td class="${ptsCls}">${fmt(col.duelPts)}</td></tr>`;
                html += row;
            }

            const totalCls = totalDuel > 0 ? 'pts-pos' : totalDuel < 0 ? 'pts-neg' : '';
            const totalColSpan = 1 + 4 * numPlayers + 1;
            html += `</tbody>
                <tfoot>
                    <tr class="score-total">
                        <td colspan="${totalColSpan}" style="text-align:right;color:#aaa">Total duel score (T0 − T1)</td>
                        <td class="${totalCls}">${fmt(totalDuel)}</td>
                    </tr>
                </tfoot>
            </table>`;
        }

        panel.innerHTML = html;
        return totalDuel;
    },

    renderBreakdown2v2(colData, totalDuel, fmt) {
        // Accumulate per-player signed contribution: T0 players gain, T1 players lose.
        const playerTotals = [0, 0, 0, 0];
        for (const col of colData) {
            for (const pr of col.pairings) {
                playerTotals[pr.t0p] += pr.duelPts;
                playerTotals[pr.t1p] -= pr.duelPts;
            }
        }

        // 8 columns: Column | T0 | adj | vs | T1 | adj | ×crush | pts
        let html = `<h4>Score Breakdown</h4>
        <table class="score-table">
            <thead>
              <tr>
                <th>Column</th>
                <th>T0</th><th>adj</th>
                <th></th>
                <th>T1</th><th>adj</th>
                <th>×crush</th><th>pts</th>
              </tr>
            </thead>
            <tbody>`;

        for (const col of colData) {
            const nPr = col.pairings.length;
            for (let i = 0; i < nPr; i++) {
                const pr = col.pairings[i];
                const ptsCls = pr.duelPts > 0 ? 'pts-pos' : pr.duelPts < 0 ? 'pts-neg' : '';
                html += `<tr>`;
                if (i === 0) {
                    html += `<td rowspan="${nPr + 1}" class="col-name">${COLUMN_NAMES[col.c]}<span class="col-mult"> ×${col.coeff}</span></td>`;
                }
                html += `<td>P${pr.t0p}</td><td class="adj-val">${pr.adjA}</td>
                         <td style="color:#555">vs</td>
                         <td>P${pr.t1p}</td><td class="adj-val">${pr.adjB}</td>
                         <td class="crush-val">×${pr.activeCrush}</td>
                         <td class="${ptsCls}">${fmt(pr.duelPts)}</td>
                       </tr>`;
            }
            // Subtotal row — shares the rowspan column cell, so 7 cells here.
            const colCls = col.duelPts > 0 ? 'pts-pos' : col.duelPts < 0 ? 'pts-neg' : '';
            html += `<tr class="col-subtotal">
                       <td colspan="6" style="text-align:right;color:#aaa">Column total (T0 − T1)</td>
                       <td class="${colCls}">${fmt(col.duelPts)}</td>
                     </tr>`;
        }

        html += `</tbody><tfoot>`;

        // Team 0 per-player then sum — 8 cells per row in tfoot.
        for (const p of [0, 2]) {
            const cls = playerTotals[p] > 0 ? 'pts-pos' : playerTotals[p] < 0 ? 'pts-neg' : '';
            html += `<tr>
                       <td colspan="7" style="text-align:right;color:#aaa">P${p} total</td>
                       <td class="${cls}">${fmt(playerTotals[p])}</td>
                     </tr>`;
        }
        const t0 = playerTotals[0] + playerTotals[2];
        const t0Cls = t0 > 0 ? 'pts-pos' : t0 < 0 ? 'pts-neg' : '';
        html += `<tr class="score-total">
                   <td colspan="7" style="text-align:right">Team 0 (P0 + P2)</td>
                   <td class="${t0Cls}">${fmt(t0)}</td>
                 </tr>`;

        // Team 1 per-player then sum.
        for (const p of [1, 3]) {
            const cls = playerTotals[p] > 0 ? 'pts-pos' : playerTotals[p] < 0 ? 'pts-neg' : '';
            html += `<tr>
                       <td colspan="7" style="text-align:right;color:#aaa">P${p} total</td>
                       <td class="${cls}">${fmt(playerTotals[p])}</td>
                     </tr>`;
        }
        const t1 = playerTotals[1] + playerTotals[3];
        const t1Cls = t1 > 0 ? 'pts-pos' : t1 < 0 ? 'pts-neg' : '';
        html += `<tr class="score-total">
                   <td colspan="7" style="text-align:right">Team 1 (P1 + P3)</td>
                   <td class="${t1Cls}">${fmt(t1)}</td>
                 </tr>`;

        html += `</tfoot></table>`;
        return html;
    },

    async updateUI() {
        if (!Game.state) return;
        const gs = Game.state;

        // Score breakdown (shows/hides itself based on game_over).
        const duel = Game.renderScoreBreakdown(gs);

        // Status bar.
        if (gs.game_over) {
            const scoreStr = duel != null ? ` (duel score: ${duel > 0 ? '+' + duel : duel})` : '';
            const r = gs.result;
            const numP = gs.num_players || (gs.player_types ? gs.player_types.length : 2);
            if (r === 1.0) {
                const winners = numP === 4 ? 'P0 and P2' : 'Player 0';
                Game.setStatus(`Game over — ${winners} win!${scoreStr}`, 'win');
            } else if (r === 0.0) {
                const winners = numP === 4 ? 'P1 and P3' : 'Player 1';
                Game.setStatus(`Game over — ${winners} win!${scoreStr}`, 'win');
            } else {
                Game.setStatus(`Game over — Draw!${scoreStr}`, 'draw');
            }
            document.getElementById('btn-step').disabled = true;
            document.getElementById('btn-play-all').disabled = true;
        } else if (gs.waiting_for_human) {
            const pIdx = gs.current_player;
            if (gs.yams_bonus) {
                Game.setStatus(`Player ${pIdx} — 🍀 Lucky YAMS! Place the max in any cell.`, 'win');
            } else {
                Game.setStatus(`Player ${pIdx}'s turn (Human) — Roll ${3 - gs.rolls_left + 1}/3`);
            }
        } else {
            const pIdx = gs.current_player;
            const pType = gs.player_types ? gs.player_types[pIdx] : 'bot';
            Game.setStatus(`Player ${pIdx}'s turn (${pType}) — click Step to advance`);
        }

        // Variant-aware UI: switch the boards area into 2x2 layout for 2v2
        // and ensure a board container exists for each player.
        const numPlayers = gs.num_players || (gs.player_types ? gs.player_types.length : 2);
        const variant = gs.game_variant || ((numPlayers === 4) ? '2v2' : '1v1');
        Game.ensureBoardContainers(numPlayers, variant);

        // Player type labels.
        if (gs.player_types) {
            for (let p = 0; p < numPlayers; ++p) {
                const el = document.getElementById(`p${p}-type`);
                if (el) el.textContent = `(${gs.player_types[p]})`;
            }
        }

        // Dice.
        Game.renderDice(gs.dice, gs.waiting_for_human, gs.rolls_left);

        // Reroll button.
        document.getElementById('btn-reroll').disabled = !gs.waiting_for_human || gs.rolls_left <= 0;
        document.getElementById('rolls-left').textContent = gs.rolls_left != null ? gs.rolls_left : '-';

        // NN eval display (debug mode).
        const nnPanel = document.getElementById('nn-eval-panel');
        const nnValue = document.getElementById('nn-eval-value');
        if (gs.current_board_nn_value !== undefined) {
            nnPanel.style.display = 'block';
            nnValue.textContent = `${(gs.current_board_nn_value * 100).toFixed(1)}% win prob (last placing player)`;
        } else {
            nnPanel.style.display = 'none';
        }

        // Fetch options if human turn.
        let legalP0 = null, legalP1 = null;
        if (gs.waiting_for_human && !gs.game_over) {
            const opts = await API.getOptions(Game.sessionId);
            Game.options = opts;
            const placements = opts.placements || [];
            if (gs.current_player === 0) legalP0 = placements;
            else legalP1 = placements;

            // Show options panel.
            Game.renderOptions(placements);
            document.getElementById('options-panel').style.display = 'block';
        } else {
            Game.options = null;
            document.getElementById('options-panel').style.display = 'none';
        }

        // Boards. Render one per player; legalPN is non-null only for the
        // current human player's sheet.
        if (gs.boards) {
            for (let p = 0; p < numPlayers; ++p) {
                const board = gs.boards[`player${p}`];
                if (!board) continue;
                let legalForP = null;
                if (gs.waiting_for_human && !gs.game_over && p === gs.current_player) {
                    legalForP = (p === 0) ? legalP0 : (p === 1 ? legalP1 : (Game.options?.placements || []));
                }
                Board.render(`board-grid-${p}`, board, gs.coefficients,
                             legalForP, (c, r) => Game.placeScore(c, r));
            }
        }

        // History.
        Game.renderHistory(gs.history);
    },

    renderDice(dice, isHuman, rollsLeft) {
        const container = document.getElementById('dice-display');
        container.innerHTML = '';
        if (!dice) return;

        for (let i = 0; i < dice.length; i++) {
            const die = document.createElement('div');
            die.className = 'die';
            die.textContent = dice[i];

            if (isHuman && rollsLeft > 0) {
                const held = (Game.holdMask >> i) & 1;
                if (held) die.classList.add('held');
                die.addEventListener('click', () => {
                    Game.holdMask ^= (1 << i);
                    Game.renderDice(dice, isHuman, rollsLeft);
                });
            } else {
                die.classList.add('inactive');
            }

            container.appendChild(die);
        }
    },

    applyLegalCollapse() {
        const list = document.getElementById('options-list');
        const arrow = document.querySelector('#legal-placements-toggle .toggle-arrow');
        list.style.display = Game.legalCollapsed ? 'none' : '';
        if (arrow) arrow.textContent = Game.legalCollapsed ? '▶' : '▼';
    },

    renderOptions(placements) {
        const container = document.getElementById('options-list');
        container.innerHTML = '';
        if (!placements || placements.length === 0) {
            container.textContent = 'No legal placements';
        } else {
            for (const p of placements) {
                const item = document.createElement('div');
                item.className = 'option-item';
                item.innerHTML = `
                    <span>${p.column_name} / ${p.row_name}</span>
                    <span class="opt-score">${p.score}</span>
                `;
                item.addEventListener('click', () => Game.placeScore(p.column, p.row));
                container.appendChild(item);
            }
        }
        Game.applyLegalCollapse();
    },

    renderHistory(history) {
        const container = document.getElementById('history-log');
        container.innerHTML = '';
        if (!history) return;

        for (let i = history.length - 1; i >= 0; i--) {
            const turn = history[i];
            const wrapper = document.createElement('div');
            wrapper.className = 'history-entry-wrapper';

            // --- Summary line ---
            const entry = document.createElement('div');
            entry.className = 'history-entry';

            const pClass = turn.player === 0 ? 'p0' : 'p1';
            const colName = turn.placement.column_name || COLUMN_NAMES[turn.placement.column] || '?';
            const rowName = turn.placement.row_name || ROW_NAMES[turn.placement.row] || '?';

            // Build dice chain: [init] → [d,d,hd,hd,d] → ... → Col/Row = score
            let chain = `[${(turn.initial_dice || []).join(',')}]`;
            if (turn.holds && turn.holds.length > 0) {
                for (const hold of turn.holds) {
                    if (!hold.dice_after || hold.dice_after.length === 0) continue;
                    const diceStr = hold.dice_after.map((v, di) =>
                        (hold.held_flags && hold.held_flags[di]) ? `h${v}` : String(v)
                    ).join(',');
                    chain += ` → [${diceStr}]`;
                }
            }
            chain += ` → ${colName}/${rowName} = ${turn.score}`;

            const hasDebug = turn.debug &&
                ((turn.debug.hold_evals && turn.debug.hold_evals.length > 0) ||
                 (turn.debug.placement_evals && turn.debug.placement_evals.length > 0) ||
                 turn.debug.board_nn_value !== undefined);

            entry.innerHTML =
                `<span class="${pClass}">P${turn.player}</span> ${chain}` +
                ' <span class="copy-board" title="Copy the board position after this turn">⧉</span>' +
                (hasDebug ? ' <span class="debug-toggle" title="Show bot evaluations">▶</span>' : '');

            // Copy the board position as it stood right after this turn, so it
            // can be pasted into "Load Position" to branch from here.
            entry.querySelector('.copy-board').addEventListener('click', (e) => {
                e.stopPropagation();
                Game.copyPositionAt(i, e.target);
            });

            // --- Debug panel (collapsed by default) ---
            let debugPanel = null;
            if (hasDebug) {
                debugPanel = Debug.buildPanel(turn.debug);
                debugPanel.style.display = 'none';

                entry.querySelector('.debug-toggle').addEventListener('click', (e) => {
                    e.stopPropagation();
                    const open = debugPanel.style.display !== 'none';
                    debugPanel.style.display = open ? 'none' : 'block';
                    e.target.textContent = open ? '▶' : '▼';
                });
            }

            wrapper.appendChild(entry);
            if (debugPanel) wrapper.appendChild(debugPanel);
            container.appendChild(wrapper);
        }
    },

    setStatus(text, cls) {
        const bar = document.getElementById('game-status');
        bar.textContent = text;
        bar.className = 'status-bar' + (cls ? ` ${cls}` : '');
    }
};
