// Recorded Games viewer for the Pro Yams admin UI.
//
// Reads the games directory written by the play servers (pro_yams_play) via
// /api/games/{stats,list,<uuid>}. Shows per-checkpoint performance against
// humans (incl. drop rate), a filterable/paginated game list, and a
// turn-by-turn replay reconstructed from the stored move history.

const GAMES_COLUMN_NAMES = ['Down', 'Free', 'Up', 'Mid', 'Turbo', 'UpDown'];
const GAMES_COLUMN_KEYS  = ['down', 'free', 'up', 'mid', 'turbo', 'updown'];
const GAMES_ROW_NAMES     = ['1s','2s','3s','4s','5s','6s','SS','LS','FH','K','STR','U8','Y'];

const Games = {
    pageSize: 50,
    offset: 0,
    total: 0,
    filters: { checkpoint: '', variant: '', outcome: '' },

    // Replay state
    replay: { record: null, states: null, turn: 0, numPlayers: 2 },

    init() {
        const byId = (id) => document.getElementById(id);
        byId('btn-games-refresh').addEventListener('click', () => Games.loadAll());
        byId('btn-games-apply').addEventListener('click', () => {
            Games.filters.checkpoint = byId('games-filter-checkpoint').value;
            Games.filters.variant    = byId('games-filter-variant').value;
            Games.filters.outcome    = byId('games-filter-outcome').value;
            Games.offset = 0;
            Games.loadList();
        });
        byId('btn-games-prev').addEventListener('click', () => {
            Games.offset = Math.max(0, Games.offset - Games.pageSize);
            Games.loadList();
        });
        byId('btn-games-next').addEventListener('click', () => {
            if (Games.offset + Games.pageSize < Games.total) {
                Games.offset += Games.pageSize;
                Games.loadList();
            }
        });

        byId('btn-games-replay-close').addEventListener('click', () => Games.closeReplay());
        byId('games-replay-backdrop').addEventListener('click', () => Games.closeReplay());
        byId('btn-replay-first').addEventListener('click', () => Games.gotoTurn(0));
        byId('btn-replay-prev').addEventListener('click', () => Games.gotoTurn(Games.replay.turn - 1));
        byId('btn-replay-next').addEventListener('click', () => Games.gotoTurn(Games.replay.turn + 1));
        byId('btn-replay-last').addEventListener('click', () => Games.gotoTurn(Games.replay.states.length - 1));
        byId('btn-replay-breakdown').addEventListener('click', () => Games.toggleBreakdown());
        byId('btn-replay-eval').addEventListener('click', () => Games.toggleEval());

        // Lazy-load the first time the tab is opened.
        document.querySelector('.tab-btn[data-tab="games"]').addEventListener('click', () => {
            if (!Games._loaded) { Games._loaded = true; Games.loadAll(); }
        });
    },

    async loadAll() {
        await Games.loadStats();
        await Games.loadList();
    },

    // ---------------- Stats ----------------
    async loadStats() {
        const stats = await API.getGamesStats();
        const tbl = document.getElementById('games-stats-table');
        const overallEl = document.getElementById('games-overall');
        if (!stats) {
            tbl.innerHTML = '';
            overallEl.textContent = 'No games directory configured on the server.';
            return;
        }
        const o = stats.overall || {};
        overallEl.textContent =
            `Started ${o.started || 0} · Finished ${o.finished || 0} · ` +
            `Dropped ${o.dropped || 0} (${pct(o.drop_rate)}) · ` +
            `Human win-rate ${pct(o.human_win_rate)} · Avg margin ${fmtSignedNum(o.avg_human_margin)}`;

        const rows = (stats.by_checkpoint || []).slice().sort((a, b) => {
            if (a.checkpoint === b.checkpoint) return a.variant < b.variant ? -1 : 1;
            return a.checkpoint < b.checkpoint ? -1 : 1;
        });

        let html = `<thead><tr>
            <th>Checkpoint</th><th>Variant</th>
            <th>Started</th><th>Finished</th><th>Dropped</th><th>Drop %</th>
            <th>Human W</th><th>L</th><th>D</th><th>Win %</th><th>Avg margin</th>
          </tr></thead><tbody>`;
        for (const r of rows) {
            html += `<tr>
                <td>${escapeHtml(r.checkpoint || '—')}</td>
                <td>${r.variant || ''}</td>
                <td>${r.started}</td>
                <td>${r.finished}</td>
                <td>${r.dropped}</td>
                <td>${pct(r.drop_rate)}</td>
                <td>${r.human_wins}</td>
                <td>${r.human_losses}</td>
                <td>${r.draws}</td>
                <td>${pct(r.human_win_rate)}</td>
                <td class="${marginCls(r.avg_human_margin)}">${fmtSignedNum(r.avg_human_margin)}</td>
              </tr>`;
        }
        html += '</tbody>';
        tbl.innerHTML = html;

        // Populate the checkpoint filter (preserve current selection).
        const sel = document.getElementById('games-filter-checkpoint');
        const cur = sel.value;
        const ckpts = [...new Set(rows.map(r => r.checkpoint).filter(Boolean))].sort();
        sel.innerHTML = '<option value="">All</option>' +
            ckpts.map(c => `<option value="${escapeAttr(c)}">${escapeHtml(c)}</option>`).join('');
        sel.value = cur;

        Games.renderHumans(stats.by_human || []);
    },

    // Per-human performance, keyed by client IP + user-agent. Sorted by most
    // recently active. The human "wins" when they beat the NN.
    renderHumans(humans) {
        const tbl = document.getElementById('games-humans-table');
        if (!tbl) return;
        const rows = humans.slice().sort((a, b) => (b.last_ts_ms || 0) - (a.last_ts_ms || 0));

        let html = `<thead><tr>
            <th>IP</th><th>User-agent</th><th>Last seen</th>
            <th>Started</th><th>Finished</th><th>Dropped</th><th>Drop %</th>
            <th>W</th><th>L</th><th>D</th><th>Win %</th><th>Avg margin</th>
          </tr></thead><tbody>`;
        if (rows.length === 0) {
            html += `<tr><td colspan="12" style="text-align:center; color:var(--muted);">No human games.</td></tr>`;
        }
        for (const r of rows) {
            html += `<tr>
                <td>${escapeHtml(r.ip || '?')}</td>
                <td class="games-ua-cell" title="${escapeAttr(r.user_agent || '')}">${escapeHtml(uaShort(r.user_agent))}</td>
                <td>${fmtTime(r.last_ts_ms)}</td>
                <td>${r.started}</td>
                <td>${r.finished}</td>
                <td>${r.dropped}</td>
                <td>${pct(r.drop_rate)}</td>
                <td>${r.human_wins}</td>
                <td>${r.human_losses}</td>
                <td>${r.draws}</td>
                <td>${pct(r.human_win_rate)}</td>
                <td class="${marginCls(r.avg_human_margin)}">${fmtSignedNum(r.avg_human_margin)}</td>
              </tr>`;
        }
        html += '</tbody>';
        tbl.innerHTML = html;
    },

    // ---------------- List ----------------
    async loadList() {
        const data = await API.listGames({
            checkpoint: Games.filters.checkpoint,
            variant:    Games.filters.variant,
            outcome:    Games.filters.outcome,
            limit:      Games.pageSize,
            offset:     Games.offset,
        });
        Games.total = data.total || 0;
        const tbl = document.getElementById('games-list-table');
        const games = data.games || [];

        let html = `<thead><tr>
            <th>Finished</th><th>Variant</th><th>Human seats</th>
            <th>Checkpoint</th><th>Result</th><th>Margin</th><th>Duration</th><th></th>
          </tr></thead><tbody>`;
        if (games.length === 0) {
            html += `<tr><td colspan="8" style="text-align:center; color:var(--muted);">No games.</td></tr>`;
        }
        for (const g of games) {
            const hr = humanResult(g);
            const hm = humanMargin(g);
            html += `<tr class="games-row" data-uuid="${escapeAttr(g.uuid)}">
                <td>${fmtTime(g.finish_ts_ms)}</td>
                <td>${g.variant || ''}</td>
                <td>${(g.human_seats || []).join(', ')}</td>
                <td>${escapeHtml(g.checkpoint || '—')}</td>
                <td class="res-${hr}">${hr.toUpperCase()}</td>
                <td class="${marginCls(hm)}">${fmtSignedNum(hm)}</td>
                <td>${fmtDuration(g.duration_ms)}</td>
                <td><button class="games-view-btn">View</button></td>
              </tr>`;
        }
        html += '</tbody>';
        tbl.innerHTML = html;

        tbl.querySelectorAll('.games-row').forEach(tr => {
            tr.addEventListener('click', () => Games.openReplay(tr.dataset.uuid));
        });

        const from = Games.total === 0 ? 0 : Games.offset + 1;
        const to = Math.min(Games.offset + Games.pageSize, Games.total);
        document.getElementById('games-page-info').textContent = `${from}–${to} of ${Games.total}`;
        document.getElementById('btn-games-prev').disabled = Games.offset <= 0;
        document.getElementById('btn-games-next').disabled = Games.offset + Games.pageSize >= Games.total;
    },

    // ---------------- Replay ----------------
    async openReplay(uuid) {
        const record = await API.getGameRecord(uuid);
        if (!record) return;
        const fs = record.final_state || {};
        const numPlayers = fs.num_players || 2;
        const history = fs.history || [];

        // Reconstruct the board after each placement. states[i] = boards after
        // i moves; states[0] is the empty board.
        const emptyBoards = () => {
            const b = {};
            for (let p = 0; p < numPlayers; p++) {
                b['player' + p] = {};
                for (const ck of GAMES_COLUMN_KEYS) {
                    b['player' + p][ck] = {};
                    for (const rk of GAMES_ROW_NAMES) b['player' + p][ck][rk] = undefined;
                }
            }
            return b;
        };
        const deepCopy = (b) => JSON.parse(JSON.stringify(b));

        const states = [];
        let boards = emptyBoards();
        states.push(deepCopy(boards));
        for (const t of history) {
            const ck = GAMES_COLUMN_KEYS[t.placement.column];
            const rk = GAMES_ROW_NAMES[t.placement.row];
            const pcol = boards['player' + t.player][ck];
            pcol[rk] = t.score;
            // SS/LS mutual destruction (mirrors engine placement.cc): scratching
            // one of SS/LS retroactively zeroes the other if it already held a
            // non-zero score. The history's stored score for the earlier cell is
            // stale, so re-apply the rule here or the board (and the score
            // breakdown) would over-count the partner cell.
            if (t.score === 0) {
                if (rk === 'SS' && pcol['LS'] > 0) pcol['LS'] = 0;
                else if (rk === 'LS' && pcol['SS'] > 0) pcol['SS'] = 0;
            }
            states.push(deepCopy(boards));
        }

        Games.replay = {
            record, states, turn: states.length - 1, numPlayers, history,
            showBreakdown: false,
            // NN eval (lazy, enabled via the "Load checkpoint" button):
            evalOn: false, evalCache: {}, evalErr: null,
        };
        document.getElementById('games-replay-modal').style.display = 'block';
        Games.resetEvalButton();
        Games.renderReplayMeta();
        Games.renderReplayBody();
        Games.renderEvalLine();
    },

    closeReplay() {
        document.getElementById('games-replay-modal').style.display = 'none';
        Games.replay.record = null;
    },

    gotoTurn(turn) {
        const states = Games.replay.states;
        if (!states) return;
        Games.replay.turn = Math.max(0, Math.min(states.length - 1, turn));
        Games.replay.showBreakdown = false;  // navigating returns to the board view
        Games.renderReplayBody();
        if (Games.replay.evalOn) Games.evaluateCurrent();
        Games.renderEvalLine();
    },

    toggleBreakdown() {
        if (!Games.replay.states) return;
        Games.replay.showBreakdown = !Games.replay.showBreakdown;
        Games.renderReplayBody();
    },

    // ---------------- NN eval ----------------
    // Turn the per-step win-rate display on/off. When turning it on we kick off
    // an evaluation of the current step; the server loads the game's checkpoint
    // on demand (cached, so repeated previews of the same checkpoint are free).
    resetEvalButton() {
        const btn = document.getElementById('btn-replay-eval');
        if (!btn) return;
        const hasCkpt = !!(Games.replay.record && Games.replay.record.checkpoint);
        btn.disabled = !hasCkpt;
        btn.classList.remove('active');
        btn.textContent = hasCkpt ? '🎯 Load checkpoint' : '🎯 No checkpoint';
    },

    async toggleEval() {
        if (!Games.replay.states) return;
        if (!Games.replay.record || !Games.replay.record.checkpoint) return;
        Games.replay.evalOn = !Games.replay.evalOn;
        const btn = document.getElementById('btn-replay-eval');
        if (btn) {
            btn.classList.toggle('active', Games.replay.evalOn);
            btn.textContent = Games.replay.evalOn ? '🎯 Checkpoint loaded' : '🎯 Load checkpoint';
        }
        if (Games.replay.evalOn) Games.evaluateCurrent();
        Games.renderEvalLine();
    },

    // Build a /api/game position object from the reconstructed board at `turn`.
    // Empty cells are omitted (the server treats absent cells as unfilled);
    // 0 means a scratch, >0 a placed score — matching parse_position's reader.
    buildEvalPosition(turn) {
        const { states, numPlayers, record } = Games.replay;
        const fs = record.final_state || {};
        const coeffs = record.coefficients || fs.coefficients || [];
        const boards = states[turn];
        const outBoards = {};
        for (let p = 0; p < numPlayers; p++) {
            const src = boards['player' + p];
            const dst = {};
            for (const ck of GAMES_COLUMN_KEYS) {
                dst[ck] = {};
                for (const rk of GAMES_ROW_NAMES) {
                    const v = src[ck][rk];
                    if (v !== undefined) dst[ck][rk] = v;
                }
            }
            outBoards['player' + p] = dst;
        }
        return {
            variant: record.variant,
            coefficients: coeffs,
            current_player: 0,
            boards: outBoards,
        };
    },

    // Evaluate the current step from the perspective of the player who just
    // placed (history[turn-1].player). Results are cached per turn. The empty
    // start state (turn 0) has no placement, so nothing is evaluated there.
    async evaluateCurrent() {
        const { turn, history, record } = Games.replay;
        if (turn <= 0) return;
        const cache = Games.replay.evalCache;
        if (cache[turn] && !cache[turn].error) return;  // done or in flight

        const player = history[turn - 1].player;
        const position = Games.buildEvalPosition(turn);
        cache[turn] = { pending: true, player };
        Games.replay.evalErr = null;
        Games.renderEvalLine();

        const r = await API.evalGamePosition({
            checkpoint: record.checkpoint, player, position,
        });

        // The modal may have closed or moved on while we awaited.
        if (!Games.replay.record || Games.replay.record.uuid !== record.uuid) return;
        if (r.error) {
            delete cache[turn];
            Games.replay.evalErr = r.error;
        } else {
            cache[turn] = { value: r.nn_value, player: r.player };
        }
        Games.renderEvalLine();
    },

    renderEvalLine() {
        const el = document.getElementById('replay-eval-info');
        if (!el) return;
        const { evalOn, turn, evalCache, evalErr } = Games.replay;
        if (!evalOn) { el.textContent = ''; return; }
        if (evalErr) { el.innerHTML = `<span class="eval-err">eval failed: ${escapeHtml(evalErr)}</span>`; return; }
        if (turn <= 0) { el.innerHTML = `<span class="muted">step to a placed move to see its win-rate</span>`; return; }
        const e = evalCache[turn];
        if (!e || e.pending) { el.innerHTML = `<span class="muted">evaluating…</span>`; return; }
        el.innerHTML =
            `<span class="eval-label">P${e.player} win-rate (checkpoint)</span> ` +
            `<span class="eval-val">${pct(e.value)}</span>`;
    },

    // Dispatch the modal body to either the turn board or the score breakdown,
    // and keep the toggle button's label in sync with the current view.
    renderReplayBody() {
        const showing = Games.replay.showBreakdown;
        const btn = document.getElementById('btn-replay-breakdown');
        if (btn) btn.textContent = showing ? '◀ Back to replay' : '📊 Score breakdown';
        if (showing) Games.renderBreakdown();
        else Games.renderReplayTurn();
    },

    renderReplayMeta() {
        const r = Games.replay.record;
        const fs = r.final_state || {};
        const hr = humanResult(r);
        const hm = humanMargin(r);
        const coeffs = (r.coefficients || fs.coefficients || []);
        document.getElementById('games-replay-meta').innerHTML =
            `<div><strong>${escapeHtml(r.uuid)}</strong></div>` +
            `<div>${r.variant || ''} · checkpoint <strong>${escapeHtml(r.checkpoint || '—')}</strong> · ` +
            `human seats [${(r.human_seats || []).join(', ')}] · ` +
            `result <span class="res-${hr}">${hr.toUpperCase()}</span> (margin ${fmtSignedNum(hm)})</div>` +
            `<div>${fmtTime(r.start_ts_ms)} → ${fmtTime(r.finish_ts_ms)} · ${fmtDuration(r.duration_ms)} · ` +
            `IP ${escapeHtml(r.ip || '?')}${r.x_forwarded_for ? ' (xff ' + escapeHtml(r.x_forwarded_for) + ')' : ''}</div>` +
            `<div class="games-coeffs">Coefficients: ${GAMES_COLUMN_NAMES.map((n, i) => `${n}×${coeffs[i]}`).join('  ')}</div>` +
            `<div class="games-ua">${escapeHtml(r.user_agent || '')}</div>`;
    },

    renderReplayTurn() {
        const { states, turn, numPlayers, history, record } = Games.replay;
        const boards = states[turn];
        const fs = record.final_state || {};
        const coeffs = record.coefficients || fs.coefficients || [];

        // The move that produced this state (turn>0) — for highlighting + info.
        const justTurn = turn > 0 ? history[turn - 1] : null;
        const info = document.getElementById('replay-turn-info');
        if (justTurn) {
            const colName = GAMES_COLUMN_NAMES[justTurn.placement.column];
            const rowName = GAMES_ROW_NAMES[justTurn.placement.row];
            const sc = justTurn.score;
            info.textContent = `Move ${turn}/${states.length - 1} · P${justTurn.player} → ` +
                `${rowName} in ${colName} = ${sc === 0 ? 'scratch' : sc}`;
        } else {
            info.textContent = `Start (0/${states.length - 1})`;
        }

        const body = document.getElementById('games-replay-body');
        const highlight = justTurn
            ? { player: justTurn.player, column: justTurn.placement.column, row: justTurn.placement.row }
            : null;

        let html = '<div class="replay-dice">' + Games.renderTurnDice(justTurn) + '</div>';
        html += '<div class="replay-boards">';
        for (let p = 0; p < numPlayers; p++) {
            html += Games.renderBoard(p, boards['player' + p], coeffs, highlight);
        }
        html += '</div>';
        body.innerHTML = html;
    },

    renderTurnDice(t) {
        if (!t) return '<span class="muted">—</span>';
        const dieRow = (dice) => (dice || []).map(d => `<span class="replay-die">${d}</span>`).join('');
        let html = `<span class="replay-dice-label">P${t.player} roll:</span> ${dieRow(t.initial_dice)}`;
        for (const h of (t.holds || [])) {
            html += ` <span class="replay-arrow">→</span> ${dieRow(h.dice_after)}`;
        }
        return html;
    },

    renderBoard(player, board, coeffs, highlight) {
        let html = `<div class="replay-board"><div class="replay-board-title">Player ${player}</div>`;
        html += '<table class="replay-grid"><thead><tr><th></th>';
        for (let c = 0; c < 6; c++) html += `<th>${GAMES_COLUMN_NAMES[c]}<br><span class="muted">×${coeffs[c] ?? ''}</span></th>`;
        html += '</tr></thead><tbody>';
        for (let r = 0; r < 13; r++) {
            html += `<tr><td class="replay-rowlabel">${GAMES_ROW_NAMES[r]}</td>`;
            for (let c = 0; c < 6; c++) {
                const v = board[GAMES_COLUMN_KEYS[c]][GAMES_ROW_NAMES[r]];
                const hl = highlight && highlight.player === player &&
                           highlight.column === c && highlight.row === r;
                let txt = '', cls = 'replay-cell';
                if (v === undefined) { txt = ''; }
                else if (v === 0) { txt = '0'; cls += ' scratched'; }
                else { txt = v; }
                if (hl) cls += ' just-placed';
                html += `<td class="${cls}">${txt}</td>`;
            }
            html += '</tr>';
        }
        html += '</tbody></table></div>';
        return html;
    },

    // Re-derive the full final-score calculation from the reconstructed final
    // board — the exact math of the engine's compute_duel_columns(): per-player
    // raw scores, per-pairing crush multiplier, clean-column bonus, and the
    // resulting Team-0-perspective points per column. Cross-checked against the
    // recorded column_margins so any drift from the engine is visible.
    renderBreakdown() {
        const { states, numPlayers, record } = Games.replay;
        const fs = record.final_state || {};
        const coeffs = record.coefficients || fs.coefficients || [];
        const finalBoards = states[states.length - 1];

        // Team membership: 1v1 → {0} vs {1}; 2v2 → {0,2} vs {1,3}.
        const teamOf = (p) => p % 2;            // even seats = Team 0
        const team0 = [], team1 = [];
        for (let p = 0; p < numPlayers; p++) (teamOf(p) === 0 ? team0 : team1).push(p);

        // D[p][c]: raw-score breakdown + clean-column eligibility for player p,
        // column c, read straight off the final board.
        const D = [];
        for (let p = 0; p < numPlayers; p++) {
            D[p] = [];
            for (let c = 0; c < 6; c++) D[p][c] = colBreakdown(finalBoards, p, c);
        }

        let html = '<div class="breakdown">';

        // ---- Raw column scores ----
        html += '<h4 class="breakdown-h">Raw column scores <span class="muted">(cell sum + upper bonus)</span></h4>';
        html += '<table class="breakdown-raw"><thead><tr><th></th>';
        for (let c = 0; c < 6; c++)
            html += `<th>${GAMES_COLUMN_NAMES[c]}<br><span class="muted">×${coeffs[c] ?? ''}</span></th>`;
        html += '</tr></thead><tbody>';
        for (let p = 0; p < numPlayers; p++) {
            html += `<tr><td class="breakdown-plabel">P${p} <span class="team-tag team${teamOf(p)}">T${teamOf(p)}</span></td>`;
            for (let c = 0; c < 6; c++) {
                const d = D[p][c];
                const bonusTxt = d.bonus ? ` <span class="up-bonus">+${d.bonus}↑</span>` : '';
                const cleanTxt = d.clean ? ' <span class="clean-flag" title="upper ≥ 60 and no lower scratch">✓clean</span>' : '';
                html += `<td><span class="raw-total">${d.raw}</span>` +
                        `<div class="raw-detail">${d.cellSum}${bonusTxt}${cleanTxt}` +
                        `<br><span class="muted">upper ${d.upperSum}</span></div></td>`;
            }
            html += '</tr>';
        }
        html += '</tbody></table>';

        // ---- Per-column duel ----
        html += '<h4 class="breakdown-h">Per-column duel</h4>';
        let grand = 0;
        for (let c = 0; c < 6; c++) {
            let colTotal = 0, rows = '';
            for (const t0p of team0) {
                for (const t1p of team1) {
                    const d0 = D[t0p][c], d1 = D[t1p][c];
                    const crush0 = crushMultiplier(d0.raw, d1.raw);
                    const crush1 = crushMultiplier(d1.raw, d0.raw);
                    const crush  = Math.max(crush0, crush1);
                    const bonusVal = crush > 1 ? 100 : 200;
                    const adj0 = d0.raw + (d0.clean ? bonusVal : 0);
                    const adj1 = d1.raw + (d1.clean ? bonusVal : 0);
                    const diff = adj0 - adj1;
                    const pts  = diff * crush * (coeffs[c] || 0);
                    colTotal += pts;

                    const crushTxt = crush > 1
                        ? `×${crush} <span class="muted">(P${crush0 >= crush1 ? t0p : t1p} crushes)</span>`
                        : '×1';
                    const cleanTxt = `${d0.clean ? '+' + bonusVal : '—'} / ${d1.clean ? '+' + bonusVal : '—'}`;
                    rows += `<tr>
                        <td>P${t0p} vs P${t1p}</td>
                        <td>${crushTxt}</td>
                        <td>${cleanTxt}</td>
                        <td>${adj0} − ${adj1} = ${diff}</td>
                        <td class="muted">${diff} × ${crush} × ${coeffs[c]}</td>
                        <td class="${marginCls(pts)}">${fmtSignedNum(pts)}</td>
                      </tr>`;
                }
            }
            grand += colTotal;

            // Cross-check the recorded margin for this column.
            const recorded = (record.column_margins || [])[c];
            const mismatch = (recorded != null && recorded !== colTotal)
                ? ` <span class="breakdown-warn" title="recorded ${recorded}">⚠ recorded ${fmtSignedNum(recorded)}</span>`
                : '';

            html += `<div class="breakdown-col">
                <div class="breakdown-col-head">${GAMES_COLUMN_NAMES[c]} <span class="muted">(×${coeffs[c]})</span> →
                  <span class="${marginCls(colTotal)}">${fmtSignedNum(colTotal)}</span>${mismatch}</div>
                <table class="breakdown-pairings"><thead><tr>
                  <th>Pairing</th><th>Crush</th><th>Clean bonus T0/T1</th><th>Adjusted diff</th><th>Calc</th><th>Points</th>
                </tr></thead><tbody>${rows}</tbody></table></div>`;
        }

        const recTotal = record.total_margin;
        const totMismatch = (recTotal != null && recTotal !== grand)
            ? ` <span class="breakdown-warn">⚠ recorded ${fmtSignedNum(recTotal)}</span>`
            : '';
        html += `<div class="breakdown-grand">Total margin (Team 0): ` +
                `<span class="${marginCls(grand)}">${fmtSignedNum(grand)}</span>${totMismatch}</div>`;
        html += '</div>';

        document.getElementById('games-replay-body').innerHTML = html;
    },
};

// ---------------- Helpers ----------------
// Scoring helpers — must mirror src/engine/duel.cc exactly.
function upperSectionBonus(sum) {
    if (sum >= 100) return 500;
    if (sum >= 90)  return 200;
    if (sum >= 80)  return 100;
    if (sum >= 70)  return 50;
    if (sum >= 60)  return 30;
    return 0;
}
function crushMultiplier(myRaw, oppRaw) {
    if (oppRaw === 0 && myRaw > 0) return 5;
    if (oppRaw > 0 && myRaw >= 5 * oppRaw) return 5;
    if (oppRaw > 0 && myRaw >= 4 * oppRaw) return 4;
    if (oppRaw > 0 && myRaw >= 3 * oppRaw) return 3;
    if (oppRaw > 0 && myRaw >= 2 * oppRaw) return 2;
    return 1;
}
// Read a single (player, column) from the reconstructed final board: cell sum,
// upper-section sum/bonus, raw score, and clean-column eligibility. Rows 0–5 are
// the upper section; a lower-section cell holding 0 is a scratch.
function colBreakdown(boards, player, c) {
    const cells = boards['player' + player][GAMES_COLUMN_KEYS[c]];
    let cellSum = 0, upperSum = 0, lowerScratch = false;
    for (let r = 0; r < GAMES_ROW_NAMES.length; r++) {
        const v = cells[GAMES_ROW_NAMES[r]];
        if (r < 6) { if (v > 0) upperSum += v; }
        else if (v === 0) { lowerScratch = true; }
        if (v > 0) cellSum += v;
    }
    const bonus = upperSectionBonus(upperSum);
    return {
        cellSum, upperSum, bonus,
        raw: cellSum + bonus,
        clean: upperSum >= 60 && !lowerScratch,
    };
}

function humanResult(g) {
    const result = g.result;
    const team = (g.human_team != null) ? g.human_team : -1;
    if (result === 0.5 || team < 0) return 'draw';
    const team0Won = result > 0.5;
    const humanWon = (team0Won && team === 0) || (!team0Won && team === 1);
    return humanWon ? 'win' : 'loss';
}
function humanMargin(g) {
    const m = g.total_margin || 0;
    return (g.human_team === 0) ? m : -m;
}
function pct(x) { return x == null ? '—' : (100 * x).toFixed(1) + '%'; }
function fmtSignedNum(x) {
    if (x == null) return '—';
    const n = Math.round(x * 10) / 10;
    return n > 0 ? `+${n}` : `${n}`;
}
function marginCls(x) { return x > 0 ? 'pts-pos' : x < 0 ? 'pts-neg' : 'pts-zero'; }
function fmtTime(ms) {
    if (!ms) return '—';
    return new Date(ms).toLocaleString();
}
function fmtDuration(ms) {
    if (!ms) return '—';
    const s = Math.round(ms / 1000);
    if (s < 60) return `${s}s`;
    const m = Math.floor(s / 60);
    return `${m}m ${s % 60}s`;
}
// Condense a user-agent to the browser/OS that matters at a glance; the full
// string is kept in the cell's title attribute.
function uaShort(ua) {
    if (!ua) return '?';
    const os =
        /Windows/.test(ua)            ? 'Windows' :
        /iPhone|iPad|iOS/.test(ua)    ? 'iOS'     :
        /Android/.test(ua)            ? 'Android' :
        /Mac OS X|Macintosh/.test(ua) ? 'macOS'   :
        /Linux/.test(ua)              ? 'Linux'   : '';
    let browser = 'Other';
    let m;
    if ((m = ua.match(/Edg\/(\d+)/)))                            browser = 'Edge ' + m[1];
    else if (/OPR\/|Opera/.test(ua))                            browser = 'Opera';
    else if ((m = ua.match(/Firefox\/(\d+)/)))                  browser = 'Firefox ' + m[1];
    else if ((m = ua.match(/Chrome\/(\d+)/)))                   browser = 'Chrome ' + m[1];
    else if (/Safari/.test(ua) && (m = ua.match(/Version\/(\d+)/))) browser = 'Safari ' + m[1];
    return os ? `${browser} · ${os}` : browser;
}
function escapeHtml(s) {
    return String(s == null ? '' : s)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
function escapeAttr(s) { return escapeHtml(s).replace(/"/g, '&quot;'); }
