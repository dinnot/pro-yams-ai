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
    filters: { checkpoint: '', variant: '', outcome: '', flagged: false },

    // Replay state
    replay: { record: null, states: null, turn: 0, numPlayers: 2 },

    init() {
        const byId = (id) => document.getElementById(id);
        byId('btn-games-refresh').addEventListener('click', () => Games.loadAll());
        byId('btn-games-apply').addEventListener('click', () => {
            Games.filters.checkpoint = byId('games-filter-checkpoint').value;
            Games.filters.variant    = byId('games-filter-variant').value;
            Games.filters.outcome    = byId('games-filter-outcome').value;
            Games.filters.flagged    = byId('games-filter-flagged').checked;
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
        byId('btn-replay-moves').addEventListener('click', () => Games.toggleMoves());
        byId('btn-replay-accuracy').addEventListener('click', () => Games.evaluateAccuracy());
        document.querySelectorAll('input[name="moves-filter"]').forEach(r => {
            r.addEventListener('change', () => {
                Games.replay.movesFilter = r.value;
                if (Games.replay.showMoves) Games.renderReplayBody();
            });
        });

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
            flagged:    Games.filters.flagged,
            limit:      Games.pageSize,
            offset:     Games.offset,
        });
        Games.total = data.total || 0;
        const tbl = document.getElementById('games-list-table');
        const games = data.games || [];

        let html = `<thead><tr>
            <th>Finished</th><th>Variant</th><th>Human seats</th>
            <th>Checkpoint</th><th>Result</th><th>Margin</th><th>Duration</th><th></th><th></th>
          </tr></thead><tbody>`;
        if (games.length === 0) {
            html += `<tr><td colspan="9" style="text-align:center; color:var(--muted);">No games.</td></tr>`;
        }
        for (const g of games) {
            const hr = humanResult(g);
            const hm = humanMargin(g);
            const flag = g.flagged
                ? `<span class="games-flag" title="Flagged: AI played weird moves">🚩</span>` : '';
            html += `<tr class="games-row${g.flagged ? ' games-row-flagged' : ''}" data-uuid="${escapeAttr(g.uuid)}">
                <td>${fmtTime(g.finish_ts_ms)}</td>
                <td>${g.variant || ''}</td>
                <td>${(g.human_seats || []).join(', ')}</td>
                <td>${escapeHtml(g.checkpoint || '—')}</td>
                <td class="res-${hr}">${hr.toUpperCase()}</td>
                <td class="${marginCls(hm)}">${fmtSignedNum(hm)}</td>
                <td>${fmtDuration(g.duration_ms)}</td>
                <td>${flag}</td>
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
            // Per-move accuracy (win% given up on the move being shown), cached
            // by turn. Populated lazily while the checkpoint eval is on, and in
            // bulk by the "Evaluate human accuracy" button.
            moveAccCache: {}, moveAccErr: null,
            // Move-eval table (lazy, enabled via the "Move evals" button):
            showMoves: false, movesFilter: 'valid', movesCache: {}, movesErr: null,
        };
        document.getElementById('games-replay-modal').style.display = 'block';
        Games.resetEvalButton();
        Games.resetMovesButton();
        Games.resetAccuracyButton();
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
        if (Games.replay.showMoves) Games.evaluateMovesCurrent();
        Games.renderReplayBody();
        if (Games.replay.evalOn) { Games.evaluateCurrent(); Games.evaluateMoveAccCurrent(); }
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
        if (Games.replay.evalOn) { Games.evaluateCurrent(); Games.evaluateMoveAccCurrent(); }
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

    // Args for /api/games/accuracy_turn for the move at history index `i`
    // (the board before it is states[i]).
    accuracyTurnArgs(i) {
        const { history, record } = Games.replay;
        const t = history[i];
        return {
            checkpoint: record.checkpoint,
            player: t.player,
            position: Games.buildEvalPosition(i),
            initial_dice: t.initial_dice,
            holds: (t.holds || []).map(h => ({ mask: h.mask, dice_after: h.dice_after })),
            placement: { column: t.placement.column, row: t.placement.row },
            score: t.score,
        };
    },

    // Per-move accuracy for the move that produced the current state (the move
    // history[turn-1]). Cached per turn; shares the cache the bulk "Evaluate
    // human accuracy" button fills.
    async evaluateMoveAccCurrent() {
        const { turn, history, record } = Games.replay;
        if (turn <= 0) return;
        const cache = Games.replay.moveAccCache;
        if (cache[turn] && !cache[turn].error) return;  // done or in flight

        cache[turn] = { pending: true, player: history[turn - 1].player };
        Games.replay.moveAccErr = null;
        Games.renderEvalLine();

        const r = await API.accuracyTurn(Games.accuracyTurnArgs(turn - 1));
        if (!Games.replay.record || Games.replay.record.uuid !== record.uuid) return;
        if (r.error) { delete cache[turn]; Games.replay.moveAccErr = r.error; }
        else cache[turn] = {
            player: r.player,
            holdDeltas: r.hold_deltas || [],
            placeDelta: r.has_place ? r.place_delta : null,
            rollLucks: r.roll_lucks || [],
        };
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
        let html = (!e || e.pending)
            ? `<span class="muted">evaluating…</span>`
            : `<span class="eval-label">P${e.player} win-rate (checkpoint)</span> ` +
              `<span class="eval-val">${pct(e.value)}</span>`;
        html += ' · ' + Games.moveAccText(turn);
        el.innerHTML = html;
    },

    // Inline "win% lost on this move" + "turn luck" fragment for the eval line.
    moveAccText(turn) {
        const a = Games.replay.moveAccCache[turn];
        if (!a || a.pending) return `<span class="muted">move eval…</span>`;
        const pp = (x) => x == null ? '—' : (100 * x).toFixed(2) + 'pp';
        const holds = (a.holdDeltas && a.holdDeltas.length)
            ? a.holdDeltas.reduce((s, x) => s + x, 0) : null;
        // Turn luck = mean of this turn's per-roll luck, remapped to 0..100%.
        const lk = (a.rollLucks && a.rollLucks.length)
            ? a.rollLucks.reduce((s, x) => s + x, 0) / a.rollLucks.length : null;
        const luckPct = lk == null ? '—' : ((lk + 100) / 2).toFixed(1) + '%';
        return `<span class="acc-label">lost</span> ` +
            `holds <span class="acc-val">${pp(holds)}</span> · ` +
            `placement <span class="acc-val">${pp(a.placeDelta)}</span> · ` +
            `<span class="acc-label" title="Luck of this turn's dice (50% = as expected).">` +
            `turn luck</span> <span class="acc-val">${luckPct}</span>`;
    },

    // ---------------- Move evals (all possible next moves) ----------------
    // Like the eval line, but instead of a single win-rate it loads the
    // checkpoint's evaluation of *every* legal next-move afterstate for the
    // player about to move, shown as a table. The filter toggles between the
    // placement-step subset valid for the rolled dice and all possibilities.
    resetMovesButton() {
        const btn = document.getElementById('btn-replay-moves');
        if (!btn) return;
        const hasCkpt = !!(Games.replay.record && Games.replay.record.checkpoint);
        btn.disabled = !hasCkpt;
        btn.classList.remove('active');
        btn.textContent = hasCkpt ? '📋 Move evals' : '📋 No checkpoint';
        const filt = document.getElementById('replay-moves-filter');
        if (filt) filt.hidden = true;
    },

    toggleMoves() {
        if (!Games.replay.states) return;
        if (!Games.replay.record || !Games.replay.record.checkpoint) return;
        Games.replay.showMoves = !Games.replay.showMoves;
        if (Games.replay.showMoves) Games.replay.showBreakdown = false;
        const btn = document.getElementById('btn-replay-moves');
        if (btn) btn.classList.toggle('active', Games.replay.showMoves);
        if (Games.replay.showMoves) Games.evaluateMovesCurrent();
        Games.renderReplayBody();
    },

    // Final dice for the turn that produced state `turn` = the dice after the
    // last reroll, or the initial roll if the player never rerolled.
    turnFinalDice(turnRec) {
        if (!turnRec) return null;
        const holds = turnRec.holds || [];
        if (holds.length) return holds[holds.length - 1].dice_after || turnRec.initial_dice;
        return turnRec.initial_dice;
    },

    // Evaluate every possible next move from the board *before* the current
    // step's placement, from the mover's perspective. Results cached per turn.
    async evaluateMovesCurrent() {
        const { turn, history, record } = Games.replay;
        if (turn <= 0) return;
        const cache = Games.replay.movesCache;
        if (cache[turn] && !cache[turn].error) return;  // done or in flight

        const moveRec = history[turn - 1];
        const player  = moveRec.player;
        const position = Games.buildEvalPosition(turn - 1);  // board before the move
        const dice = Games.turnFinalDice(moveRec);
        cache[turn] = { pending: true, player };
        Games.replay.movesErr = null;
        if (Games.replay.showMoves) Games.renderReplayBody();

        const r = await API.evalGameMoves({
            checkpoint: record.checkpoint, player, dice, position,
        });

        // The modal may have closed or navigated away while we awaited.
        if (!Games.replay.record || Games.replay.record.uuid !== record.uuid) return;
        if (r.error) {
            delete cache[turn];
            Games.replay.movesErr = r.error;
        } else {
            cache[turn] = { moves: r.moves || [], player: r.player };
        }
        if (Games.replay.showMoves && Games.replay.turn === turn) Games.renderReplayBody();
    },

    // Build the move-eval panel and append it below the already-rendered board
    // (renderReplayTurn ran first). Never clears the body.
    renderMoveEvals() {
        const { turn, history, movesCache, movesErr, movesFilter } = Games.replay;
        const body = document.getElementById('games-replay-body');
        const panel = document.createElement('div');
        panel.className = 'moves-panel';

        if (turn <= 0) {
            panel.innerHTML = `<div class="moves-msg muted">Step to a placed move to see its move evaluations.</div>`;
            body.appendChild(panel);
            return;
        }
        if (movesErr) {
            panel.innerHTML = `<div class="moves-msg eval-err">eval failed: ${escapeHtml(movesErr)}</div>`;
            body.appendChild(panel);
            return;
        }
        const entry = movesCache[turn];
        if (!entry || entry.pending) {
            panel.innerHTML = `<div class="moves-msg muted">evaluating all moves…</div>`;
            body.appendChild(panel);
            return;
        }

        const moveRec = history[turn - 1];
        const chosen = moveRec.placement;
        const chosenScore = moveRec.score;
        const validOnly = movesFilter !== 'all';

        // Turbo afterstates (requires_roll) are never valid_for_dice, but we keep
        // them visible even in the "valid after roll" view so the roll-vs-place-
        // in-Turbo decision can be compared against the available placements.
        let moves = entry.moves.slice();
        if (validOnly) moves = moves.filter(m => m.valid_for_dice || m.requires_roll);
        moves.sort((a, b) => b.eval_value - a.eval_value);

        const dice = Games.turnFinalDice(moveRec) || [];
        const diceHtml = dice.map(d => `<span class="replay-die">${d}</span>`).join('');

        let html = `<div class="moves-head">` +
            `<span class="moves-player">P${entry.player} move evals</span> · ` +
            `<span class="muted">${validOnly ? 'valid after roll' : 'all possible'} — ${moves.length} moves</span>` +
            `<div class="moves-dice">roll: ${diceHtml}</div></div>`;

        html += `<table class="moves-table"><thead><tr>` +
            `<th>#</th><th>Column</th><th>Row</th><th>Score</th><th>Eval</th>` +
            (validOnly ? '' : '<th>Valid?</th>') + `<th></th></tr></thead><tbody>`;

        if (moves.length === 0) {
            html += `<tr><td colspan="${validOnly ? 6 : 7}" class="muted" style="text-align:center;">No moves.</td></tr>`;
        }
        moves.forEach((m, i) => {
            const isChosen = m.column === chosen.column && m.row === chosen.row &&
                             m.score === chosenScore;
            const cls = [isChosen ? 'moves-chosen' : '',
                         m.requires_roll ? 'moves-turbo' : ''].filter(Boolean).join(' ');
            const colName = GAMES_COLUMN_NAMES[m.column] ?? m.column_name ?? m.column;
            const colCell = m.requires_roll
                ? `${colName} <span class="moves-roll-tag" title="Only placeable with a roll left">needs roll</span>`
                : colName;
            html += `<tr class="${cls}">` +
                `<td>${i + 1}</td>` +
                `<td>${colCell}</td>` +
                `<td>${GAMES_ROW_NAMES[m.row] ?? m.row_name ?? m.row}</td>` +
                `<td>${m.score === 0 ? 'scratch' : m.score}</td>` +
                `<td>${Number(m.eval_value).toFixed(4)}</td>` +
                (validOnly ? '' : `<td>${m.requires_roll ? '⟳' : (m.valid_for_dice ? '✓' : '')}</td>`) +
                `<td>${isChosen ? '◀ played' : ''}</td>` +
                `</tr>`;
        });
        html += `</tbody></table>`;
        panel.innerHTML = html;
        body.appendChild(panel);
    },

    // ---------------- Accuracy & luck ----------------
    // Sweep every turn in the game once. For human turns, score each reroll and
    // the final placement against the checkpoint's best action and report the
    // average win-probability given up (holds vs placements). For all turns,
    // compute per-turn luck (how the dice deviated from average) and report it
    // split into human vs. computer seats.
    resetAccuracyButton() {
        const btn = document.getElementById('btn-replay-accuracy');
        if (btn) {
            const hasCkpt = !!(Games.replay.record && Games.replay.record.checkpoint);
            btn.disabled = !hasCkpt;
            btn.textContent = hasCkpt ? '📈 Evaluate accuracy & luck' : '📈 No checkpoint';
        }
        const info = document.getElementById('replay-accuracy-info');
        if (info) info.innerHTML = '';
    },

    async evaluateAccuracy() {
        const { record, history } = Games.replay;
        if (!record || !record.checkpoint || !history) return;
        const btn  = document.getElementById('btn-replay-accuracy');
        const info = document.getElementById('replay-accuracy-info');
        const humanSeats = record.human_seats ||
                           (record.final_state && record.final_state.human_seats) || [];

        // Luck applies to every seat, accuracy only to human seats. One batched
        // request scores every turn (one NN pass server-side); we then split
        // per-turn luck into human vs. computer and pool human accuracy deltas.
        if (btn) btn.disabled = true;
        const uuid = record.uuid;
        if (info) info.innerHTML = `<span class="muted">evaluating ${history.length} turns…</span>`;

        const turns = history.map((t, i) => {
            const a = Games.accuracyTurnArgs(i);
            return { player: a.player, position: a.position, initial_dice: a.initial_dice,
                     holds: a.holds, placement: a.placement, score: a.score };
        });
        const resp = await API.accuracyGame({ checkpoint: record.checkpoint, turns });
        // Bail if the modal moved on while awaiting.
        if (!Games.replay.record || Games.replay.record.uuid !== uuid) return;
        if (btn) btn.disabled = false;
        if (resp.error) {
            if (info) info.innerHTML = `<span class="eval-err">eval failed: ${escapeHtml(resp.error)}</span>`;
            return;
        }

        const mean = (a) => a.length ? a.reduce((s, x) => s + x, 0) / a.length : null;
        const holdDeltas = [], placeDeltas = [];
        const humanTurnLucks = [], compTurnLucks = [];
        (resp.turns || []).forEach((r, i) => {
            const isHuman = humanSeats.indexOf(history[i].player) >= 0;
            const turnLuck = mean(r.roll_lucks || []);  // mean of that turn's rolls
            if (turnLuck != null) (isHuman ? humanTurnLucks : compTurnLucks).push(turnLuck);
            if (isHuman) {
                (r.hold_deltas || []).forEach(d => holdDeltas.push(d));
                if (r.has_place) placeDeltas.push(r.place_delta);
            }
            // Seed the per-move cache so stepping to this move is instant.
            Games.replay.moveAccCache[i + 1] = {
                player: r.player,
                holdDeltas: r.hold_deltas || [],
                placeDelta: r.has_place ? r.place_delta : null,
                rollLucks: r.roll_lucks || [],
            };
        });

        Games.renderAccuracy(holdDeltas, placeDeltas,
                             mean(humanTurnLucks), mean(compTurnLucks), 0);
        // Refresh the current turn view so its luck shows without a manual step.
        Games.renderReplayBody();
    },

    renderAccuracy(holdDeltas, placeDeltas, humanLuck, compLuck, failed) {
        const info = document.getElementById('replay-accuracy-info');
        if (!info) return;
        const mean = (a) => a.length ? a.reduce((s, x) => s + x, 0) / a.length : null;
        // Deltas are win-probability lost (0..1); show as percentage points.
        const pp = (x) => x == null ? '—' : (100 * x).toFixed(2) + 'pp';
        // Backend luck is a mean-zero -100..+100 index; display it remapped to a
        // 0..100% scale where 50% = rolled as expected, 100% = luckiest possible.
        const luck = (x) => x == null ? '—' : ((x + 100) / 2).toFixed(1) + '%';
        const all = holdDeltas.concat(placeDeltas);
        const overall = mean(all), holds = mean(holdDeltas), places = mean(placeDeltas);
        const warn = failed ? ` <span class="eval-err">(${failed} turn(s) failed)</span>` : '';
        const accLine = holdDeltas.length || placeDeltas.length
            ? `<div><span class="acc-label">Avg win% lost vs. best</span> — ` +
              `<span class="acc-val">overall ${pp(overall)}</span> · ` +
              `holds ${pp(holds)} <span class="muted">(${holdDeltas.length})</span> · ` +
              `placements ${pp(places)} <span class="muted">(${placeDeltas.length})</span>` +
              `<span class="muted"> · lower is better</span></div>`
            : `<div><span class="muted">No human turns to score for accuracy.</span></div>`;
        info.innerHTML = accLine +
            `<div title="How the dice deviated from expectation for each player, ` +
            `scaled by each roll's full range. 50% = rolled as expected, ` +
            `above 50% = luckier than expected, below 50% = unluckier.">` +
            `<span class="acc-label">Luck</span> — ` +
            `human <span class="acc-val">${luck(humanLuck)}</span> · ` +
            `computer <span class="acc-val">${luck(compLuck)}</span>` +
            `<span class="muted"> · 50% = as expected</span></div>${warn}`;
    },

    // Dispatch the modal body. The score breakdown replaces the board; the
    // move-eval table is additive and renders *below* the board so the play
    // tables stay visible while you read the evaluations.
    renderReplayBody() {
        const { showBreakdown, showMoves } = Games.replay;
        const bbtn = document.getElementById('btn-replay-breakdown');
        if (bbtn) bbtn.textContent = showBreakdown ? '◀ Back to replay' : '📊 Score breakdown';
        const mbtn = document.getElementById('btn-replay-moves');
        if (mbtn && !mbtn.disabled) mbtn.classList.toggle('active', showMoves);
        const filt = document.getElementById('replay-moves-filter');
        if (filt) filt.hidden = !(showMoves && !showBreakdown);

        if (showBreakdown) { Games.renderBreakdown(); return; }
        Games.renderReplayTurn();
        if (showMoves) Games.renderMoveEvals();  // appended below the board
    },

    renderReplayMeta() {
        const r = Games.replay.record;
        const fs = r.final_state || {};
        const hr = humanResult(r);
        const hm = humanMargin(r);
        const coeffs = (r.coefficients || fs.coefficients || []);
        const flagLine = r.flagged
            ? `<div class="games-flag-meta">🚩 Flagged by player — “AI played weird moves this game”` +
              `${r.flag_note && r.flag_note !== 'ai_weird_moves' ? ' · ' + escapeHtml(r.flag_note) : ''}</div>`
            : '';
        document.getElementById('games-replay-meta').innerHTML =
            flagLine +
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
            let line = `Move ${turn}/${states.length - 1} · P${justTurn.player} → ` +
                `${rowName} in ${colName} = ${sc === 0 ? 'scratch' : sc}`;
            // Append this turn's luck if it's been evaluated (e.g. via the
            // "Evaluate accuracy & luck" button). 50% = rolled as expected.
            const cached = Games.replay.moveAccCache[turn];
            if (cached && cached.rollLucks && cached.rollLucks.length) {
                const lk = cached.rollLucks.reduce((s, x) => s + x, 0) / cached.rollLucks.length;
                line += ` · luck ${((lk + 100) / 2).toFixed(1)}%`;
            }
            info.textContent = line;
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
