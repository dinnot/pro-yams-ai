// Tournament tab — participant management, polling, win-rate matrix.

const Tournament = (() => {
    const POLL_MS = 1000;

    let participants = [];   // [{id, name, type, path}]
    let nextLocalId = 1;
    let availableModels = []; // [{path, filename, display_name}]
    let pollHandle = null;

    function el(id) { return document.getElementById(id); }

    async function refreshModels() {
        try {
            const r = await API.listModels();
            availableModels = r.models || [];
        } catch (e) {
            availableModels = [];
        }
        const sel = el('tour-part-ckpt');
        sel.innerHTML = '';
        if (availableModels.length === 0) {
            const opt = document.createElement('option');
            opt.value = '';
            opt.textContent = '(no checkpoints)';
            sel.appendChild(opt);
            return;
        }
        for (const m of availableModels) {
            const opt = document.createElement('option');
            opt.value = m.path;
            opt.textContent = m.display_name;
            sel.appendChild(opt);
        }
    }

    function onTypeChange() {
        const t = el('tour-part-type').value;
        el('tour-part-ckpt').style.display = (t === 'kNN') ? '' : 'none';
        // Suggest a default name if empty.
        const nameInput = el('tour-part-name');
        if (!nameInput.value) {
            if (t === 'kHeuristicV1') nameInput.placeholder = 'Heuristic V1';
            else if (t === 'kHeuristicV2') nameInput.placeholder = 'Heuristic V2';
            else if (t === 'kHeuristicV3') nameInput.placeholder = 'Heuristic V3';
            else if (t && t.startsWith('kHeuristicV')) nameInput.placeholder = 'Heuristic ' + t.substr('kHeuristic'.length);
            else nameInput.placeholder = 'NN';
        }
    }

    function defaultName(type, path) {
        if (type === 'kHeuristicV1') return 'Heuristic V1';
        if (type === 'kHeuristicV2') return 'Heuristic V2';
        if (type === 'kHeuristicV3') return 'Heuristic V3';
        if (type && type.startsWith('kHeuristicV')) return 'Heuristic ' + type.substr('kHeuristic'.length);
        if (path) {
            const fn = path.split('/').pop() || path;
            return fn;
        }
        return 'NN';
    }

    function addParticipant() {
        const type = el('tour-part-type').value;
        const path = (type === 'kNN') ? el('tour-part-ckpt').value : '';
        if (type === 'kNN' && !path) {
            alert('No checkpoint selected (or none available).');
            return;
        }
        const userName = el('tour-part-name').value.trim();
        const name = userName || defaultName(type, path);
        const id = `p${nextLocalId++}_${type}`;
        participants.push({ id, name, type, path });
        el('tour-part-name').value = '';
        renderParticipants();
    }

    function removeParticipant(id) {
        participants = participants.filter(p => p.id !== id);
        renderParticipants();
    }

    function renderParticipants() {
        const ul = el('tour-participants-list');
        ul.innerHTML = '';
        if (participants.length === 0) {
            const li = document.createElement('li');
            li.className = 'tour-empty';
            li.textContent = 'No participants yet — add at least 2 bots.';
            ul.appendChild(li);
            return;
        }
        for (const p of participants) {
            const li = document.createElement('li');
            const nameSpan = document.createElement('span');
            nameSpan.className = 'tour-pname';
            nameSpan.textContent = p.name;
            const typeSpan = document.createElement('span');
            typeSpan.className = 'tour-ptype';
            typeSpan.textContent = ` [${p.type}${p.path ? ': ' + (p.path.split('/').pop()) : ''}]`;
            const btn = document.createElement('button');
            btn.textContent = '✕';
            btn.title = 'Remove';
            btn.className = 'tour-remove';
            btn.addEventListener('click', () => removeParticipant(p.id));
            li.appendChild(nameSpan);
            li.appendChild(typeSpan);
            li.appendChild(btn);
            ul.appendChild(li);
        }
    }

    async function startTournament() {
        if (participants.length < 2) {
            alert('Need at least 2 participants.');
            return;
        }
        const games = parseInt(el('tour-games').value, 10) || 100;
        if (games <= 0) {
            alert('Games per matchup must be positive.');
            return;
        }
        el('tour-error').style.display = 'none';
        try {
            const r = await API.startTournament(participants, games);
            if (r && r.error) {
                showError(r.error);
                return;
            }
            startPolling();
        } catch (e) {
            showError(String(e));
        }
    }

    async function stopTournament() {
        try { await API.stopTournament(); } catch (_) {}
    }

    function showError(msg) {
        const e = el('tour-error');
        e.textContent = msg;
        e.style.display = '';
    }

    function startPolling() {
        if (pollHandle) clearInterval(pollHandle);
        pollHandle = setInterval(poll, POLL_MS);
        poll();
    }

    function stopPolling() {
        if (pollHandle) {
            clearInterval(pollHandle);
            pollHandle = null;
        }
    }

    async function poll() {
        let st;
        try { st = await API.getTournamentStatus(); }
        catch (e) { return; }
        renderStatus(st);
        if (!st.is_running) {
            stopPolling();
        }
    }

    // Fit ELO ratings via Bradley-Terry MM iterations on the pairwise grid.
    // Returns a map id -> elo, anchored to mean 1500. Unplayed participants get 1500.
    function computeElos(participants, grid) {
        const ids = participants.map(p => p.id);
        const n = ids.length;
        const elos = {};
        if (n === 0) return elos;
        if (n === 1) { elos[ids[0]] = 1500; return elos; }

        const wins = {};            // wins + 0.5 * draws across all opponents
        const pairs = {};           // pairs[a][b] = games played a-vs-b
        for (const id of ids) { wins[id] = 0; pairs[id] = {}; }
        for (const a of participants) {
            for (const b of participants) {
                if (a.id === b.id) continue;
                const m = (grid[a.id] || {})[b.id];
                if (!m || m.games === 0) continue;
                wins[a.id] += m.wins_a + 0.5 * m.draws;
                pairs[a.id][b.id] = m.games;
            }
        }

        // Pseudo-count regularizer: pretend each participant played 1 game (half win,
        // half loss) vs a virtual opponent of strength 1. Keeps ratings finite when a
        // bot has 0% or 100% so far.
        const REG = 0.5;

        let strength = {};
        for (const id of ids) strength[id] = 1.0;

        const STRENGTH_MIN = 1e-4, STRENGTH_MAX = 1e4;

        for (let it = 0; it < 300; it++) {
            const next = {};
            for (const a of ids) {
                let denom = REG / (strength[a] + 1.0);
                for (const b of ids) {
                    if (a === b) continue;
                    const g = pairs[a][b] || 0;
                    if (g === 0) continue;
                    denom += g / (strength[a] + strength[b]);
                }
                const numer = wins[a] + REG;
                let s = (denom > 0) ? numer / denom : strength[a];
                if (s < STRENGTH_MIN) s = STRENGTH_MIN;
                if (s > STRENGTH_MAX) s = STRENGTH_MAX;
                next[a] = s;
            }
            // Normalize so geometric mean of strengths = 1.
            let logSum = 0;
            for (const id of ids) logSum += Math.log(next[id]);
            const norm = Math.exp(logSum / n);
            for (const id of ids) next[id] = next[id] / norm;
            strength = next;
        }

        for (const id of ids) elos[id] = 400 * Math.log10(strength[id]);
        let sum = 0; for (const id of ids) sum += elos[id];
        const off = 1500 - sum / n;
        for (const id of ids) elos[id] += off;
        return elos;
    }

    // Per-bot totals across all opponents: games, W-L-D, overall win rate,
    // total average margin, and fitted ELO.
    function computeBotStats(participants, grid) {
        const stats = {};
        for (const p of participants) {
            stats[p.id] = {
                id: p.id, name: p.name, type: p.type, path: p.path,
                games: 0, wins: 0, losses: 0, draws: 0, sum_margin: 0,
            };
        }
        for (const a of participants) {
            for (const b of participants) {
                if (a.id === b.id) continue;
                const m = (grid[a.id] || {})[b.id];
                if (!m || m.games === 0) continue;
                const s = stats[a.id];
                s.games  += m.games;
                s.wins   += m.wins_a;
                s.losses += m.wins_b;
                s.draws  += m.draws;
                s.sum_margin += (m.avg_margin_a || 0) * m.games;
            }
        }
        const elos = computeElos(participants, grid);
        for (const p of participants) {
            const s = stats[p.id];
            s.elo = elos[p.id];
            s.win_rate = s.games > 0 ? (s.wins + 0.5 * s.draws) / s.games : null;
            s.avg_margin = s.games > 0 ? s.sum_margin / s.games : 0;
        }
        return stats;
    }

    function renderSummary(participants, grid) {
        const table = el('tour-summary');
        table.innerHTML = '';
        if (!participants || participants.length === 0) return;

        const stats = computeBotStats(participants, grid);
        const rows = participants.map(p => stats[p.id]);
        // Sort: bots with games first (by ELO desc); then unplayed by name.
        rows.sort((x, y) => {
            if ((x.games > 0) !== (y.games > 0)) return x.games > 0 ? -1 : 1;
            if (x.games === 0 && y.games === 0) return x.name.localeCompare(y.name);
            return y.elo - x.elo;
        });

        const thead = document.createElement('thead');
        const hr = document.createElement('tr');
        for (const label of ['#', 'Bot', 'Games', 'W-L-D', 'Win %', 'Avg Margin', 'ELO']) {
            const th = document.createElement('th');
            th.textContent = label;
            hr.appendChild(th);
        }
        thead.appendChild(hr);
        table.appendChild(thead);

        const tbody = document.createElement('tbody');
        rows.forEach((s, idx) => {
            const tr = document.createElement('tr');

            const rank = document.createElement('td');
            rank.textContent = s.games > 0 ? String(idx + 1) : '—';
            tr.appendChild(rank);

            const name = document.createElement('td');
            name.textContent = s.name;
            name.title = s.type + (s.path ? ` (${s.path})` : '');
            name.className = 'tour-sum-name';
            tr.appendChild(name);

            const games = document.createElement('td');
            games.textContent = String(s.games);
            tr.appendChild(games);

            const wld = document.createElement('td');
            wld.textContent = `${s.wins}-${s.losses}-${s.draws}`;
            tr.appendChild(wld);

            const wr = document.createElement('td');
            if (s.win_rate === null) {
                wr.textContent = '—';
            } else {
                wr.textContent = (s.win_rate * 100).toFixed(1) + '%';
                wr.style.background = colorForWinRate(s.win_rate);
            }
            tr.appendChild(wr);

            const am = document.createElement('td');
            if (s.games === 0) {
                am.textContent = '—';
            } else {
                const sign = s.avg_margin >= 0 ? '+' : '';
                am.textContent = sign + s.avg_margin.toFixed(1);
                am.className = s.avg_margin >= 0 ? 'tour-margin-pos' : 'tour-margin-neg';
            }
            tr.appendChild(am);

            const elo = document.createElement('td');
            elo.textContent = s.games > 0 ? Math.round(s.elo).toString() : '—';
            elo.className = 'tour-elo';
            tr.appendChild(elo);

            tbody.appendChild(tr);
        });
        table.appendChild(tbody);
    }

    function colorForWinRate(wr) {
        if (wr === null || wr === undefined || Number.isNaN(wr)) return '#1a1f2e';
        // Red (0) → grey (0.5) → green (1).
        const x = Math.max(0, Math.min(1, wr));
        const r = Math.round(160 * (1 - x) + 50 * x);
        const g = Math.round(50  * (1 - x) + 160 * x);
        const b = 60;
        return `rgb(${r}, ${g}, ${b})`;
    }

    function renderStatus(st) {
        // Progress bar
        const totalDone = st.games_completed || 0;
        const total = st.total_games || 0;
        const prog = el('tour-progress');
        const bar = el('tour-progress-bar');
        const txt = el('tour-progress-text');
        if (total > 0) {
            prog.style.display = '';
            bar.value = totalDone;
            bar.max = total;
            const pct = total > 0 ? ((totalDone / total) * 100).toFixed(1) : 0;
            txt.textContent = `${totalDone} / ${total} (${pct}%)${st.is_running ? '' : ' — done'}`;
        } else {
            prog.style.display = 'none';
        }

        el('btn-tour-start').disabled = !!st.is_running;
        el('btn-tour-stop').disabled = !st.is_running;

        if (st.error) showError(st.error);

        // Per-bot summary (ELO + total avg margin) and head-to-head matrix.
        const ps = st.participants || [];
        const grid = st.grid || {};
        renderSummary(ps, grid);

        const table = el('tour-matrix');
        table.innerHTML = '';

        if (ps.length === 0) return;

        // Header row
        const thead = document.createElement('thead');
        const hr = document.createElement('tr');
        hr.appendChild(document.createElement('th'));
        for (const b of ps) {
            const th = document.createElement('th');
            th.textContent = b.name;
            th.title = b.type + (b.path ? ` (${b.path})` : '');
            hr.appendChild(th);
        }
        thead.appendChild(hr);
        table.appendChild(thead);

        // Rows
        const tbody = document.createElement('tbody');
        for (const a of ps) {
            const tr = document.createElement('tr');
            const rh = document.createElement('th');
            rh.textContent = a.name;
            rh.title = a.type + (a.path ? ` (${a.path})` : '');
            tr.appendChild(rh);

            for (const b of ps) {
                const td = document.createElement('td');
                if (a.id === b.id) {
                    td.className = 'tour-diag';
                    td.textContent = '—';
                } else {
                    const m = (grid[a.id] || {})[b.id];
                    if (!m || m.games === 0) {
                        td.className = 'tour-empty-cell';
                        td.textContent = '·';
                    } else {
                        const wr = (m.wins_a + 0.5 * m.draws) / m.games;
                        td.style.background = colorForWinRate(wr);
                        const pctStr = (wr * 100).toFixed(1) + '%';
                        const marginStr = (m.avg_margin_a >= 0 ? '+' : '') +
                            m.avg_margin_a.toFixed(0);
                        td.innerHTML = `<div class="tour-wr">${pctStr}</div>` +
                                       `<div class="tour-margin">[${marginStr}]</div>` +
                                       `<div class="tour-games">${m.wins_a}-${m.wins_b}-${m.draws}</div>`;
                        td.title = `${a.name} vs ${b.name}: ${m.wins_a}W / ${m.wins_b}L / ${m.draws}D over ${m.games} games`;
                    }
                }
                tr.appendChild(td);
            }
            tbody.appendChild(tr);
        }
        table.appendChild(tbody);
    }

    function init() {
        el('tour-part-type').addEventListener('change', onTypeChange);
        el('btn-tour-add').addEventListener('click', addParticipant);
        el('btn-tour-refresh-models').addEventListener('click', refreshModels);
        el('btn-tour-start').addEventListener('click', startTournament);
        el('btn-tour-stop').addEventListener('click', stopTournament);

        renderParticipants();
        refreshModels();
        // Probe initial status (in case a tournament is already running).
        poll();
    }

    return { init };
})();
