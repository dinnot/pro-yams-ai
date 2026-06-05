// Pro Yams Play — human vs. NN.
//
// 1v1: one combined sheet — current player's board as main, opponent's
//      values stacked at the top of each cell.
// 2v2: same single-sheet trick, but the top of each cell splits into
//      three sub-cells — left = player who rolled previously, middle =
//      current player's teammate, right = player who plays next. The
//      human controls seats 0 AND 2 (both of Team 0); seats 1 and 3
//      default to NN. computeDuel and the endgame breakdown iterate
//      every cross-team pairing the same way compute_duel<Yams2v2>
//      does in src/engine/duel.cc.

const COLUMN_NAMES = ['Down', 'Free', 'Up', 'Mid', 'Turbo', 'UpDown'];
const COLUMN_KEYS  = ['down',  'free', 'up', 'mid', 'turbo', 'updown'];
const ROW_NAMES    = ['1s','2s','3s','4s','5s','6s','SS','LS','FH','K','STR','U8','Y'];

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Animation timing (ms). Tweak to taste.
const T = {
    preRollPause:    1400,
    rollSpin:         500,
    nnHoldHighlight: 1000,
    nnAfterRoll:      450,
    nnBeforePlace:    900,
    placeFlash:       700,
    afterPlace:       800,
};

// Compressed timing for shared-game "catch-up" — a spectator replaying turns
// that already happened (the partner's placement, intervening NN turns). Using
// the full T here makes the lagging player fall further behind with each turn;
// these snappier values keep them close to real time.
const T_FAST = {
    preRollPause:    280,
    rollSpin:        240,
    nnHoldHighlight: 340,
    nnAfterRoll:     140,
    nnBeforePlace:   160,
    placeFlash:      320,
    afterPlace:      220,
};

// Team membership: in 1v1 each player is its own (singleton) team. In 2v2
// P0/P2 are Team 0 and P1/P3 are Team 1 — matches Yams2v2 in C++.
function teamOf(player, numPlayers) {
    return numPlayers === 4 ? (player & 1) : player;
}
function areTeammates(p, q, numPlayers) {
    return teamOf(p, numPlayers) === teamOf(q, numPlayers);
}
function variantFromState(gs) {
    if (!gs) return '1v1';
    if (gs.game_variant) return gs.game_variant;
    const n = gs.num_players || (gs.player_types ? gs.player_types.length : 2);
    return n === 4 ? '2v2' : '1v1';
}

const Play = {
    sessionId: null,
    serverState: null,
    displayBoards: null,
    displayDice: null,
    nnHeldMask: 0,
    nnRollingFlags: null,
    holdMask: 0,
    humanSeats: [0],
    humanTeam: 0,
    numPlayers: 2,
    variant: '1v1',
    serverVariant: '1v1',
    lastMode: null,

    // --- Shared multiplayer ("Play with a friend": two humans vs NN) ---
    // sharedMode is on for a matchmade game where this browser owns exactly one
    // human seat (mySeat); partnerSeat is the other human seat on my team. The
    // game is driven by sharedLoop() polling the authoritative server state
    // rather than the single-client driveLoop().
    sharedMode: false,
    mySeat: 0,
    partnerSeat: 2,
    lastVersion: null,   // last shared-game state version seen (for long-poll)
    mmTicket: null,
    mmCancelled: false,
    takeoverShown: false,
    _sharedLooping: false,
    _heartbeatTimer: null,
    // Tracks the in-progress turn we're spectating (the other human's), so we
    // animate the roll/rerolls live and don't re-animate them when the finished
    // turn lands in history. { player, initial, shownRerolls } or null.
    liveSpec: null,
    _holdPreviewTimer: null,
    // One-shot: make the first shared diff replay history from turn 0 so opening
    // NN moves animate for both players regardless of who triggered them.
    _animateFromStart: false,
    // Server-Sent Events: the primary push channel while watching (not my turn).
    // _sseQueue holds the latest pushed state (coalesced); _sseDraining
    // serializes processing so an in-flight animation finishes first.
    _sse: null,
    _sseQueue: null,
    _sseDraining: false,
    // Seats whose human was replaced by the NN this game (from the server's
    // nn_takeover_seats). Used so a taken-over teammate's turns are labelled and
    // treated as AI rather than as a live human.
    takenOverSeats: [],
    _takeoverPending: false,
    options: null,
    animating: false,
    justPlaced: null,
    hasShownDice: false,
    animationPlayer: null,

    // --- Connection / fault tolerance ---
    // The server is authoritative; on any failed request we re-sync from it
    // (GET /api/game/:id) and resume rather than letting the animation chain
    // die. needsRecovery latches a pending recovery; recovering guards against
    // overlapping recovery attempts; _pendingStartMode remembers a game that
    // failed to start so it can be retried once the connection returns.
    needsRecovery: false,
    recovering: false,
    _pendingStartMode: null,
    _retryTimer: null,

    isHuman(p) { return this.humanSeats.indexOf(p) >= 0; },
    isHumanTeam(p) { return teamOf(p, this.numPlayers) === this.humanTeam; },

    init() {
        document.getElementById('btn-play')
            .addEventListener('click', () => Play.start('1v1'));
        document.getElementById('btn-play-friend')
            .addEventListener('click', () => Play.startFriend());
        document.getElementById('btn-play-team')
            .addEventListener('click', () => Play.start('2v2-team'));
        document.getElementById('btn-play-solo')
            .addEventListener('click', () => Play.start('2v2-solo'));
        document.getElementById('btn-mm-cancel')
            .addEventListener('click', () => Play.cancelMatchmaking());
        document.getElementById('btn-takeover')
            .addEventListener('click', () => Play.requestTakeover());
        document.getElementById('btn-again')
            .addEventListener('click', () => {
                if (Play.lastMode === '2v2-friend') Play.startFriend();
                else Play.start(Play.lastMode || '1v1');
            });
        document.getElementById('btn-flag')
            .addEventListener('click', () => Play.flagGame());
        document.getElementById('btn-reroll').addEventListener('click', () => Play.humanReroll());
        document.getElementById('header').addEventListener('click', () => Play.toSplash());

        // Reconnect opportunities: returning from another app (mobile),
        // regaining focus, or the network coming back. Mobile browsers throttle
        // timers and abort in-flight fetches for backgrounded tabs, so these
        // events are the reliable signal that it's worth retrying.
        document.addEventListener('visibilitychange', () => {
            if (document.visibilityState === 'visible') Play.onReconnectOpportunity();
        });
        window.addEventListener('online', () => Play.onReconnectOpportunity());
        window.addEventListener('focus',  () => Play.onReconnectOpportunity());

        this.configureSplash();
    },

    // The server's variant is fixed at launch — probe once and show the
    // matching set of play buttons.
    async configureSplash() {
        let info;
        try { info = await API.info(); } catch (_) { info = {}; }
        this.serverVariant = info.game_variant ||
            (info.num_players === 4 ? '2v2' : '1v1');
        const is2v2 = this.serverVariant === '2v2';
        document.getElementById('btn-play').hidden        = is2v2;
        document.getElementById('btn-play-friend').hidden = !is2v2;
        document.getElementById('btn-play-team').hidden   = !is2v2;
        document.getElementById('btn-play-solo').hidden   = !is2v2;
    },

    toSplash() {
        // Leaving a shared game: don't delete the session (that would end it for
        // the partner) — just stop participating; the server hands my seat to
        // the NN after the heartbeat timeout. For single-client games, delete.
        if (this.mmTicket) { API.matchmakeCancel(this.mmTicket); this.mmTicket = null; }
        this.mmCancelled = true;
        this.stopHeartbeat();
        this.stopSharedEvents();
        if (this.sessionId && !this.sharedMode) { API.deleteGame(this.sessionId); }
        this.sessionId = null;
        this.sharedMode = false;
        this._sharedLooping = false;
        this.liveSpec = null;
        this._takeoverPending = false;
        this.lastVersion = null;
        this._animateFromStart = false;
        if (this._holdPreviewTimer) { clearTimeout(this._holdPreviewTimer); this._holdPreviewTimer = null; }
        document.getElementById('matchmaking').hidden = true;
        document.getElementById('disconnect-prompt').hidden = true;
        document.getElementById('takeover-banner').hidden = true;
        this._pendingStartMode = null;
        this.clearRecovery();
        this.serverState = null;
        this.displayBoards = null;
        this.displayDice = null;
        this.nnHeldMask = 0;
        this.nnRollingFlags = null;
        this.holdMask = 0;
        this.options = null;
        this.animating = false;
        this.justPlaced = null;
        this.hasShownDice = false;
        this.animationPlayer = null;
        document.getElementById('splash').hidden = false;
        document.getElementById('game').hidden = true;
        document.getElementById('endgame').hidden = true;
        document.getElementById('footer').hidden = true;
        document.getElementById('sheet').hidden = false;
        document.body.classList.remove('cur-p0', 'cur-p1', 'cur-p2', 'cur-p3');
        this.setTurnInfo('');
    },

    async start(mode) {
        // start() drives the single-client modes (1v1 / whole-team / NN-teammate).
        // Ensure any prior shared game is torn down first.
        this.stopHeartbeat();
        this.stopSharedEvents();
        this.sharedMode = false;
        document.getElementById('matchmaking').hidden = true;
        if (this.sessionId) { API.deleteGame(this.sessionId); this.sessionId = null; }
        this.holdMask = 0;
        this.nnHeldMask = 0;
        this.nnRollingFlags = null;
        this.animating = false;
        this.justPlaced = null;
        this.options = null;
        this.hasShownDice = false;
        this.displayDice = [0, 0, 0, 0, 0];
        this.clearRecovery();

        // Make sure we know the server's variant — configureSplash() runs
        // it in init() but if start() is called before that finishes (or
        // the probe failed), do it now.
        if (!this.serverVariant) {
            try {
                const info = await API.info();
                this.serverVariant = info.game_variant ||
                    (info.num_players === 4 ? '2v2' : '1v1');
            } catch (_) { this.serverVariant = '1v1'; }
        }
        // Coerce the requested mode to one that the server can host.
        if (this.serverVariant === '1v1') mode = '1v1';
        else if (mode === '1v1') mode = '2v2-team';
        this.lastMode = mode;

        const playerTypes = this.configureSeatsForMode(mode);
        document.body.dataset.variant = this.variant;

        document.getElementById('splash').hidden = true;
        document.getElementById('endgame').hidden = true;
        document.getElementById('game').hidden = false;
        document.getElementById('footer').hidden = false;
        document.getElementById('sheet').hidden = false;

        // Remember the mode so recovery can retry game creation if the very
        // first request fails (no session to re-sync to yet).
        this._pendingStartMode = mode;
        try {
            const data = await API.newGame(playerTypes);
            if (data && data.error) { this.setTurnInfo('Error: ' + data.error); return; }
            this.sessionId    = data.session_id;
            this.serverState  = data.game_state;
            // Reconcile with server-reported variant in case it disagrees.
            this.numPlayers = this.serverState.num_players || this.numPlayers;
            this.variant    = variantFromState(this.serverState);
            document.body.dataset.variant = this.variant;
            this.displayBoards = cloneBoards(this.serverState.boards);

            this.renderAll();
            await this.driveLoop();

            // Reached a stable point (human turn / game over) without error.
            this._pendingStartMode = null;
            this.clearRecovery();
        } catch (e) {
            this.handleFailure(e);
        }
    },

    // Translate a mode label into seat assignments + the playerTypes
    // payload to send to the server. Updates state in place and returns
    // the playerTypes array.
    configureSeatsForMode(mode) {
        if (mode === '1v1') {
            const humanIsP0 = Math.random() < 0.5;
            this.humanSeats = humanIsP0 ? [0] : [1];
            this.humanTeam  = humanIsP0 ? 0 : 1;
            this.numPlayers = 2;
            this.variant    = '1v1';
            return humanIsP0 ? ['human', 'nn'] : ['nn', 'human'];
        }
        this.numPlayers = 4;
        this.variant    = '2v2';
        if (mode === '2v2-team') {
            // Random Team 0 vs Team 1 — the human controls both teammates.
            const team = Math.random() < 0.5 ? 0 : 1;
            this.humanSeats = team === 0 ? [0, 2] : [1, 3];
            this.humanTeam  = team;
            return team === 0
                ? ['human', 'nn', 'human', 'nn']
                : ['nn', 'human', 'nn', 'human'];
        }
        // '2v2-solo': human takes one of the four seats; the other three
        // (including the human's own teammate) are NN.
        const seat = Math.floor(Math.random() * 4);
        this.humanSeats = [seat];
        this.humanTeam  = seat & 1;
        const pts = ['nn', 'nn', 'nn', 'nn'];
        pts[seat] = 'human';
        return pts;
    },

    // =================================================================
    // "Play with a friend" — two humans share one session vs NN.
    //
    // Matchmaking: POST /api/matchmake/join either pairs us with a waiting
    // player immediately or returns a ticket we poll until a partner joins.
    // Once paired, enterSharedGame() runs a poll-driven loop (sharedLoop) so
    // each browser acts only on its own seat and spectates everything else.
    // =================================================================

    showMatchmaking(on) {
        const el = document.getElementById('matchmaking');
        if (el) el.hidden = !on;
    },

    async startFriend() {
        // Tear down any prior game/loop and reset matchmaking flags.
        this.stopHeartbeat();
        this.stopSharedEvents();
        if (this.sessionId && !this.sharedMode) { API.deleteGame(this.sessionId); }
        this.sessionId   = null;
        this.sharedMode  = false;
        this.mmCancelled = false;
        this.mmTicket    = null;
        this.clearRecovery();

        document.getElementById('splash').hidden   = true;
        document.getElementById('endgame').hidden  = true;
        document.getElementById('game').hidden     = true;
        document.getElementById('footer').hidden   = true;
        this.showMatchmaking(true);

        let resp;
        try { resp = await API.matchmakeJoin(); }
        catch (e) { this.showMatchmaking(false); this.toSplash(); return; }
        if (this.mmCancelled) return;

        if (resp && resp.status === 'matched') {
            this.showMatchmaking(false);
            await this.enterSharedGame(resp.session_id, resp.seat);
            return;
        }
        if (!resp || resp.status !== 'waiting') {
            this.showMatchmaking(false); this.toSplash(); return;
        }

        this.mmTicket = resp.ticket;
        // Poll our ticket until a partner joins (or the user cancels).
        while (!this.mmCancelled) {
            await sleep(1500);
            if (this.mmCancelled) return;
            let p;
            try { p = await API.matchmakePoll(this.mmTicket); }
            catch (e) { continue; }  // transient — keep polling
            if (this.mmCancelled) return;
            if (p && p.status === 'matched') {
                this.mmTicket = null;
                this.showMatchmaking(false);
                await this.enterSharedGame(p.session_id, p.seat);
                return;
            }
            if (p && p.status === 'expired') {
                // Ticket gone (TTL / server restart) — rejoin the queue.
                try {
                    const r = await API.matchmakeJoin();
                    if (this.mmCancelled) return;
                    if (r && r.status === 'matched') {
                        this.showMatchmaking(false);
                        await this.enterSharedGame(r.session_id, r.seat);
                        return;
                    }
                    if (r && r.status === 'waiting') this.mmTicket = r.ticket;
                } catch (_) { /* keep trying */ }
            }
        }
    },

    cancelMatchmaking() {
        this.mmCancelled = true;
        if (this.mmTicket) { API.matchmakeCancel(this.mmTicket); this.mmTicket = null; }
        this.showMatchmaking(false);
        this.toSplash();
    },

    async enterSharedGame(id, seat) {
        this.sharedMode    = true;
        this.sessionId     = id;
        this.mySeat        = seat;
        this.partnerSeat   = (seat + 2) % 4;   // the other seat on my team
        this.humanTeam     = teamOf(seat, 4);
        this.numPlayers    = 4;
        this.variant       = '2v2';
        this.lastMode      = '2v2-friend';
        this.humanSeats    = [this.mySeat, this.partnerSeat];
        this.takeoverShown   = false;
        this.liveSpec        = null;
        this.takenOverSeats  = [];
        this._takeoverPending = false;
        this.lastVersion     = null;
        document.getElementById('disconnect-prompt').hidden = true;

        this.holdMask        = 0;
        this.nnHeldMask      = 0;
        this.nnRollingFlags  = null;
        this.animating       = false;
        this.justPlaced      = null;
        this.options         = null;
        this.hasShownDice    = false;
        this.displayDice     = [0, 0, 0, 0, 0];
        this.animationPlayer = null;
        this.clearRecovery();

        document.body.dataset.variant = '2v2';
        document.getElementById('splash').hidden     = true;
        document.getElementById('endgame').hidden    = true;
        document.getElementById('matchmaking').hidden = true;
        document.getElementById('game').hidden       = false;
        document.getElementById('footer').hidden     = false;
        document.getElementById('sheet').hidden      = false;

        let data;
        try { data = await API.getGame(id); }
        catch (e) { this.handleFailure(e); return; }
        if (data && data.error) { this.setTurnInfo('Error: ' + data.error); return; }

        // Show the game's opening BEFORE any NN moves: hide the opening NN
        // placements and flag the first diff to animate from turn 0. Otherwise
        // whichever human connected after the AI already moved would see it
        // already done — its baseline state would include the placement, so the
        // diff is empty. Safe at fresh entry: no human has acted yet, so the
        // whole current history is opening NN turns that should animate.
        const openingBoards = cloneBoards(data.boards);
        for (const turn of (data.history || [])) {
            const pb = openingBoards['player' + turn.player];
            const col = pb && pb[COLUMN_KEYS[turn.placement.column]];
            if (col) col[ROW_NAMES[turn.placement.row]] = -1;  // hide
        }
        this.serverState     = data;
        this.lastVersion     = (data.version !== undefined) ? data.version : null;
        this.numPlayers      = data.num_players || 4;
        this.variant         = variantFromState(data);
        this.displayBoards   = openingBoards;
        this._animateFromStart = true;  // first applySharedDiff replays from 0
        this.renderAll();

        this.startHeartbeat();
        this.startSharedEvents();
    },

    // Keep our seat marked alive on the server so we aren't handed to the NN
    // while deliberating (or between polls). Runs for the whole shared game.
    startHeartbeat() {
        this.stopHeartbeat();
        if (!this.sessionId) return;
        API.heartbeat(this.sessionId, this.mySeat);
        this._heartbeatTimer = setInterval(() => {
            if (this.sessionId && this.sharedMode) API.heartbeat(this.sessionId, this.mySeat);
        }, 5000);
    },
    stopHeartbeat() {
        if (this._heartbeatTimer) { clearInterval(this._heartbeatTimer); this._heartbeatTimer = null; }
    },

    // Poll-driven driver for a shared game. Exits (handing control to the user)
    // when it becomes my turn; humanPlace() restarts it after the placement
    // chain. Tolerant of transient failures: a failed poll just retries.
    async sharedLoop() {
        if (this._sharedLooping) return;
        this._sharedLooping = true;
        try {
            // Process the state we ALREADY have first, then block for the next
            // change. (Long-polling first would stall for the whole timeout at
            // game start, since the opening position is already actionable and
            // nothing new will arrive until we act on it.)
            let data = this.serverState;
            while (this.sharedMode && this.sessionId) {
                if (!data || data.error) {
                    // Lost our state — fetch the current one immediately.
                    try { data = await API.longPoll(this.sessionId, null); }
                    catch (e) { await sleep(800); data = null; continue; }
                    if (!this.sharedMode) return;
                    if (!data || data.error) { await sleep(1200); data = null; continue; }
                }
                if (data.version !== undefined) this.lastVersion = data.version;

                // Animate any turns played by the partner or the NN since we
                // last synced.
                await this.applySharedDiff(data);
                if (!this.sharedMode) return;

                this.maybeShowTakeover(data);
                this.updateDisconnectPrompt(data);

                if (this.serverState.game_over) {
                    this.renderAll();
                    this.showEndgame();
                    return;
                }

                if (this.serverState.waiting_for_human) {
                    if (this.serverState.current_player === this.mySeat) {
                        // My turn — hand off to the interactive flow, stop looping.
                        this.liveSpec  = null;
                        this.holdMask  = 0;
                        this.nnHeldMask = 0;
                        this._sharedLooping = false;
                        try { await this.beginHumanTurn(); }
                        catch (e) { this.handleFailure(e); }
                        return;
                    }
                    // The other human is taking their turn — show its current
                    // state, then block until their next action arrives.
                    await this.spectateLiveTurn(this.serverState);
                    if (!this.sharedMode) return;
                    try { data = await API.longPoll(this.sessionId, this.lastVersion); }
                    catch (e) { await sleep(800); data = null; }
                } else {
                    // Pending NN seats — advance them and process the returned
                    // state next iteration (no extra round-trip / no blocking).
                    try { data = await API.playAll(this.sessionId); }
                    catch (e) { await sleep(800); data = null; }
                }
            }
        } finally {
            this._sharedLooping = false;
        }
    },

    // ----- Server-Sent Events: push-driven "watching" mode -----------------
    // Used while it's NOT my turn. The server streams the full game state on
    // every change; we process each (serialized so animations don't overlap)
    // exactly like the poll loop body. Falls back to the long-poll sharedLoop
    // when EventSource is unavailable. During my own turn the stream is closed
    // (nothing to watch); humanPlace reopens it after I place.
    startSharedEvents() {
        this.stopSharedEvents();
        if (!this.sharedMode || !this.sessionId) return;
        if (typeof EventSource === 'undefined') { this.sharedLoop(); return; }

        const id = this.sessionId;
        const es = new EventSource(`/api/game/${id}/events`);
        this._sse = es;
        es.onmessage = (ev) => {
            if (this._sse !== es) return;  // stale
            let data; try { data = JSON.parse(ev.data); } catch (_) { return; }
            this.showReconnecting(false);
            this._sseQueue = data;          // coalesce to the latest snapshot
            this._drainSharedQueue();
        };
        es.onerror = () => {
            if (this._sse !== es) return;
            if (es.readyState === EventSource.CLOSED) {
                // Server rejected / session gone — fall back to recovery.
                this.stopSharedEvents();
                this.handleFailure(new Error('event stream closed'));
            } else {
                // Transient: the browser is auto-reconnecting.
                this.showReconnecting(true);
            }
        };
    },

    stopSharedEvents() {
        if (this._sse) { this._sse.close(); this._sse = null; }
        this._sseQueue = null;
    },

    async _drainSharedQueue() {
        if (this._sseDraining) return;
        this._sseDraining = true;
        try {
            while (this._sseQueue && this.sharedMode && this.sessionId) {
                const data = this._sseQueue;
                this._sseQueue = null;
                await this._handleSharedState(data);
            }
        } finally {
            this._sseDraining = false;
        }
    },

    // Process one pushed state snapshot (same decisions as the poll loop body).
    async _handleSharedState(data) {
        if (!this.sharedMode || !this.sessionId || !data || data.error) return;
        if (data.version !== undefined) this.lastVersion = data.version;

        await this.applySharedDiff(data);
        if (!this.sharedMode) return;
        this.maybeShowTakeover(data);
        this.updateDisconnectPrompt(data);

        if (this.serverState.game_over) {
            this.stopSharedEvents();
            this.renderAll();
            this.showEndgame();
            return;
        }

        if (this.serverState.waiting_for_human) {
            if (this.serverState.current_player === this.mySeat) {
                // My turn — stop watching and hand off to the interactive flow.
                this.liveSpec   = null;
                this.holdMask   = 0;
                this.nnHeldMask = 0;
                this.stopSharedEvents();
                try { await this.beginHumanTurn(); }
                catch (e) { this.handleFailure(e); }
                return;
            }
            // The other human is mid-turn — show its current state. The next
            // push (their next action) arrives on the stream.
            await this.spectateLiveTurn(this.serverState);
        } else {
            // Pending NN seats — advance them; the resulting change is pushed.
            try { await API.playAll(this.sessionId); } catch (_) { /* retry on next push */ }
        }
    },

    // Animate the turns in `data.history` that we haven't shown yet (the
    // partner's and the NN's moves). My own turns never appear here — they are
    // applied synchronously in humanPlace. Mirrors runNnTurn's reveal trick.
    async applySharedDiff(data) {
        // On first entry, replay from turn 0 so the opening NN move(s) animate
        // even if our baseline already includes them (the other player triggered
        // them before we connected).
        const oldLen   = this._animateFromStart
            ? 0
            : (this.serverState ? this.serverState.history.length : 0);
        this._animateFromStart = false;
        const newTurns = (data.history || []).slice(oldLen);

        if (newTurns.length === 0) {
            this.serverState   = data;
            this.displayBoards = cloneBoards(data.boards);
            return;
        }

        // Hide each newly-placed cell so the animation reveals them in order.
        const newDisplayBoards = cloneBoards(data.boards);
        for (const t of newTurns) {
            const cKey = COLUMN_KEYS[t.placement.column];
            const rKey = ROW_NAMES[t.placement.row];
            newDisplayBoards['player' + t.player][cKey][rKey] = -1;
        }
        this.displayBoards = newDisplayBoards;
        this.serverState   = data;

        this.animating = true;
        this.nnHeldMask = 0;  // drop any stale spectated-hold highlight
        for (const t of newTurns) {
            if (!this.sharedMode) { this.animating = false; return; }

            const isHumanSeat = data.player_types &&
                                data.player_types[t.player] === 'human';
            // A human teammate's turn was already shown live (its rolls/holds via
            // spectateLiveTurn), so just reveal the placement — instantly, the
            // way the active player sees it. NN moves keep their normal pacing.
            const placementOnly = isHumanSeat || !!(this.liveSpec &&
                this.liveSpec.player === t.player &&
                sameDice(this.liveSpec.initial, t.initial_dice));
            if (this.liveSpec && this.liveSpec.player === t.player) this.liveSpec = null;

            const tm = isHumanSeat ? T_FAST : T;
            this.animationPlayer = t.player;
            this.setTurnInfo(this.turnLabelFor(t.player), this.turnClassFor(t.player));
            this.renderAll();
            // No pre-pause before a teammate's placement reveal — show it at once.
            await sleep(isHumanSeat ? 0 : (placementOnly ? tm.nnBeforePlace : tm.preRollPause));
            await this.animateBotTurn(t.player, t,
                { placementOnly, instantReveal: isHumanSeat, timing: tm });
        }
        this.animationPlayer = null;
        this.nnHeldMask      = 0;
        this.nnRollingFlags  = null;
        this.animating       = false;
        this.renderAll();
    },

    // Spectate the other human's in-progress turn: animate the initial roll and
    // each committed reroll as they arrive, and reflect the tentative hold
    // preview. State is carried in this.liveSpec so we only animate new steps
    // each poll; applySharedDiff later reveals just the placement.
    async spectateLiveTurn(data) {
        const lt = data.live_turn;
        if (!lt) return;
        const seat = lt.player;

        // Start (or restart) watching when the player or their initial roll
        // changes — the latter guards against a seat replayed by the NN after a
        // takeover (different dice ⇒ treat as a brand-new turn).
        const fresh = !this.liveSpec || this.liveSpec.player !== seat ||
                      !sameDice(this.liveSpec.initial, lt.initial_dice);
        if (fresh) {
            this.liveSpec = { player: seat,
                              initial: (lt.initial_dice || []).slice(),
                              shownRerolls: 0, rolling: false };
            this.animationPlayer = seat;
            this.displayBoards = cloneBoards(data.boards);
            this.setCurrentPlayerBodyClass(seat);
            this.setTurnInfo(this.turnLabelFor(seat), this.turnClassFor(seat));
            this.animating = true;
            await this.animateRoll(lt.initial_dice, T_FAST);
            this.hasShownDice = true;
            this.animating = false;
        }

        // Reveal any rerolls committed since we last looked.
        const holds = lt.holds || [];
        if (holds.length > this.liveSpec.shownRerolls) {
            this.animating = true;
            for (let k = this.liveSpec.shownRerolls; k < holds.length; k++) {
                const hold = holds[k];
                if (this.liveSpec.rolling) {
                    // We were already spinning optimistically (the player's
                    // "rolling" signal arrived first) — snap straight to the
                    // result, no extra spin.
                    this.liveSpec.rolling = false;
                } else {
                    // Brief spin. The held dice were already shown live via the
                    // preview, so don't re-highlight them first (#2).
                    const mask = hold.mask | 0;
                    const spinning = [];
                    for (let i = 0; i < this.displayDice.length; i++) {
                        spinning.push(((mask >> i) & 1) === 0);
                    }
                    this.nnRollingFlags = spinning;
                    this.renderAll();
                    await sleep(T_FAST.rollSpin);
                }

                this.displayDice = (hold.dice_after || []).slice();
                let postMask = 0;
                const flags = hold.held_flags || [];
                for (let i = 0; i < flags.length; i++) if (flags[i]) postMask |= (1 << i);
                this.nnHeldMask = postMask;
                this.nnRollingFlags = null;
                this.renderAll();
                await sleep(T_FAST.nnAfterRoll);
            }
            this.liveSpec.shownRerolls = holds.length;
            this.animating = false;
        }

        // Optimistic spin: the player committed a reroll but the new dice aren't
        // here yet. Spin the un-held dice now and hold there until the result
        // push reveals them (above). Idempotent across repeated pushes.
        if (lt.rolling && !this.liveSpec.rolling) {
            this.liveSpec.rolling = true;
            const mask = lt.preview_mask | 0;
            const spinning = [];
            for (let i = 0; i < this.displayDice.length; i++) {
                spinning.push(((mask >> i) & 1) === 0);
            }
            this.nnHeldMask = mask;
            this.nnRollingFlags = spinning;
            this.animationPlayer = seat;
            this.renderAll();
            return;
        }

        if (!lt.rolling) {
            // Deliberating — reflect the current dice and tentative hold preview.
            this.displayDice = (data.dice || []).slice();
            this.nnHeldMask  = lt.preview_mask | 0;
            this.nnRollingFlags = null;
            this.animationPlayer = seat;
            this.renderAll();
        }
    },

    // Debounced push of my tentative hold selection (shared games only) so my
    // teammate can watch the hold pattern form.
    sendHoldPreview() {
        if (!this.sharedMode) return;
        if (this._holdPreviewTimer) clearTimeout(this._holdPreviewTimer);
        this._holdPreviewTimer = setTimeout(() => {
            this._holdPreviewTimer = null;
            if (this.sharedMode && this.sessionId && this.serverState &&
                this.serverState.current_player === this.mySeat) {
                API.holdPreview(this.sessionId, this.mySeat, this.holdMask);
            }
        }, 180);
    },

    // Show/hide the "teammate seems to have left — play their seat as the AI?"
    // prompt based on the server's disconnected_seats. The flip is the surviving
    // player's choice; if the partner resumes, the flag clears and this hides.
    updateDisconnectPrompt(data) {
        const el = document.getElementById('disconnect-prompt');
        if (!el) return;
        const disc  = data.disconnected_seats || [];
        const taken = data.nn_takeover_seats || [];
        const show = this.sharedMode &&
                     disc.indexOf(this.partnerSeat) >= 0 &&
                     taken.indexOf(this.partnerSeat) < 0 &&
                     !this._takeoverPending;
        el.hidden = !show;
    },

    // The surviving player opts to switch their disconnected teammate to the AI.
    async requestTakeover() {
        if (!this.sharedMode || !this.sessionId || this._takeoverPending) return;
        this._takeoverPending = true;
        const btn = document.getElementById('btn-takeover');
        if (btn) { btn.disabled = true; btn.textContent = 'Switching…'; }

        let data;
        try { data = await API.takeover(this.sessionId, this.partnerSeat); }
        catch (e) { data = { error: String(e) }; }

        this._takeoverPending = false;
        if (btn) { btn.disabled = false; btn.textContent = 'Play their seat as the AI'; }

        if (data && !data.error) {
            // Reflect it immediately (banner + AI label); the poll loop / place
            // chain animates the AI's turns from here.
            document.getElementById('disconnect-prompt').hidden = true;
            this.maybeShowTakeover(data);
        }
    },

    // Show a one-time banner when a human seat was taken over by the NN (a
    // teammate disconnected, or — rarely — this client itself was timed out).
    maybeShowTakeover(data) {
        const seats = data.nn_takeover_seats || [];
        this.takenOverSeats = seats.slice();  // so labels/animation treat them as AI
        // A taken-over seat's in-progress turn is discarded server-side, so stop
        // expecting it to land in history as the turn we were spectating.
        if (this.liveSpec && seats.indexOf(this.liveSpec.player) >= 0) this.liveSpec = null;
        if (seats.length === 0 || this.takeoverShown) return;
        this.takeoverShown = true;
        const el = document.getElementById('takeover-banner');
        if (!el) return;
        const mineTaken = seats.indexOf(this.mySeat) >= 0;
        const span = el.querySelector('span');
        if (span) {
            span.textContent = mineTaken
                ? 'You were disconnected — the AI is playing your seat.'
                : 'Your teammate left — the AI is now playing their seat.';
        }
        el.hidden = false;
        setTimeout(() => { el.hidden = true; }, 6000);
    },

    // =================================================================
    // Connection / fault tolerance
    //
    // Every network call can fail when a mobile tab is backgrounded (the OS
    // aborts in-flight fetches and throttles timers). The server keeps the
    // authoritative game state, so instead of letting the animation chain die
    // we latch a recovery, re-sync from GET /api/game/:id, and resume play.
    // =================================================================

    showReconnecting(on) {
        const el = document.getElementById('reconnect-banner');
        if (el) el.hidden = !on;
    },

    // An operation failed (network error, timeout, or server 5xx). Latch a
    // recovery and kick it off if we're visible. Never throws.
    handleFailure(e) {
        console.warn('[pro-yams] request failed; will recover', e);
        this.needsRecovery = true;
        this.showReconnecting(true);
        this.ensureRetryPump();
        // If backgrounded, don't start a doomed attempt now — the
        // visibilitychange on return will trigger a fresh one immediately.
        if (document.visibilityState === 'visible') this.onReconnectOpportunity();
    },

    // While a recovery is pending, a low-frequency timer keeps retrying even
    // without a visibility/online/focus event. (Throttled in the background,
    // which is fine — the user can't see the game then anyway.)
    ensureRetryPump() {
        if (this._retryTimer) return;
        this._retryTimer = setInterval(() => this.onReconnectOpportunity(), 2500);
    },

    clearRecovery() {
        this.needsRecovery = false;
        this.showReconnecting(false);
        if (this._retryTimer) { clearInterval(this._retryTimer); this._retryTimer = null; }
    },

    onReconnectOpportunity() {
        if (!this.needsRecovery || this.recovering) return;
        this.recover();
    },

    // Re-sync from the server's authoritative state and resume. Self-guarding:
    // at most one recovery runs at a time, and a failed attempt simply leaves
    // needsRecovery set for the next opportunity (timer / visibility / online).
    async recover() {
        if (this.recovering) return;
        if (!this.sessionId && !this._pendingStartMode) { this.clearRecovery(); return; }

        this.recovering = true;
        this.showReconnecting(true);
        try {
            if (!this.sessionId) {
                // The game never got created — retry creation. start() runs its
                // own guarded flow and clears recovery on success.
                await this.start(this._pendingStartMode);
                return;
            }

            const data = await API.getGame(this.sessionId);
            if (data && data.error) {
                // Session is gone (e.g. server restarted) — nothing to resume.
                this.clearRecovery();
                this.setTurnInfo('Game expired — tap the title to start a new one.', 'loss');
                return;
            }

            // Connection is back. Apply the authoritative state and drop the
            // banner before running any (possibly lengthy) catch-up animations.
            this.applyAuthoritativeState(data);
            this.clearRecovery();
            await this.resume();
        } catch (e) {
            // Still unreachable; keep the banner and let the next opportunity retry.
            console.warn('[pro-yams] recovery attempt failed; will retry', e);
            this.needsRecovery = true;
            this.showReconnecting(true);
            this.ensureRetryPump();
        } finally {
            this.recovering = false;
        }
    },

    // Replace all local/animation state with the server's authoritative state,
    // discarding any optimistic edits and stale animation flags.
    applyAuthoritativeState(data) {
        this.serverState   = data;
        if (data.version !== undefined) this.lastVersion = data.version;
        this.numPlayers    = data.num_players || this.numPlayers;
        this.variant       = variantFromState(data);
        document.body.dataset.variant = this.variant;
        this.displayBoards = cloneBoards(data.boards);
        this.displayDice   = (data.dice && data.dice.length ? data.dice
                                                            : [0, 0, 0, 0, 0]).slice();
        this.holdMask        = 0;
        this.nnHeldMask      = 0;
        this.nnRollingFlags  = null;
        this.justPlaced      = null;
        this.animationPlayer = null;
        this.options         = null;
        this.animating       = false;
        this.hasShownDice    = true;  // we have a real dealt state to show
        this.renderAll();
    },

    // Resume play from the (freshly applied) authoritative state.
    async resume() {
        const gs = this.serverState;
        if (!gs) return;
        if (gs.game_over) { this.renderAll(); this.showEndgame(); return; }
        // Shared games: if it's my turn, re-enter the interactive flow;
        // otherwise resume watching via the event stream.
        if (this.sharedMode) {
            if (gs.waiting_for_human && gs.current_player === this.mySeat) {
                this.stopSharedEvents();
                await this.enterHumanTurnFromState();
            } else {
                this.startSharedEvents();
            }
            return;
        }
        if (gs.waiting_for_human) await this.enterHumanTurnFromState();
        else await this.driveLoop();
    },

    // Land directly into a human turn from the current state, skipping the
    // dice-roll animation (the dice are already known from the server state).
    async enterHumanTurnFromState() {
        const gs = this.serverState;
        const seat = gs.current_player;
        this.animating = true;
        this.animationPlayer = null;
        this.setTurnInfo(this.turnLabelFor(seat), this.turnClassFor(seat));
        this.setCurrentPlayerBodyClass(seat);
        this.displayDice = (gs.dice || []).slice();
        this.nnRollingFlags = null;
        this.hasShownDice = true;
        this.renderAll();
        await this.refreshHumanOptions();
        this.animating = false;
        this.renderAll();
    },

    async driveLoop() {
        if (this.animating) return;
        while (this.serverState && !this.serverState.game_over &&
               !this.serverState.waiting_for_human) {
            await this.runNnTurn();
            if (!this.serverState || this.serverState.game_over) break;
        }
        if (this.serverState && this.serverState.game_over) {
            this.renderAll();
            this.showEndgame();
            return;
        }
        if (this.serverState && this.serverState.waiting_for_human) {
            await this.beginHumanTurn();
        }
    },

    async simulateStep() {
        if (!this.serverState || this.serverState.game_over) return;
        if (this.serverState.waiting_for_human) {
            this.animating = true;
            const playingSeat = this.serverState.current_player;
            this.animationPlayer = playingSeat;
            this.setTurnInfo(this.turnLabelFor(playingSeat), this.turnClassFor(playingSeat));
            this.renderAll();

            const data = await API.botStep(this.sessionId);
            if (data.error) {
                this.setTurnInfo('Error: ' + data.error);
                this.animating = false;
                return;
            }

            const oldLen = this.serverState.history.length;
            const newTurns = data.history.slice(oldLen);
            const botTurn = newTurns.find((t) => t.player === playingSeat);

            const newDisplayBoards = cloneBoards(data.boards);
            if (botTurn) {
                const colKey = COLUMN_KEYS[botTurn.placement.column];
                const rowKey = ROW_NAMES[botTurn.placement.row];
                newDisplayBoards['player' + playingSeat][colKey][rowKey] = -1;
            }
            this.displayBoards = newDisplayBoards;
            this.serverState = data;

            if (botTurn) {
                await this.animateBotTurn(playingSeat, botTurn);
            }

            this.nnHeldMask = 0;
            this.nnRollingFlags = null;
            this.animating = false;
            this.renderAll();
        } else {
            await this.runNnTurn();
        }
    },

    async simulate(turns = 100) {
        if (!this.sessionId || !this.serverState) {
            console.warn("No active game session to simulate. Start a game first.");
            return;
        }
        console.log(`Simulating ${turns} turns...`);
        const origT = Object.assign({}, T);
        for (const k of Object.keys(T)) {
            T[k] = 1;
        }
        try {
            let i = 0;
            while (this.serverState && !this.serverState.game_over && i < turns) {
                await this.simulateStep();
                i++;
                await sleep(5);
            }
            if (this.serverState && this.serverState.game_over) {
                this.renderAll();
                this.showEndgame();
            }
        } finally {
            Object.assign(T, origT);
            console.log("Simulation finished.");
        }
    },

    // -----------------------------------------------------------------
    // NN turn — runs one bot seat through /step. In 2v2 there can be up
    // to three consecutive bot turns before the human plays again.
    // -----------------------------------------------------------------
    async runNnTurn() {
        this.animating = true;
        const playingSeat = this.serverState.current_player;
        this.animationPlayer = playingSeat;
        this.setTurnInfo(this.turnLabelFor(playingSeat), this.turnClassFor(playingSeat));
        this.renderAll();

        if (this.hasShownDice) {
            await sleep(T.preRollPause);
        }

        const data = await API.step(this.sessionId);
        if (data.error) {
            this.setTurnInfo('Error: ' + data.error);
            this.animating = false;
            return;
        }

        const oldLen = this.serverState.history.length;
        const newTurns = data.history.slice(oldLen);
        const nnTurn = newTurns.find((t) => t.player === playingSeat);

        const newDisplayBoards = cloneBoards(data.boards);
        if (nnTurn) {
            const colKey = COLUMN_KEYS[nnTurn.placement.column];
            const rowKey = ROW_NAMES[nnTurn.placement.row];
            newDisplayBoards['player' + playingSeat][colKey][rowKey] = -1;
        }
        this.displayBoards = newDisplayBoards;
        this.serverState = data;

        if (nnTurn) {
            await this.animateBotTurn(playingSeat, nnTurn);
        }

        this.nnHeldMask = 0;
        this.nnRollingFlags = null;
        this.animating = false;
        this.renderAll();
    },

    // -----------------------------------------------------------------
    // Animate one bot's full turn (initial roll → holds → placement).
    // -----------------------------------------------------------------
    async animateBotTurn(seat, turnRec, opts = {}) {
        // opts.timing: animation profile (defaults to T; T_FAST for shared
        // catch-up). opts.placementOnly: the dice/rerolls were already shown
        // live (a spectated human turn), so jump straight to the placement.
        const tm = opts.timing || T;
        if (!opts.placementOnly) {
            await this.animateRoll(turnRec.initial_dice, tm);
            this.hasShownDice = true;

            for (const hold of (turnRec.holds || [])) {
                const mask = hold.mask | 0;
                this.nnHeldMask = mask;
                this.renderAll();
                await sleep(tm.nnHoldHighlight);

                const rolling = [];
                for (let i = 0; i < this.displayDice.length; i++) {
                    rolling.push(((mask >> i) & 1) === 0);
                }
                this.nnRollingFlags = rolling;
                this.renderAll();
                await sleep(tm.rollSpin);

                this.displayDice = (hold.dice_after || []).slice();
                let postMask = 0;
                const flags = hold.held_flags || [];
                for (let i = 0; i < flags.length; i++) {
                    if (flags[i]) postMask |= (1 << i);
                }
                this.nnHeldMask = postMask;
                this.nnRollingFlags = null;
                this.renderAll();
                await sleep(tm.nnAfterRoll);
            }

            this.nnHeldMask = 0;
            this.renderAll();
            await sleep(tm.nnBeforePlace);
        }

        const colKey = COLUMN_KEYS[turnRec.placement.column];
        const rowKey = ROW_NAMES[turnRec.placement.row];
        if (opts.instantReveal) {
            // Show the value at once with a simultaneous flash — matches the
            // active player's optimistic placement, so a spectating teammate
            // sees the placement land immediately rather than after a flash gap.
            this.displayBoards['player' + seat][colKey][rowKey] = turnRec.score;
            this.justPlaced = { player: seat, column: turnRec.placement.column,
                                 row: turnRec.placement.row };
            this.flashCell(seat, turnRec.placement.column, turnRec.placement.row, tm.placeFlash);
            this.renderAll();
            await sleep(tm.afterPlace);
            this.justPlaced = null;
        } else {
            this.flashCell(seat, turnRec.placement.column, turnRec.placement.row, tm.placeFlash);
            await sleep(tm.placeFlash);

            this.displayBoards['player' + seat][colKey][rowKey] = turnRec.score;
            this.justPlaced = { player: seat, column: turnRec.placement.column,
                                 row: turnRec.placement.row };
            this.renderAll();
            await sleep(tm.afterPlace);
            this.justPlaced = null;
        }
    },

    // -----------------------------------------------------------------
    // Human turn start
    // -----------------------------------------------------------------
    async beginHumanTurn() {
        this.animating = true;
        const seat = this.serverState.current_player;
        this.setTurnInfo(this.turnLabelFor(seat), this.turnClassFor(seat));
        this.setCurrentPlayerBodyClass(seat);
        this.renderAll();

        if (this.hasShownDice) {
            await sleep(T.preRollPause);
        }
        this.animationPlayer = null;
        await this.animateRoll(this.serverState.dice);
        this.hasShownDice = true;

        await this.refreshHumanOptions();
        this.animating = false;
        this.renderAll();
    },

    async refreshHumanOptions() {
        this.options = await API.getOptions(this.sessionId);
    },

    async animateRoll(targetDice, timing) {
        const tm = timing || T;
        this.nnRollingFlags = [true, true, true, true, true];
        this.renderAll();
        await sleep(tm.rollSpin);
        this.displayDice = (targetDice || []).slice();
        this.nnRollingFlags = null;
        this.renderAll();
    },

    // -----------------------------------------------------------------
    // Human reroll
    // -----------------------------------------------------------------
    async humanReroll() {
        if (this.animating || this.recovering ||
            !this.serverState || !this.serverState.waiting_for_human) return;
        if (this.sharedMode && this.serverState.current_player !== this.mySeat) return;
        if (this.serverState.rolls_left <= 0) return;

        try {
            this.animating = true;
            const mask = this.holdMask;
            // Tell a spectating teammate we're rolling now, so they start the
            // spin in step with us instead of after the result round-trips back.
            if (this.sharedMode) API.rolling(this.sessionId, this.mySeat, mask);
            const rolling = [];
            for (let i = 0; i < this.displayDice.length; i++) {
                rolling.push(((mask >> i) & 1) === 0);
            }
            this.nnRollingFlags = rolling;
            this.renderAll();
            await sleep(T.rollSpin);

            const data = await API.hold(this.sessionId, mask);
            this.nnRollingFlags = null;
            if (data.error) {
                this.setTurnInfo('Error: ' + data.error);
                this.animating = false;
                return;
            }

            const prevDice = this.displayDice.slice();
            this.serverState = data;
            this.displayBoards = cloneBoards(data.boards);
            this.displayDice = (data.dice || []).slice();
            this.holdMask = remapHoldMask(prevDice, mask, this.displayDice);
            this.animating = false;

            await this.refreshHumanOptions();
            this.renderAll();
            this.sendHoldPreview();
        } catch (e) {
            this.handleFailure(e);
        }
    },

    // -----------------------------------------------------------------
    // Human placement — server's /place chains all bot turns up to the
    // next human prompt (or game end). In 2v2 that's up to 3 bot turns.
    // -----------------------------------------------------------------
    async humanPlace(column, row) {
        if (this.animating || this.recovering ||
            !this.serverState || !this.serverState.waiting_for_human) return;
        if (this.sharedMode && this.serverState.current_player !== this.mySeat) return;
        const placements = (this.options && this.options.placements) || [];
        const opt = placements.find((p) => p.column === column && p.row === row);
        if (!opt) return;

        try {
            this.animating = true;

            // Optimistic local update for the just-placed human cell. In 2v2
            // the active human seat may be P0 or P2 — read it from the server
            // state rather than assuming a single human seat.
            const placingSeat = this.serverState.current_player;
            const colKey = COLUMN_KEYS[column];
            const rowKey = ROW_NAMES[row];
            this.displayBoards['player' + placingSeat][colKey][rowKey] = opt.score;
            this.justPlaced = { player: placingSeat, column, row };
            this.options = null;
            this.renderAll();
            await sleep(T.afterPlace);
            this.justPlaced = null;

            const data = await API.place(this.sessionId, column, row);
            if (data.error) {
                this.setTurnInfo('Error: ' + data.error);
                this.animating = false;
                return;
            }

            // The partner may have been handed to the NN while I was deciding;
            // pick that up now so the teammate's chained turn is labelled/treated
            // as AI (and the banner shows) before we animate it.
            if (this.sharedMode) { this.maybeShowTakeover(data); this.updateDisconnectPrompt(data); }

            const oldLen = this.serverState.history.length;
            const newTurns = data.history.slice(oldLen);
            // Animate every chained turn that isn't mine. In shared mode the
            // teammate seat is "human" in humanSeats but, once the NN has taken
            // it over, the server plays it inside my placement chain — so filter
            // by "not me" rather than "not human", otherwise the AI teammate's
            // turn would pop in with no roll animation.
            const botTurns = newTurns.filter((t) =>
                this.sharedMode ? (t.player !== this.mySeat) : !this.isHuman(t.player));

            // Hide every bot's just-placed cell so the animation reveals each one
            // at the right moment.
            const newDisplayBoards = cloneBoards(data.boards);
            for (const t of botTurns) {
                const cKey = COLUMN_KEYS[t.placement.column];
                const rKey = ROW_NAMES[t.placement.row];
                newDisplayBoards['player' + t.player][cKey][rKey] = -1;
            }
            this.displayBoards = newDisplayBoards;
            this.serverState = data;

            // After my placement the chain is all NN seats (the server stops at
            // the next human), so animate them at normal NN speed.
            for (const t of botTurns) {
                this.animationPlayer = t.player;
                this.setTurnInfo(this.turnLabelFor(t.player), this.turnClassFor(t.player));
                this.renderAll();
                await sleep(T.preRollPause);
                await this.animateBotTurn(t.player, t);
            }

            this.holdMask = 0;
            this.nnHeldMask = 0;
            this.nnRollingFlags = null;
            this.animating = false;

            if (this.serverState.game_over) {
                this.renderAll();
                this.showEndgame();
                return;
            }

            if (this.sharedMode) this.startSharedEvents();
            else await this.driveLoop();
        } catch (e) {
            this.handleFailure(e);
        }
    },

    // -----------------------------------------------------------------
    // Rendering
    // -----------------------------------------------------------------

    renderAll() {
        this.renderTurnInfo();
        this.renderSheet();
        this.renderDiceRow();
    },

    setTurnInfo(text, cls) {
        const el = document.getElementById('turn-info');
        el.textContent = text || '';
        el.className = 'turn-info' + (cls ? ` ${cls}` : '');
    },

    turnLabelFor(seat) {
        if (this.numPlayers === 4) {
            const team = teamOf(seat, 4);
            if (this.sharedMode) {
                if (seat === this.mySeat)      return `Your turn · P${seat}`;
                if (seat === this.partnerSeat) {
                    return this.takenOverSeats.indexOf(seat) >= 0
                        ? `NN teammate · P${seat}`   // partner left; AI plays it
                        : `Teammate · P${seat}`;
                }
                return `NN · P${seat} · Team ${team}`;
            }
            if (this.isHuman(seat)) return `Your turn · P${seat} · Team ${team}`;
            return `NN · P${seat} · Team ${team}`;
        }
        return this.isHuman(seat) ? 'Your turn' : 'NN turn';
    },
    turnClassFor(seat) {
        // In 2v2 use the per-player palette so the header color matches
        // the cell colors. In 1v1 keep the simple you/nn labels.
        if (this.numPlayers === 4) return `p${seat}`;
        return this.isHuman(seat) ? 'you' : 'nn';
    },

    setCurrentPlayerBodyClass(cur) {
        const body = document.body;
        body.classList.remove('cur-p0', 'cur-p1', 'cur-p2', 'cur-p3');
        if (cur !== null && cur !== undefined) {
            body.classList.add(`cur-p${cur}`);
        }
    },

    renderTurnInfo() {
        const gs = this.serverState;
        if (!gs) { this.setTurnInfo(''); return; }
        if (gs.game_over) return;  // showEndgame handles the final label
        const cur = (this.animationPlayer !== null && this.animationPlayer !== undefined)
                    ? this.animationPlayer : gs.current_player;
        this.setCurrentPlayerBodyClass(cur);
        // In a shared game only *my* seat is interactive; the partner's seat is
        // human in the session but a spectated turn from this browser.
        const isMyInteractive = this.sharedMode ? (cur === this.mySeat) : this.isHuman(cur);
        if (isMyInteractive) {
            // "Lucky Yams": a first-roll five-of-a-kind lets the player write
            // the max in any cell. Show a distinct banner so it's clear this is
            // not a normal Yams placement.
            if (gs.yams_bonus && gs.waiting_for_human && !this.animating) {
                this.setTurnInfo('Lucky YAMS! Place the max in any cell.', 'lucky-yams');
                return;
            }
            const rollNum = 3 - gs.rolls_left + 1;
            const base = this.numPlayers === 4
                ? `Your turn · P${cur} · Roll ${rollNum}/3`
                : `Your turn · Roll ${rollNum}/3`;
            this.setTurnInfo(base, this.turnClassFor(cur));
        } else {
            this.setTurnInfo(this.turnLabelFor(cur), this.turnClassFor(cur));
        }
    },

    renderDiceRow() {
        const container = document.getElementById('dice-row');
        container.innerHTML = '';
        if (!this.displayDice) return;

        const gs = this.serverState;
        // In shared mode only my own seat may act on the dice.
        const myTurn = !this.sharedMode || (gs && gs.current_player === this.mySeat);
        const isHumanTurn = gs && gs.waiting_for_human && !this.animating && myTurn;
        const canReroll   = isHumanTurn && gs.rolls_left > 0;

        for (let i = 0; i < this.displayDice.length; i++) {
            const die = document.createElement('div');
            die.className = 'die';
            const v = this.displayDice[i];
            if (!this.hasShownDice || v === 0) {
                die.classList.add('blank');
                die.textContent = '·';
            } else {
                die.textContent = v;
            }

            if (this.nnRollingFlags && this.nnRollingFlags[i]) {
                die.classList.add('rolling');
            }
            if (this.nnHeldMask && ((this.nnHeldMask >> i) & 1)) {
                die.classList.add('nn-held');
            }
            if (isHumanTurn && canReroll) {
                if ((this.holdMask >> i) & 1) die.classList.add('held');
                die.addEventListener('click', () => {
                    this.holdMask ^= (1 << i);
                    this.renderDiceRow();
                    this.sendHoldPreview();
                });
            } else {
                die.classList.add('inactive');
            }
            container.appendChild(die);
        }

        const rerollBtn = document.getElementById('btn-reroll');
        rerollBtn.disabled = !canReroll;
    },

    renderSheet() {
        if (this.variant === '2v2') return this.renderSheets2v2();
        return this.renderSheet1v1();
    },

    // ---------- 1v1: combined sheet (top corner = opp, main = current) ----
    renderSheet1v1() {
        const container = document.getElementById('sheet');
        container.innerHTML = '';
        container.className = 'sheet';
        if (!this.displayBoards || !this.serverState) return;

        const gs = this.serverState;
        const cur = (this.animationPlayer !== null && this.animationPlayer !== undefined)
                    ? this.animationPlayer : gs.current_player;
        const opp = 1 - cur;

        const isHumanTurn = gs.waiting_for_human && !this.animating;
        const legalPlacements = (isHumanTurn && this.options) ? this.options.placements : null;
        const { legalSet, legalScores } = this.buildLegalLookup(legalPlacements);

        const topBoards = [
            { board: this.displayBoards['player' + opp], cls: 'p-opp' },
        ];
        const grid = this.buildBoardGrid(
            this.displayBoards['player' + cur],
            topBoards,
            legalSet, legalScores,
            (c, r) => this.humanPlace(c, r));
        container.appendChild(grid);
    },

    // ---------- 2v2: same single-sheet trick, but the top splits into
    //            three sub-cells: left=prev, middle=teammate, right=next.
    //            Seat order is P0→P1→P2→P3, so for the current seat:
    //              prev = (cur-1) mod 4, team = (cur+2) mod 4, next = (cur+1) mod 4
    renderSheets2v2() {
        const container = document.getElementById('sheet');
        container.innerHTML = '';
        container.className = 'sheet sheets-2v2';
        if (!this.displayBoards || !this.serverState) return;

        const gs = this.serverState;
        const cur  = (this.animationPlayer !== null && this.animationPlayer !== undefined)
                    ? this.animationPlayer : gs.current_player;
        const prev = (cur + 3) % 4;
        const team = (cur + 2) % 4;
        const next = (cur + 1) % 4;

        const isHumanTurn = gs.waiting_for_human && !this.animating;
        const legalPlacements = (isHumanTurn && this.options) ? this.options.placements : null;
        const { legalSet, legalScores } = this.buildLegalLookup(legalPlacements);

        // Compact legend strip so the player→color mapping is explicit.
        container.appendChild(this.buildLegend2v2(cur, prev, team, next));

        const topBoards = [
            { board: this.displayBoards['player' + prev], cls: 'p-prev' },
            { board: this.displayBoards['player' + team], cls: 'p-team' },
            { board: this.displayBoards['player' + next], cls: 'p-next' },
        ];
        const grid = this.buildBoardGrid(
            this.displayBoards['player' + cur],
            topBoards,
            legalSet, legalScores,
            (c, r) => this.humanPlace(c, r));
        container.appendChild(grid);
    },

    buildLegend2v2(cur, prev, team, next) {
        const tag = (cls, seat, role) => {
            const span = document.createElement('span');
            span.className = `legend-item ${cls}`;
            const who = this.isHuman(seat) ? 'You' : 'NN';
            span.textContent = `P${seat} ${role} · ${who}`;
            return span;
        };
        const bar = document.createElement('div');
        bar.className = 'player-legend';
        bar.appendChild(tag('legend-prev', prev, '◀ prev'));
        bar.appendChild(tag('legend-team', team, '▲ team'));
        bar.appendChild(tag('legend-next', next, 'next ▶'));
        const cur1 = document.createElement('span');
        cur1.className = `legend-item legend-cur p${cur}`;
        cur1.textContent = `P${cur} ${this.isHuman(cur) ? 'YOU' : 'NN'}`;
        bar.appendChild(cur1);
        return bar;
    },

    buildLegalLookup(legalPlacements) {
        const legalSet = new Set();
        const legalScores = {};
        if (legalPlacements) {
            for (const p of legalPlacements) {
                legalSet.add(`${p.column},${p.row}`);
                legalScores[`${p.column},${p.row}`] = p.score;
            }
        }
        return { legalSet, legalScores };
    },

    // ---------- Board grid construction (shared 1v1/2v2) -----------------
    // topBoards: array of { board, cls } describing each top sub-cell.
    //   1v1: 1 entry (opp).
    //   2v2: 3 entries (prev, teammate, next), rendered as a 3-way split.
    buildBoardGrid(mainBoard, topBoards, legalSet, legalScores, onPlace) {
        const grid = document.createElement('div');
        grid.className = 'board-grid';
        if (topBoards && topBoards.length > 1) grid.classList.add('multi-top');

        const corner = document.createElement('div');
        corner.className = 'row-label';
        grid.appendChild(corner);

        const coefficients = this.serverState.coefficients || [];
        for (let c = 0; c < 6; c++) {
            const hdr = document.createElement('div');
            hdr.className = 'col-header';
            const coeff = coefficients[c];
            hdr.innerHTML = `${COLUMN_NAMES[c]}` +
                (coeff !== undefined ? `<span class="col-coeff">×${coeff}</span>` : '');
            grid.appendChild(hdr);
        }

        for (let r = 0; r < 13; r++) {
            const label = document.createElement('div');
            label.className = 'row-label';
            label.textContent = ROW_NAMES[r];
            grid.appendChild(label);

            for (let c = 0; c < 6; c++) {
                const cell = document.createElement('div');
                cell.className = 'board-cell';
                cell.dataset.cell = `c${c}-r${r}`;

                const colKey = COLUMN_KEYS[c];
                const rowKey = ROW_NAMES[r];
                const mainVal = mainBoard?.[colKey]?.[rowKey];

                cell.appendChild(this.buildCellTop(topBoards, colKey, rowKey));

                const mainEl = document.createElement('div');
                mainEl.className = 'cell-main';
                const legalKey = `${c},${r}`;
                const isLegal = legalSet && legalSet.has(legalKey);
                if (isLegal) {
                    cell.classList.add('legal');
                    const sc = legalScores[legalKey];
                    mainEl.textContent = (sc !== undefined ? sc : '');
                    if (onPlace) cell.addEventListener('click', () => onPlace(c, r));
                } else if (mainVal === undefined || mainVal === -1) {
                    mainEl.textContent = '';
                } else if (mainVal === 0) {
                    mainEl.textContent = '0';
                    mainEl.classList.add('scratched');
                } else {
                    mainEl.textContent = mainVal;
                }
                cell.appendChild(mainEl);

                if (this.justPlaced && this.justPlaced.column === c &&
                    this.justPlaced.row === r) {
                    cell.classList.add('just-placed');
                }
                grid.appendChild(cell);
            }

            if (r === 5) {
                const bLabel = document.createElement('div');
                bLabel.className = 'row-label';
                bLabel.textContent = 'Bonus';
                bLabel.style.color = 'var(--accent-2)';
                bLabel.style.fontSize = '0.6rem';
                grid.appendChild(bLabel);
                for (let c = 0; c < 6; c++) {
                    const bonusCell = document.createElement('div');
                    bonusCell.className = 'bonus-cell';
                    const colKey = COLUMN_KEYS[c];

                    bonusCell.appendChild(this.buildBonusTop(topBoards, colKey));

                    const { sum: mSum, allFilled: mFilled } = upperColumnSum(mainBoard, colKey);
                    const mainEl = document.createElement('div');
                    mainEl.className = 'cell-main';
                    mainEl.textContent = mFilled ? `${mSum}/${upperBonus(mSum)}`
                                                : (mSum > 0 ? String(mSum) : '');
                    bonusCell.appendChild(mainEl);

                    grid.appendChild(bonusCell);
                }
            }
        }
        return grid;
    },

    // Build the top sub-region of a regular score cell — either a single
    // `.cell-top` (1v1) or a `.cell-top-row` of three `.top-sub` siblings (2v2).
    buildCellTop(topBoards, colKey, rowKey) {
        const fillValue = (el, v) => {
            if (v === undefined || v === -1) {
                el.textContent = '';
            } else if (v === 0) {
                el.textContent = '0';
                el.classList.add('scratched');
            } else {
                el.textContent = v;
            }
        };
        if (!topBoards || topBoards.length === 0) {
            const empty = document.createElement('div');
            empty.className = 'cell-top';
            return empty;
        }
        if (topBoards.length === 1) {
            const topEl = document.createElement('div');
            topEl.className = `cell-top ${topBoards[0].cls || ''}`;
            fillValue(topEl, topBoards[0].board?.[colKey]?.[rowKey]);
            return topEl;
        }
        const row = document.createElement('div');
        row.className = 'cell-top-row';
        for (const tb of topBoards) {
            const sub = document.createElement('div');
            sub.className = `top-sub ${tb.cls || ''}`;
            fillValue(sub, tb.board?.[colKey]?.[rowKey]);
            row.appendChild(sub);
        }
        return row;
    },

    buildBonusTop(topBoards, colKey) {
        const fillBonus = (el, board) => {
            const { sum, allFilled } = upperColumnSum(board, colKey);
            el.textContent = allFilled ? `${sum}/${upperBonus(sum)}`
                                       : (sum > 0 ? String(sum) : '');
        };
        if (!topBoards || topBoards.length === 0) {
            const empty = document.createElement('div');
            empty.className = 'cell-top';
            return empty;
        }
        if (topBoards.length === 1) {
            const topEl = document.createElement('div');
            topEl.className = `cell-top ${topBoards[0].cls || ''}`;
            fillBonus(topEl, topBoards[0].board);
            return topEl;
        }
        const row = document.createElement('div');
        row.className = 'cell-top-row';
        for (const tb of topBoards) {
            const sub = document.createElement('div');
            sub.className = `top-sub ${tb.cls || ''}`;
            fillBonus(sub, tb.board);
            row.appendChild(sub);
        }
        return row;
    },

    flashCell(player, column, row, duration) {
        // Both variants use a single combined sheet; the cell flashes
        // regardless of whose value is going in (the current player's
        // board is in main, so the just-targeted cell is always main).
        const target = document.querySelector(`[data-cell="c${column}-r${row}"]`);
        if (!target) return;
        target.classList.add('target-flash');
        setTimeout(() => target.classList.remove('target-flash'), duration || T.placeFlash);
    },

    showEndgame() {
        const gs = this.serverState;
        if (!gs || !gs.game_over) return;
        const dp = document.getElementById('disconnect-prompt');
        if (dp) dp.hidden = true;
        const r = gs.result;
        const humanTeam = this.humanTeam;
        let cls, text;
        if (r === 0.5) { cls = 'draw'; text = 'Draw!'; }
        else {
            // Result is from team 0's perspective: 1.0 means Team 0 wins.
            const team0Won = (r === 1.0);
            const humanWon = (team0Won && humanTeam === 0) ||
                             (!team0Won && humanTeam === 1);
            cls  = humanWon ? 'win'  : 'loss';
            text = humanWon
                ? (this.numPlayers === 4 ? 'Team win! 🎉' : 'You win! 🎉')
                : (this.numPlayers === 4 ? 'Team loss.'  : 'NN wins.');
        }
        this.setTurnInfo(text, cls);

        const duel = computeDuel(gs);  // duel.totalDuel = Team 0 − Team 1
        const humanTotal = humanTeam === 0 ? duel.totalDuel : -duel.totalDuel;

        const txtEl = document.getElementById('endgame-text');
        txtEl.textContent = text;
        txtEl.className = 'endgame-text ' + cls;

        const marginEl = document.getElementById('endgame-margin');
        const marginCls = humanTotal > 0 ? 'pts-pos'
                        : humanTotal < 0 ? 'pts-neg' : 'pts-zero';
        marginEl.innerHTML = `Margin: <span class="${marginCls}">${fmtSigned(humanTotal)}</span> pts`;

        const tbl = document.getElementById('endgame-breakdown');
        tbl.innerHTML = renderBreakdownTable(duel.colData, gs, humanTeam, humanTotal);

        this.showLuck();

        const flagBtn = document.getElementById('btn-flag');
        if (flagBtn) {
            flagBtn.disabled = false;
            flagBtn.classList.remove('flagged');
            flagBtn.textContent = '⚠ AI played weird moves this game';
        }

        document.getElementById('sheet').hidden = true;
        document.getElementById('footer').hidden = true;
        document.getElementById('endgame').hidden = false;
    },

    // Fetch and display per-seat luck for the finished game (server replays the
    // move history through the NN). Async + best-effort: a failure just hides the
    // line rather than disrupting the endgame screen. Luck is shown as 0..100%
    // where 50% = rolled as expected, higher = luckier than expected.
    async showLuck() {
        const el = document.getElementById('endgame-luck');
        if (!el || !this.sessionId) return;
        const sid = this.sessionId;
        el.textContent = 'Computing luck…';
        let resp;
        try { resp = await API.getLuck(sid); }
        catch (e) { el.textContent = ''; return; }
        // Bail if the user moved on (new game / closed) while awaiting.
        if (this.sessionId !== sid) return;
        if (!resp || resp.error || !Array.isArray(resp.seat_luck)) { el.textContent = ''; return; }

        const toPct = (x) => (x + 100) / 2;
        const mean  = (a) => a.length ? a.reduce((s, x) => s + x, 0) / a.length : null;
        const humanVals = [], compVals = [];
        resp.seat_luck.forEach((v, p) => {
            if (v == null) return;
            (this.isHuman(p) ? humanVals : compVals).push(toPct(v));
        });
        const h = mean(humanVals), c = mean(compVals);
        const fmt = (x) => x == null ? '—' : x.toFixed(1) + '%';
        const youLabel = this.numPlayers === 4 ? 'Your team' : 'You';
        const oppLabel = this.numPlayers === 4 ? 'Their team' : 'NN';
        el.innerHTML =
            `Luck — <span class="luck-label">${youLabel}</span> ` +
            `<span class="luck-val">${fmt(h)}</span> · ` +
            `<span class="luck-label">${oppLabel}</span> ` +
            `<span class="luck-val">${fmt(c)}</span> ` +
            `<span class="luck-note">(50% = average)</span>`;
    },

    // Flag the finished game for review. One-shot: once flagged, the button
    // shows a thank-you and is disabled for the rest of this endgame screen.
    async flagGame() {
        const btn = document.getElementById('btn-flag');
        if (!this.sessionId || (btn && btn.disabled)) return;
        if (btn) btn.disabled = true;
        const res = await API.flagGame(this.sessionId);
        if (res && res.error) {
            if (btn) {
                btn.disabled = false;
                btn.textContent = '⚠ Could not flag — tap to retry';
            }
            return;
        }
        if (btn) {
            btn.classList.add('flagged');
            btn.textContent = '✓ Flagged for review — thanks!';
        }
    },
};

// ---------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------

function sameDice(a, b) {
    if (!a || !b || a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
    return true;
}

function cloneBoards(boards) {
    if (!boards) return null;
    const out = {};
    for (const pkey of Object.keys(boards)) {
        out[pkey] = {};
        for (const ckey of Object.keys(boards[pkey])) {
            out[pkey][ckey] = Object.assign({}, boards[pkey][ckey]);
        }
    }
    return out;
}

function upperBonus(sum) {
    if (sum >= 100) return 500;
    if (sum >=  90) return 200;
    if (sum >=  80) return 100;
    if (sum >=  70) return  50;
    if (sum >=  60) return  30;
    return 0;
}

function upperColumnSum(board, colKey) {
    let sum = 0, allFilled = true;
    for (let r = 0; r < 6; r++) {
        const v = board?.[colKey]?.[ROW_NAMES[r]];
        if (v === undefined || v === -1) allFilled = false;
        else if (v > 0) sum += v;
    }
    return { sum, allFilled };
}

function fmtSigned(n) {
    if (n > 0) return `+${n}`;
    return String(n);
}

const crushMult = (my, opp) => {
    if (opp === 0 && my > 0)            return 5;
    if (opp > 0 && my >= 5 * opp)       return 5;
    if (opp > 0 && my >= 4 * opp)       return 4;
    if (opp > 0 && my >= 3 * opp)       return 3;
    if (opp > 0 && my >= 2 * opp)       return 2;
    return 1;
};

// Mirror of compute_duel<Traits>() in src/engine/duel.cc. Iterates every
// cross-team pairing (1 in 1v1, 4 in 2v2). Returns:
//   { totalDuel: Team-0 minus Team-1 margin (signed),
//     colData:   per-column rendering data including per-player adj scores,
//                pairings array, and aggregated column-duel points } .
function computeDuel(gs) {
    const numPlayers = gs.num_players || 2;
    const team0 = numPlayers === 4 ? [0, 2] : [0];
    const team1 = numPlayers === 4 ? [1, 3] : [1];
    const UPPER = ROW_NAMES.slice(0, 6);
    const LOWER = ROW_NAMES.slice(6);
    const coeffs = gs.coefficients || [];

    let totalDuel = 0;
    const colData = [];

    for (let c = 0; c < 6; c++) {
        const colKey = COLUMN_KEYS[c];
        const coeff  = coeffs[c];

        // Per-player raw + clean eligibility.
        const players = [];
        for (let p = 0; p < numPlayers; p++) {
            const b = gs.boards?.[`player${p}`];
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
                // adj is set per-pairing below; for the per-player breakdown
                // we'll fill in the max bonus seen across this player's pairings.
                cBonus: 0,
                adjusted: cellsSum + uBonus,
            });
        }

        // Cross-team pairings.
        const pairings = [];
        let colDuel = 0;
        for (const t0p of team0) {
            for (const t1p of team1) {
                const a = players[t0p], b = players[t1p];
                const crush0 = crushMult(a.raw, b.raw);
                const crush1 = crushMult(b.raw, a.raw);
                const activeCrush = Math.max(crush0, crush1);
                const cleanBonus = activeCrush > 1 ? 100 : 200;
                const adjA = a.raw + (a.isClean ? cleanBonus : 0);
                const adjB = b.raw + (b.isClean ? cleanBonus : 0);
                const duelPts = (adjA - adjB) * activeCrush * coeff;
                colDuel += duelPts;
                pairings.push({ t0p, t1p, activeCrush, cleanBonus,
                                adjA, adjB, duelPts });
                // For the per-player breakdown we display the max clean bonus
                // this player received across their pairings (informational).
                if (a.isClean && cleanBonus > a.cBonus) {
                    a.cBonus = cleanBonus; a.adjusted = a.raw + cleanBonus;
                }
                if (b.isClean && cleanBonus > b.cBonus) {
                    b.cBonus = cleanBonus; b.adjusted = b.raw + cleanBonus;
                }
            }
        }

        totalDuel += colDuel;
        colData.push({ c, colKey, coeff, players, pairings, duelPts: colDuel });
    }

    return { totalDuel, colData };
}

// Render the per-column breakdown. In 1v1 we keep the original "You vs NN"
// two-column table. In 2v2 we show per-team totals plus a delta column.
function renderBreakdownTable(colData, gs, humanTeam, humanTotal) {
    const numPlayers = gs.num_players || 2;
    if (numPlayers !== 4) return renderBreakdownTable1v1(colData, humanTeam, humanTotal);
    return renderBreakdownTable2v2(colData, humanTeam, humanTotal);
}

function renderBreakdownTable1v1(colData, humanTeam, humanTotal) {
    const humanIsP0 = humanTeam === 0;
    let html = `<table class="score-table">
        <thead>
          <tr>
            <th>Column</th>
            <th class="you-col">You</th>
            <th class="nn-col">NN</th>
            <th>×</th>
            <th>Δ pts</th>
          </tr>
        </thead>
        <tbody>`;
    for (const col of colData) {
        const youAdj = humanIsP0 ? col.players[0].adjusted : col.players[1].adjusted;
        const nnAdj  = humanIsP0 ? col.players[1].adjusted : col.players[0].adjusted;
        const humanPts = humanIsP0 ? col.duelPts : -col.duelPts;
        const cls = humanPts > 0 ? 'pts-pos' : humanPts < 0 ? 'pts-neg' : 'pts-zero';
        const crushVal = col.pairings[0]?.activeCrush ?? 1;
        html += `<tr>
            <td>${COLUMN_NAMES[col.c]}<span class="col-coeff">×${col.coeff}</span></td>
            <td>${youAdj}</td>
            <td>${nnAdj}</td>
            <td>×${crushVal}</td>
            <td class="${cls}">${humanPts === 0 ? '0' : fmtSigned(humanPts)}</td>
          </tr>`;
    }
    const totCls = humanTotal > 0 ? 'pts-pos' : humanTotal < 0 ? 'pts-neg' : 'pts-zero';
    html += `</tbody>
        <tfoot>
          <tr>
            <td colspan="4" style="text-align:right; color: var(--muted);">Total</td>
            <td class="${totCls}">${fmtSigned(humanTotal)}</td>
          </tr>
        </tfoot>
      </table>`;
    return html;
}

function renderBreakdownTable2v2(colData, humanTeam, humanTotal) {
    const youLabel = humanTeam === 0 ? 'Team 0' : 'Team 1';
    const nnLabel  = humanTeam === 0 ? 'Team 1' : 'Team 0';
    // Per-column display: aggregate team adjusted-score sums for context.
    let html = `<table class="score-table">
        <thead>
          <tr>
            <th>Column</th>
            <th class="you-col">${youLabel}</th>
            <th class="nn-col">${nnLabel}</th>
            <th>Δ pts</th>
          </tr>
        </thead>
        <tbody>`;
    for (const col of colData) {
        const p0 = col.players[0].adjusted;
        const p2 = col.players[2].adjusted;
        const p1 = col.players[1].adjusted;
        const p3 = col.players[3].adjusted;

        const t0Adj = p0 + p2;
        const t1Adj = p1 + p3;

        const youAdj = humanTeam === 0 ? t0Adj : t1Adj;
        const nnAdj  = humanTeam === 0 ? t1Adj : t0Adj;

        const youSub = humanTeam === 0
            ? `<div style="font-size:0.65rem; color:var(--muted); margin-top:2px;"><span style="color:var(--p0)">P0</span>:${p0} <span style="color:var(--p2)">P2</span>:${p2}</div>`
            : `<div style="font-size:0.65rem; color:var(--muted); margin-top:2px;"><span style="color:var(--p1)">P1</span>:${p1} <span style="color:var(--p3)">P3</span>:${p3}</div>`;
        const nnSub = humanTeam === 0
            ? `<div style="font-size:0.65rem; color:var(--muted); margin-top:2px;"><span style="color:var(--p1)">P1</span>:${p1} <span style="color:var(--p3)">P3</span>:${p3}</div>`
            : `<div style="font-size:0.65rem; color:var(--muted); margin-top:2px;"><span style="color:var(--p0)">P0</span>:${p0} <span style="color:var(--p2)">P2</span>:${p2}</div>`;

        const humanPts = humanTeam === 0 ? col.duelPts : -col.duelPts;
        const cls = humanPts > 0 ? 'pts-pos' : humanPts < 0 ? 'pts-neg' : 'pts-zero';
        html += `<tr>
            <td>${COLUMN_NAMES[col.c]}<span class="col-coeff">×${col.coeff}</span></td>
            <td><strong>${youAdj}</strong>${youSub}</td>
            <td><strong>${nnAdj}</strong>${nnSub}</td>
            <td class="${cls}">${humanPts === 0 ? '0' : fmtSigned(humanPts)}</td>
          </tr>`;
    }
    // Calculate player total points
    let pPts = [0, 0, 0, 0];
    for (const col of colData) {
        for (const pairing of col.pairings) {
            pPts[pairing.t0p] += pairing.duelPts;
            pPts[pairing.t1p] += -pairing.duelPts;
        }
    }

    const p0Pts = pPts[0];
    const p2Pts = pPts[2];
    const p1Pts = pPts[1];
    const p3Pts = pPts[3];

    const p0Cls = p0Pts > 0 ? 'pts-pos' : p0Pts < 0 ? 'pts-neg' : 'pts-zero';
    const p2Cls = p2Pts > 0 ? 'pts-pos' : p2Pts < 0 ? 'pts-neg' : 'pts-zero';
    const p1Cls = p1Pts > 0 ? 'pts-pos' : p1Pts < 0 ? 'pts-neg' : 'pts-zero';
    const p3Cls = p3Pts > 0 ? 'pts-pos' : p3Pts < 0 ? 'pts-neg' : 'pts-zero';

    const isT0 = (humanTeam === 0);
    const youCell = isT0
        ? `<div style="font-size:0.75rem; margin-bottom:2px;"><span style="color:var(--p0); font-weight:600;">P0</span>: <strong class="${p0Cls}">${fmtSigned(p0Pts)}</strong></div>
           <div style="font-size:0.75rem;"><span style="color:var(--p2); font-weight:600;">P2</span>: <strong class="${p2Cls}">${fmtSigned(p2Pts)}</strong></div>`
        : `<div style="font-size:0.75rem; margin-bottom:2px;"><span style="color:var(--p1); font-weight:600;">P1</span>: <strong class="${p1Cls}">${fmtSigned(p1Pts)}</strong></div>
           <div style="font-size:0.75rem;"><span style="color:var(--p3); font-weight:600;">P3</span>: <strong class="${p3Cls}">${fmtSigned(p3Pts)}</strong></div>`;

    const nnCell = isT0
        ? `<div style="font-size:0.75rem; margin-bottom:2px;"><span style="color:var(--p1); font-weight:600;">P1</span>: <strong class="${p1Cls}">${fmtSigned(p1Pts)}</strong></div>
           <div style="font-size:0.75rem;"><span style="color:var(--p3); font-weight:600;">P3</span>: <strong class="${p3Cls}">${fmtSigned(p3Pts)}</strong></div>`
        : `<div style="font-size:0.75rem; margin-bottom:2px;"><span style="color:var(--p0); font-weight:600;">P0</span>: <strong class="${p0Cls}">${fmtSigned(p0Pts)}</strong></div>
           <div style="font-size:0.75rem;"><span style="color:var(--p2); font-weight:600;">P2</span>: <strong class="${p2Cls}">${fmtSigned(p2Pts)}</strong></div>`;

    const totCls = humanTotal > 0 ? 'pts-pos' : humanTotal < 0 ? 'pts-neg' : 'pts-zero';
    html += `</tbody>
        <tfoot>
          <tr>
            <td style="text-align:right; color: var(--muted); vertical-align:middle;">Player points</td>
            <td style="text-align:right; vertical-align:middle;">${youCell}</td>
            <td style="text-align:right; vertical-align:middle;">${nnCell}</td>
            <td></td>
          </tr>
          <tr>
            <td colspan="3" style="text-align:right; color: var(--muted);">Team total</td>
            <td class="${totCls}">${fmtSigned(humanTotal)}</td>
          </tr>
        </tfoot>
      </table>`;

    // Append the detailed duel breakdown table below the main one
    let detailHtml = `
    <h3 style="margin: 25px 0 10px; text-align: center; color: var(--text); font-size: 0.9rem; font-weight: 600; letter-spacing: 0.5px;">Detailed Duel Breakdown</h3>
    <table class="score-table" style="margin-bottom: 20px;">
        <thead>
          <tr>
            <th style="text-align:left;">Column</th>
            <th style="text-align:center;">Duel</th>
            <th style="text-align:center;">Scores</th>
            <th style="text-align:center;">Crush</th>
            <th style="text-align:right;">Points</th>
          </tr>
        </thead>
        <tbody>`;

    for (const col of colData) {
        for (let idx = 0; idx < col.pairings.length; idx++) {
            const pairing = col.pairings[idx];
            const pA = pairing.t0p;
            const pB = pairing.t1p;
            const isT0 = (humanTeam === 0);

            const diff = isT0 ? (pairing.adjA - pairing.adjB) : (pairing.adjB - pairing.adjA);
            const pts = isT0 ? pairing.duelPts : -pairing.duelPts;

            const ptsCls = pts > 0 ? 'pts-pos' : pts < 0 ? 'pts-neg' : 'pts-zero';
            const crushCls = pairing.activeCrush > 1 ? 'pts-pos' : '';

            const isLast = (idx === col.pairings.length - 1);
            const borderBottom = isLast ? 'border-bottom: 2px solid var(--panel-2);' : '';
            const cellStyleC = `style="${borderBottom} font-size:0.75rem; text-align:center;"`;
            const cellStylePoints = `style="${borderBottom} font-size:0.75rem; text-align:right; font-weight:700;"`;
            const crushWeight = pairing.activeCrush > 1 ? 'font-weight:700;' : '';
            const cellStyleCrush = `style="${borderBottom} font-size:0.75rem; text-align:center; ${crushWeight}"`;

            const rowSpanCell = idx === 0
                ? `<td rowspan="4" style="vertical-align: middle; border-bottom: 2px solid var(--panel-2); font-weight: 600; font-size:0.75rem; text-align:left;">${COLUMN_NAMES[col.c]}<span class="col-coeff">×${col.coeff}</span></td>`
                : '';

            detailHtml += `<tr>
                ${rowSpanCell}
                <td ${cellStyleC}><span style="color:var(--p${pA}); font-weight:600;">P${pA}</span> vs <span style="color:var(--p${pB}); font-weight:600;">P${pB}</span></td>
                <td ${cellStyleC}><strong>${pairing.adjA}</strong> vs <strong>${pairing.adjB}</strong> <span style="font-size:0.7rem; color:var(--muted);">(${fmtSigned(diff)})</span></td>
                <td ${cellStyleCrush} class="${crushCls}">×${pairing.activeCrush}</td>
                <td ${cellStylePoints} class="${ptsCls}">${pts === 0 ? '0' : fmtSigned(pts)}</td>
              </tr>`;
        }
    }
    detailHtml += `</tbody></table>`;
    html += detailHtml;

    return html;
}

// After a reroll the dice are re-sorted, so a hold mask over previous
// positions must be remapped to new positions.
function remapHoldMask(prevDice, prevMask, newDice) {
    const heldValues = prevDice.filter((_, i) => (prevMask >> i) & 1).sort((a, b) => a - b);
    if (heldValues.length === 0) return 0;
    let hi = 0, newMask = 0;
    for (let i = 0; i < newDice.length && hi < heldValues.length; i++) {
        if (newDice[i] === heldValues[hi]) {
            newMask |= (1 << i);
            hi++;
        }
    }
    return newMask;
}

window.simulate = (turns) => Play.simulate(turns);
document.addEventListener('DOMContentLoaded', () => Play.init());
