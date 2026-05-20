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
    options: null,
    animating: false,
    justPlaced: null,
    hasShownDice: false,
    animationPlayer: null,

    isHuman(p) { return this.humanSeats.indexOf(p) >= 0; },
    isHumanTeam(p) { return teamOf(p, this.numPlayers) === this.humanTeam; },

    init() {
        document.getElementById('btn-play')
            .addEventListener('click', () => Play.start('1v1'));
        document.getElementById('btn-play-team')
            .addEventListener('click', () => Play.start('2v2-team'));
        document.getElementById('btn-play-solo')
            .addEventListener('click', () => Play.start('2v2-solo'));
        document.getElementById('btn-again')
            .addEventListener('click', () => Play.start(Play.lastMode || '1v1'));
        document.getElementById('btn-reroll').addEventListener('click', () => Play.humanReroll());
        document.getElementById('header').addEventListener('click', () => Play.toSplash());
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
        document.getElementById('btn-play').hidden       = is2v2;
        document.getElementById('btn-play-team').hidden  = !is2v2;
        document.getElementById('btn-play-solo').hidden  = !is2v2;
    },

    toSplash() {
        if (this.sessionId) { API.deleteGame(this.sessionId); this.sessionId = null; }
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
        if (this.sessionId) { API.deleteGame(this.sessionId); this.sessionId = null; }
        this.holdMask = 0;
        this.nnHeldMask = 0;
        this.nnRollingFlags = null;
        this.animating = false;
        this.justPlaced = null;
        this.options = null;
        this.hasShownDice = false;
        this.displayDice = [0, 0, 0, 0, 0];

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

        const data = await API.newGame(playerTypes);
        this.sessionId    = data.session_id;
        this.serverState  = data.game_state;
        // Reconcile with server-reported variant in case it disagrees.
        this.numPlayers = this.serverState.num_players || this.numPlayers;
        this.variant    = variantFromState(this.serverState);
        document.body.dataset.variant = this.variant;
        this.displayBoards = cloneBoards(this.serverState.boards);

        this.renderAll();
        await this.driveLoop();
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
    async animateBotTurn(seat, turnRec) {
        await this.animateRoll(turnRec.initial_dice);
        this.hasShownDice = true;

        for (const hold of (turnRec.holds || [])) {
            const mask = hold.mask | 0;
            this.nnHeldMask = mask;
            this.renderAll();
            await sleep(T.nnHoldHighlight);

            const rolling = [];
            for (let i = 0; i < this.displayDice.length; i++) {
                rolling.push(((mask >> i) & 1) === 0);
            }
            this.nnRollingFlags = rolling;
            this.renderAll();
            await sleep(T.rollSpin);

            this.displayDice = (hold.dice_after || []).slice();
            let postMask = 0;
            const flags = hold.held_flags || [];
            for (let i = 0; i < flags.length; i++) {
                if (flags[i]) postMask |= (1 << i);
            }
            this.nnHeldMask = postMask;
            this.nnRollingFlags = null;
            this.renderAll();
            await sleep(T.nnAfterRoll);
        }

        this.nnHeldMask = 0;
        this.renderAll();
        await sleep(T.nnBeforePlace);

        const colKey = COLUMN_KEYS[turnRec.placement.column];
        const rowKey = ROW_NAMES[turnRec.placement.row];
        this.flashCell(seat, turnRec.placement.column, turnRec.placement.row);
        await sleep(T.placeFlash);

        this.displayBoards['player' + seat][colKey][rowKey] = turnRec.score;
        this.justPlaced = { player: seat, column: turnRec.placement.column,
                             row: turnRec.placement.row };
        this.renderAll();
        await sleep(T.afterPlace);
        this.justPlaced = null;
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

    async animateRoll(targetDice) {
        this.nnRollingFlags = [true, true, true, true, true];
        this.renderAll();
        await sleep(T.rollSpin);
        this.displayDice = (targetDice || []).slice();
        this.nnRollingFlags = null;
        this.renderAll();
    },

    // -----------------------------------------------------------------
    // Human reroll
    // -----------------------------------------------------------------
    async humanReroll() {
        if (this.animating || !this.serverState || !this.serverState.waiting_for_human) return;
        if (this.serverState.rolls_left <= 0) return;

        this.animating = true;
        const mask = this.holdMask;
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
    },

    // -----------------------------------------------------------------
    // Human placement — server's /place chains all bot turns up to the
    // next human prompt (or game end). In 2v2 that's up to 3 bot turns.
    // -----------------------------------------------------------------
    async humanPlace(column, row) {
        if (this.animating || !this.serverState || !this.serverState.waiting_for_human) return;
        const placements = (this.options && this.options.placements) || [];
        const opt = placements.find((p) => p.column === column && p.row === row);
        if (!opt) return;

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

        const oldLen = this.serverState.history.length;
        const newTurns = data.history.slice(oldLen);
        const botTurns = newTurns.filter((t) => !this.isHuman(t.player));

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

        await this.driveLoop();
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
        if (this.isHuman(cur)) {
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
        const isHumanTurn = gs && gs.waiting_for_human && !this.animating;
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

    flashCell(player, column, row) {
        // Both variants use a single combined sheet; the cell flashes
        // regardless of whose value is going in (the current player's
        // board is in main, so the just-targeted cell is always main).
        const target = document.querySelector(`[data-cell="c${column}-r${row}"]`);
        if (!target) return;
        target.classList.add('target-flash');
        setTimeout(() => target.classList.remove('target-flash'), T.placeFlash);
    },

    showEndgame() {
        const gs = this.serverState;
        if (!gs || !gs.game_over) return;
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

        document.getElementById('sheet').hidden = true;
        document.getElementById('footer').hidden = true;
        document.getElementById('endgame').hidden = false;
    },
};

// ---------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------

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
