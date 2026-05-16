// Pro Yams Play — human vs. NN.
//
// 1v1: one combined sheet (you/NN overlay), the human plays seat 0 or 1.
// 2v2: 2×2 grid of four sheets; the human plays seat 0 (Team 0), the other
//      three seats default to NN. computeDuel and the endgame breakdown
//      iterate every cross-team pairing the same way compute_duel<Yams2v2>
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
    humanSeat: 0,
    numPlayers: 2,
    variant: '1v1',
    options: null,
    animating: false,
    justPlaced: null,
    hasShownDice: false,
    animationPlayer: null,

    isHuman(p) { return p === this.humanSeat; },
    isHumanTeam(p) {
        return areTeammates(p, this.humanSeat, this.numPlayers);
    },

    init() {
        document.getElementById('btn-play').addEventListener('click', () => Play.start());
        document.getElementById('btn-again').addEventListener('click', () => Play.start());
        document.getElementById('btn-reroll').addEventListener('click', () => Play.humanReroll());
        document.getElementById('header').addEventListener('click', () => Play.toSplash());
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
        this.setTurnInfo('');
    },

    async start() {
        if (this.sessionId) { API.deleteGame(this.sessionId); this.sessionId = null; }
        this.holdMask = 0;
        this.nnHeldMask = 0;
        this.nnRollingFlags = null;
        this.animating = false;
        this.justPlaced = null;
        this.options = null;
        this.hasShownDice = false;
        this.displayDice = [0, 0, 0, 0, 0];

        // Variant selection: opt-in via URL hash (#2v2). Default 1v1.
        const want2v2 = window.location.hash.indexOf('2v2') >= 0;

        let playerTypes;
        if (want2v2) {
            // Human plays seat 0; the other three seats default to NN.
            this.humanSeat = 0;
            this.numPlayers = 4;
            this.variant = '2v2';
            playerTypes = ['human', 'nn', 'nn', 'nn'];
        } else {
            const humanIsP0 = Math.random() < 0.5;
            this.humanSeat = humanIsP0 ? 0 : 1;
            this.numPlayers = 2;
            this.variant = '1v1';
            playerTypes = humanIsP0 ? ['human', 'nn'] : ['nn', 'human'];
        }
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
        this.setTurnInfo('Your turn', 'you');
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

        // Optimistic local update for the human's own cell.
        const colKey = COLUMN_KEYS[column];
        const rowKey = ROW_NAMES[row];
        this.displayBoards['player' + this.humanSeat][colKey][rowKey] = opt.score;
        this.justPlaced = { player: this.humanSeat, column, row };
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
        const botTurns = newTurns.filter((t) => t.player !== this.humanSeat);

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
        if (this.isHuman(seat)) return 'Your turn';
        if (this.numPlayers === 4) {
            const team = teamOf(seat, 4);
            const role = this.isHumanTeam(seat) ? 'Teammate' : 'NN';
            return `${role} (P${seat}) — Team ${team}`;
        }
        return 'NN turn';
    },
    turnClassFor(seat) {
        if (this.isHuman(seat)) return 'you';
        return this.isHumanTeam(seat) ? 'you' : 'nn';
    },

    renderTurnInfo() {
        const gs = this.serverState;
        if (!gs) { this.setTurnInfo(''); return; }
        if (gs.game_over) return;  // showEndgame handles the final label
        const cur = (this.animationPlayer !== null && this.animationPlayer !== undefined)
                    ? this.animationPlayer : gs.current_player;
        if (this.isHuman(cur)) {
            const rollNum = 3 - gs.rolls_left + 1;
            this.setTurnInfo(`Your turn · Roll ${rollNum}/3`, 'you');
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
        container.className = 'sheet ' +
            (this.isHuman(cur) ? 'turn-you' : 'turn-nn');

        const isHumanTurn = gs.waiting_for_human && !this.animating;
        const legalPlacements = (isHumanTurn && this.options) ? this.options.placements : null;
        const { legalSet, legalScores } = this.buildLegalLookup(legalPlacements);

        const grid = this.buildBoardGrid(
            this.displayBoards['player' + cur],
            this.displayBoards['player' + opp],
            legalSet, legalScores,
            (c, r) => this.humanPlace(c, r));
        container.appendChild(grid);
    },

    // ---------- 2v2: four independent sheets in a 2x2 grid ----------------
    renderSheets2v2() {
        const container = document.getElementById('sheet');
        container.innerHTML = '';
        container.className = 'sheet sheets-2v2';
        if (!this.displayBoards || !this.serverState) return;

        const gs = this.serverState;
        const isHumanTurn = gs.waiting_for_human && !this.animating;
        const legalPlacements = (isHumanTurn && this.options) ? this.options.placements : null;
        const { legalSet, legalScores } = this.buildLegalLookup(legalPlacements);

        const cur = (this.animationPlayer !== null && this.animationPlayer !== undefined)
                    ? this.animationPlayer : gs.current_player;

        const wrapper = document.createElement('div');
        wrapper.className = 'sheets-grid';
        // Layout order: [P0 P1 / P2 P3] — Team 0 (P0/P2) left, Team 1 (P1/P3) right.
        const order = [0, 1, 2, 3];
        for (const seat of order) {
            const slot = document.createElement('div');
            slot.className = 'mini-sheet';
            slot.classList.add(teamOf(seat, 4) === 0 ? 'team-0' : 'team-1');
            if (seat === cur) slot.classList.add('current');
            if (this.isHuman(seat)) slot.classList.add('is-human');

            const header = document.createElement('div');
            header.className = 'mini-sheet-header';
            const role = this.isHuman(seat) ? 'You'
                      : (this.isHumanTeam(seat) ? `Teammate` : `NN`);
            header.textContent = `P${seat} · ${role} · Team ${teamOf(seat, 4)}`;
            slot.appendChild(header);

            // Only the human's sheet shows legal placements when it's their turn.
            const ownLegalSet = this.isHuman(seat) && seat === cur ? legalSet : new Set();
            const ownLegalScores = ownLegalSet.size > 0 ? legalScores : {};

            const grid = this.buildBoardGrid(
                this.displayBoards['player' + seat],
                /* topBoard */ null,
                ownLegalSet, ownLegalScores,
                (c, r) => this.humanPlace(c, r));
            slot.appendChild(grid);
            wrapper.appendChild(slot);
        }
        container.appendChild(wrapper);
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
    buildBoardGrid(mainBoard, topBoard, legalSet, legalScores, onPlace) {
        const grid = document.createElement('div');
        grid.className = 'board-grid';

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
                const topVal  = topBoard?.[colKey]?.[rowKey];

                const topEl = document.createElement('div');
                topEl.className = 'cell-top';
                if (!topBoard || topVal === undefined || topVal === -1) {
                    topEl.textContent = '';
                } else if (topVal === 0) {
                    topEl.textContent = '0';
                    topEl.classList.add('scratched');
                } else {
                    topEl.textContent = topVal;
                }
                cell.appendChild(topEl);

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

                    const { sum: mSum, allFilled: mFilled } = upperColumnSum(mainBoard, colKey);
                    const topEl = document.createElement('div');
                    topEl.className = 'cell-top';
                    if (topBoard) {
                        const { sum: tSum, allFilled: tFilled } = upperColumnSum(topBoard, colKey);
                        topEl.textContent = tFilled ? `${tSum}/${upperBonus(tSum)}`
                                                   : (tSum > 0 ? String(tSum) : '');
                    }
                    bonusCell.appendChild(topEl);

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

    flashCell(player, column, row) {
        // The 1v1 combined sheet has one cell per (col,row); the cell flashes
        // regardless of whose value is going in. The 2v2 layout has separate
        // sheets, so we target the per-seat cell when available.
        const perSeat = document.querySelector(
            `.mini-sheet[data-seat="${player}"] [data-cell="c${column}-r${row}"]`);
        const target = perSeat || document.querySelector(
            `[data-cell="c${column}-r${row}"]`);
        if (!target) return;
        target.classList.add('target-flash');
        setTimeout(() => target.classList.remove('target-flash'), T.placeFlash);
    },

    showEndgame() {
        const gs = this.serverState;
        if (!gs || !gs.game_over) return;
        const r = gs.result;
        const humanTeam = teamOf(this.humanSeat, this.numPlayers);
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
        const t0Adj = col.players[0].adjusted + col.players[2].adjusted;
        const t1Adj = col.players[1].adjusted + col.players[3].adjusted;
        const youAdj = humanTeam === 0 ? t0Adj : t1Adj;
        const nnAdj  = humanTeam === 0 ? t1Adj : t0Adj;
        const humanPts = humanTeam === 0 ? col.duelPts : -col.duelPts;
        const cls = humanPts > 0 ? 'pts-pos' : humanPts < 0 ? 'pts-neg' : 'pts-zero';
        html += `<tr>
            <td>${COLUMN_NAMES[col.c]}<span class="col-coeff">×${col.coeff}</span></td>
            <td>${youAdj}</td>
            <td>${nnAdj}</td>
            <td class="${cls}">${humanPts === 0 ? '0' : fmtSigned(humanPts)}</td>
          </tr>`;
    }
    const totCls = humanTotal > 0 ? 'pts-pos' : humanTotal < 0 ? 'pts-neg' : 'pts-zero';
    html += `</tbody>
        <tfoot>
          <tr>
            <td colspan="3" style="text-align:right; color: var(--muted);">Team total</td>
            <td class="${totCls}">${fmtSigned(humanTotal)}</td>
          </tr>
        </tfoot>
      </table>`;
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

document.addEventListener('DOMContentLoaded', () => Play.init());
