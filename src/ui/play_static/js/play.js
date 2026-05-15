// Pro Yams Play — human vs. NN, single combined sheet, sticky dice footer.

const COLUMN_NAMES = ['Down', 'Free', 'Up', 'Mid', 'Turbo', 'UpDown'];
const COLUMN_KEYS  = ['down',  'free', 'up', 'mid', 'turbo', 'updown'];
const ROW_NAMES    = ['1s','2s','3s','4s','5s','6s','SS','LS','FH','K','STR','U8','Y'];

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Animation timing (ms). Tweak to taste.
const T = {
    preRollPause:    1400,  // pause showing previous player's final dice before next roll
    rollSpin:         500,  // dice rolling spin
    nnHoldHighlight: 1000,  // "NN is keeping these dice" highlight
    nnAfterRoll:      450,  // pause after a reroll resolves
    nnBeforePlace:    900,  // pause on final dice before placing
    placeFlash:       700,  // target-cell flash before committing the placement
    afterPlace:       800,  // pause after placement (lets the bounce play out before any layout swap)
};

const Play = {
    sessionId: null,
    serverState: null,
    displayBoards: null,
    displayDice: null,
    nnHeldMask: 0,
    nnRollingFlags: null,
    holdMask: 0,
    humanPlayer: 0,
    nnPlayer: 1,
    options: null,
    animating: false,
    justPlaced: null,
    hasShownDice: false,  // false until the very first roll reveal
    animationPlayer: null,  // when set, overrides current_player for sheet layout

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
        this.displayDice = [0, 0, 0, 0, 0];  // placeholders until first roll resolves

        const humanIsP0 = Math.random() < 0.5;
        this.humanPlayer = humanIsP0 ? 0 : 1;
        this.nnPlayer    = humanIsP0 ? 1 : 0;
        const p0 = humanIsP0 ? 'human' : 'nn';
        const p1 = humanIsP0 ? 'nn'    : 'human';

        document.getElementById('splash').hidden = true;
        document.getElementById('endgame').hidden = true;
        document.getElementById('game').hidden = false;
        document.getElementById('footer').hidden = false;
        document.getElementById('sheet').hidden = false;

        const data = await API.newGame(p0, p1);
        this.sessionId    = data.session_id;
        this.serverState  = data.game_state;
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
    // NN turn — pre-pause showing previous player's final dice, then
    // animate roll, hold steps, placement.
    // -----------------------------------------------------------------
    async runNnTurn() {
        this.animating = true;
        this.animationPlayer = this.nnPlayer;
        this.setTurnInfo('NN thinking…', 'nn');
        this.renderAll();

        // Pause so the user can see the previous player's final dice.
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
        const nnTurn = newTurns.find((t) => t.player === this.nnPlayer);

        // Hide NN's just-placed cell from displayBoards so the animation
        // reveals it at the right moment.
        const newDisplayBoards = cloneBoards(data.boards);
        if (nnTurn) {
            const colKey = COLUMN_KEYS[nnTurn.placement.column];
            const rowKey = ROW_NAMES[nnTurn.placement.row];
            newDisplayBoards['player' + this.nnPlayer][colKey][rowKey] = -1;
        }
        this.displayBoards = newDisplayBoards;
        this.serverState = data;

        if (nnTurn) {
            // 1. Reveal initial roll.
            await this.animateRoll(nnTurn.initial_dice);
            this.hasShownDice = true;

            // 2. Walk through each hold/reroll step.
            for (const hold of (nnTurn.holds || [])) {
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

            // 3. Pause on final dice, then flash + commit placement.
            this.nnHeldMask = 0;
            this.renderAll();
            await sleep(T.nnBeforePlace);

            const colKey = COLUMN_KEYS[nnTurn.placement.column];
            const rowKey = ROW_NAMES[nnTurn.placement.row];
            this.flashCell(this.nnPlayer, nnTurn.placement.column, nnTurn.placement.row);
            await sleep(T.placeFlash);

            this.displayBoards['player' + this.nnPlayer][colKey][rowKey] = nnTurn.score;
            this.justPlaced = { player: this.nnPlayer, column: nnTurn.placement.column, row: nnTurn.placement.row };
            this.renderAll();
            await sleep(T.afterPlace);
            this.justPlaced = null;
        }

        // Important: do NOT touch displayDice — keep NN's final dice on screen.
        this.nnHeldMask = 0;
        this.nnRollingFlags = null;
        this.animating = false;
        this.renderAll();
    },

    // -----------------------------------------------------------------
    // Human turn start — pre-pause showing NN's final dice, then roll
    // animation reveals the human's new initial dice.
    // -----------------------------------------------------------------
    async beginHumanTurn() {
        this.animating = true;
        // Hold off on the layout swap so the prior dice are still shown from
        // the NN's perspective during the pre-roll pause; swap right before
        // the roll animation reveals the human's dice.
        this.setTurnInfo('Your turn', 'you');
        this.renderAll();

        if (this.hasShownDice) {
            await sleep(T.preRollPause);
        }
        this.animationPlayer = null;  // swap layout to human's perspective
        await this.animateRoll(this.serverState.dice);
        this.hasShownDice = true;

        await this.refreshHumanOptions();
        this.animating = false;
        this.renderAll();
    },

    async refreshHumanOptions() {
        this.options = await API.getOptions(this.sessionId);
    },

    // Spin all dice, then resolve to targetDice.
    async animateRoll(targetDice) {
        this.nnRollingFlags = [true, true, true, true, true];
        this.renderAll();
        await sleep(T.rollSpin);
        this.displayDice = (targetDice || []).slice();
        this.nnRollingFlags = null;
        this.renderAll();
    },

    // -----------------------------------------------------------------
    // Human reroll (mid-turn)
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
    // Human placement
    // -----------------------------------------------------------------
    async humanPlace(column, row) {
        if (this.animating || !this.serverState || !this.serverState.waiting_for_human) return;
        const placements = (this.options && this.options.placements) || [];
        const opt = placements.find((p) => p.column === column && p.row === row);
        if (!opt) return;

        this.animating = true;

        // Optimistic local update.
        const colKey = COLUMN_KEYS[column];
        const rowKey = ROW_NAMES[row];
        this.displayBoards['player' + this.humanPlayer][colKey][rowKey] = opt.score;
        this.justPlaced = { player: this.humanPlayer, column, row };
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

        // /place may have advanced the NN's turn. Animate that turn if so.
        const oldLen = this.serverState.history.length;
        const newTurns = data.history.slice(oldLen);
        const nnTurn = newTurns.find((t) => t.player === this.nnPlayer);

        const newDisplayBoards = cloneBoards(data.boards);
        if (nnTurn) {
            const cKey = COLUMN_KEYS[nnTurn.placement.column];
            const rKey = ROW_NAMES[nnTurn.placement.row];
            newDisplayBoards['player' + this.nnPlayer][cKey][rKey] = -1;
        }
        this.displayBoards = newDisplayBoards;
        this.serverState = data;

        if (nnTurn) {
            // Pause showing human's final dice, then NN's animation begins
            // with a roll revealing its initial dice.
            this.animationPlayer = this.nnPlayer;  // swap sheet to NN perspective
            this.setTurnInfo('NN thinking…', 'nn');
            this.renderAll();
            await sleep(T.preRollPause);
            await this.animateRoll(nnTurn.initial_dice);

            for (const hold of (nnTurn.holds || [])) {
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

            const cKey = COLUMN_KEYS[nnTurn.placement.column];
            const rKey = ROW_NAMES[nnTurn.placement.row];
            this.flashCell(this.nnPlayer, nnTurn.placement.column, nnTurn.placement.row);
            await sleep(T.placeFlash);

            this.displayBoards['player' + this.nnPlayer][cKey][rKey] = nnTurn.score;
            this.justPlaced = { player: this.nnPlayer, column: nnTurn.placement.column, row: nnTurn.placement.row };
            this.renderAll();
            await sleep(T.afterPlace);
            this.justPlaced = null;
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

    renderTurnInfo() {
        const gs = this.serverState;
        if (!gs) { this.setTurnInfo(''); return; }
        if (gs.game_over) {
            // Final label set by showEndgame().
            return;
        }
        const cur = (this.animationPlayer !== null && this.animationPlayer !== undefined)
                    ? this.animationPlayer : gs.current_player;
        const isHuman = cur === this.humanPlayer;
        if (isHuman) {
            const rollNum = 3 - gs.rolls_left + 1;
            this.setTurnInfo(`Your turn · Roll ${rollNum}/3`, 'you');
        } else {
            this.setTurnInfo('NN turn', 'nn');
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
        const container = document.getElementById('sheet');
        container.innerHTML = '';
        if (!this.displayBoards || !this.serverState) return;

        const gs = this.serverState;
        // The "main" slot follows the active animation player when one is set
        // (e.g. while we're animating an NN turn whose state has already
        // advanced current_player back to the human), otherwise it follows
        // the live current_player.
        const cur = (this.animationPlayer !== null && this.animationPlayer !== undefined)
                    ? this.animationPlayer : gs.current_player;
        const opp = 1 - cur;
        // The sheet's "main" slot follows the current player so the swap is
        // visible to the user.
        container.className = 'sheet ' +
            (cur === this.humanPlayer ? 'turn-you' : 'turn-nn');

        const isHumanTurn = gs.waiting_for_human && !this.animating;
        const legalPlacements = (isHumanTurn && this.options) ? this.options.placements : null;
        const legalSet = new Set();
        const legalScores = {};
        if (legalPlacements) {
            for (const p of legalPlacements) {
                legalSet.add(`${p.column},${p.row}`);
                legalScores[`${p.column},${p.row}`] = p.score;
            }
        }

        const grid = document.createElement('div');
        grid.className = 'board-grid';

        // Top-left corner.
        const corner = document.createElement('div');
        corner.className = 'row-label';
        grid.appendChild(corner);

        const coefficients = gs.coefficients || [];
        for (let c = 0; c < 6; c++) {
            const hdr = document.createElement('div');
            hdr.className = 'col-header';
            const coeff = coefficients[c];
            hdr.innerHTML = `${COLUMN_NAMES[c]}` +
                (coeff !== undefined ? `<span class="col-coeff">×${coeff}</span>` : '');
            grid.appendChild(hdr);
        }

        const mainBoard = this.displayBoards['player' + cur];
        const topBoard  = this.displayBoards['player' + opp];

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
                if (topVal === undefined || topVal === -1) {
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
                const isLegal = legalSet.has(legalKey);
                if (isLegal) {
                    cell.classList.add('legal');
                    const sc = legalScores[legalKey];
                    mainEl.textContent = (sc !== undefined ? sc : '');
                    cell.addEventListener('click', () => this.humanPlace(c, r));
                } else if (mainVal === undefined || mainVal === -1) {
                    mainEl.textContent = '';
                } else if (mainVal === 0) {
                    mainEl.textContent = '0';
                    mainEl.classList.add('scratched');
                } else {
                    mainEl.textContent = mainVal;
                }
                cell.appendChild(mainEl);

                if (this.justPlaced && this.justPlaced.column === c && this.justPlaced.row === r) {
                    cell.classList.add('just-placed');
                }
                grid.appendChild(cell);
            }

            // Bonus row after the 6s.
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
                    const { sum: tSum, allFilled: tFilled } = upperColumnSum(topBoard,  colKey);

                    const topEl = document.createElement('div');
                    topEl.className = 'cell-top';
                    topEl.textContent = tFilled ? `${tSum}/${upperBonus(tSum)}` : (tSum > 0 ? String(tSum) : '');
                    bonusCell.appendChild(topEl);

                    const mainEl = document.createElement('div');
                    mainEl.className = 'cell-main';
                    mainEl.textContent = mFilled ? `${mSum}/${upperBonus(mSum)}` : (mSum > 0 ? String(mSum) : '');
                    bonusCell.appendChild(mainEl);

                    grid.appendChild(bonusCell);
                }
            }
        }
        container.appendChild(grid);
    },

    flashCell(player, column, row) {
        // The combined sheet has one cell per (col,row); the cell flashes
        // regardless of whose value is going in.
        const sel = `[data-cell="c${column}-r${row}"]`;
        const el = document.querySelector(sel);
        if (!el) return;
        el.classList.add('target-flash');
        setTimeout(() => el.classList.remove('target-flash'), T.placeFlash);
    },

    showEndgame() {
        const gs = this.serverState;
        if (!gs || !gs.game_over) return;
        const r = gs.result;
        let cls, text;
        if (r === 0.5) { cls = 'draw'; text = 'Draw!'; }
        else {
            const humanWon = (r === 1.0 && this.humanPlayer === 0) ||
                             (r === 0.0 && this.humanPlayer === 1);
            cls  = humanWon ? 'win'  : 'loss';
            text = humanWon ? 'You win! 🎉' : 'NN wins.';
        }
        this.setTurnInfo(text, cls);

        const duel = computeDuel(gs);  // { totalDuel: P0 - P1, colData: [...] }
        const humanIsP0 = (this.humanPlayer === 0);
        const humanTotal = humanIsP0 ? duel.totalDuel : -duel.totalDuel;

        // Headline.
        const txtEl = document.getElementById('endgame-text');
        txtEl.textContent = text;
        txtEl.className = 'endgame-text ' + cls;

        // Margin line.
        const marginEl = document.getElementById('endgame-margin');
        const marginCls = humanTotal > 0 ? 'pts-pos'
                        : humanTotal < 0 ? 'pts-neg' : 'pts-zero';
        marginEl.innerHTML = `Margin: <span class="${marginCls}">${fmtSigned(humanTotal)}</span> pts`;

        // Per-column table.
        const tbl = document.getElementById('endgame-breakdown');
        tbl.innerHTML = renderBreakdownTable(duel.colData, humanIsP0, humanTotal);

        // Lay out the result view: hide sheet + footer, show endgame.
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

// Mirror of compute_duel() in src/engine/duel.cc.
// Returns { totalDuel: P0 - P1 weighted sum, colData: per-column details }.
function computeDuel(gs) {
    const upperBonusFn = (sum) => {
        if (sum >= 100) return 500;
        if (sum >=  90) return 200;
        if (sum >=  80) return 100;
        if (sum >=  70) return  50;
        if (sum >=  60) return  30;
        return 0;
    };
    const crushMult = (my, opp) => {
        if (opp === 0 && my > 0)            return 5;
        if (opp > 0 && my >= 5 * opp)       return 5;
        if (opp > 0 && my >= 4 * opp)       return 4;
        if (opp > 0 && my >= 3 * opp)       return 3;
        if (opp > 0 && my >= 2 * opp)       return 2;
        return 1;
    };
    const UPPER = ROW_NAMES.slice(0, 6);
    const LOWER = ROW_NAMES.slice(6);
    const boards = [gs.boards.player0, gs.boards.player1];
    const coeffs = gs.coefficients || [];

    let totalDuel = 0;
    const colData = [];
    for (let c = 0; c < 6; c++) {
        const colKey = COLUMN_KEYS[c];
        const coeff  = coeffs[c];
        const players = [];
        for (let p = 0; p < 2; p++) {
            const b = boards[p];
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
            const uBonus = upperBonusFn(upperSum);
            players.push({ upperSum, cellsSum, uBonus,
                           raw: cellsSum + uBonus, lowerHasScratch });
        }
        const crush0 = crushMult(players[0].raw, players[1].raw);
        const crush1 = crushMult(players[1].raw, players[0].raw);
        const activeCrush = Math.max(crush0, crush1);
        const cleanBonus = activeCrush > 1 ? 100 : 200;
        for (const pl of players) {
            pl.isClean  = pl.upperSum >= 60 && !pl.lowerHasScratch;
            pl.cBonus   = pl.isClean ? cleanBonus : 0;
            pl.adjusted = pl.raw + pl.cBonus;
        }
        const duelPts = (players[0].adjusted - players[1].adjusted) *
                        activeCrush * coeff;
        totalDuel += duelPts;
        colData.push({ c, colKey, coeff, players, activeCrush, duelPts });
    }
    return { totalDuel, colData };
}

// Render the per-column breakdown as a compact HTML table, from the human's
// perspective: Δ is positive when the human wins the column.
function renderBreakdownTable(colData, humanIsP0, humanTotal) {
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
        html += `<tr>
            <td>${COLUMN_NAMES[col.c]}<span class="col-coeff">×${col.coeff}</span></td>
            <td>${youAdj}</td>
            <td>${nnAdj}</td>
            <td>×${col.activeCrush}</td>
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
