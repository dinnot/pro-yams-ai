// Gameboard position notation for Pro Yams UI.
//
// A "position" captures everything needed to resume a game from an exact
// board state — like a chess FEN string. It does NOT include the dice; those
// are rolled fresh for the player on move when a position is loaded.
//
// Wire format (single line, copy-paste friendly):
//
//   PY1;<variant>;<c0,..,c5>;<current_player>;<board0>;<board1>[;<board2>;<board3>]
//
//   variant         "1v1" or "2v2"
//   c0..c5          column coefficients (the ×8..×18 multipliers)
//   current_player  seat index whose turn it is next
//   boardN          6 columns separated by '/', each column = 13 cell values
//                   separated by ',', in COLUMN_KEYS / ROW_NAMES order.
//                   Cell value: '-' = empty, '0' = scratched, 1..100 = score.
//
// The structured object form (used to talk to the backend) mirrors the board
// JSON the server already emits: { variant, coefficients, current_player,
// boards: { player0: { down: { "1s": v, ... }, ... }, ... } }.

const Position = {
    MAGIC: 'PY1',

    numPlayersForVariant(variant) {
        return variant === '2v2' ? 4 : 2;
    },

    emptyGrid() {
        const grid = {};
        for (let c = 0; c < COLUMN_KEYS.length; c++) {
            const col = {};
            for (let r = 0; r < ROW_NAMES.length; r++) col[ROW_NAMES[r]] = -1;
            grid[COLUMN_KEYS[c]] = col;
        }
        return grid;
    },

    // Deep-copy a boards object so replaying placements never mutates the base.
    cloneBoards(boards, numPlayers) {
        const out = {};
        for (let p = 0; p < numPlayers; p++) {
            const key = 'player' + p;
            const src = boards && boards[key];
            const grid = Position.emptyGrid();
            if (src) {
                for (let c = 0; c < COLUMN_KEYS.length; c++) {
                    const colKey = COLUMN_KEYS[c];
                    const srcCol = src[colKey];
                    if (!srcCol) continue;
                    for (let r = 0; r < ROW_NAMES.length; r++) {
                        const rowKey = ROW_NAMES[r];
                        const v = srcCol[rowKey];
                        grid[colKey][rowKey] = (v == null) ? -1 : v;
                    }
                }
            }
            out[key] = grid;
        }
        return out;
    },

    // Reconstruct the board position as it stood right after turn `turnIndex`
    // (inclusive) by replaying every placement up to that point. Turn order is
    // round-robin, so the player to move next is (last player + 1) % numPlayers.
    //
    // `baseBoards` is the board state the session started from (the cells of a
    // loaded position). When omitted — a fresh game — replay begins from empty
    // grids. Without it, copying a loaded position would drop every cell that
    // was present before the first in-session placement.
    fromHistory(history, coefficients, variant, turnIndex, baseBoards) {
        const numPlayers = Position.numPlayersForVariant(variant);
        const boards = baseBoards
            ? Position.cloneBoards(baseBoards, numPlayers)
            : (() => {
                const b = {};
                for (let p = 0; p < numPlayers; p++) b['player' + p] = Position.emptyGrid();
                return b;
            })();

        for (let t = 0; t <= turnIndex && t < history.length; t++) {
            const turn = history[t];
            const colKey = COLUMN_KEYS[turn.placement.column];
            const rowKey = ROW_NAMES[turn.placement.row];
            boards['player' + turn.player][colKey][rowKey] = turn.score;
        }

        const lastPlayer = history[turnIndex].player;
        return {
            variant,
            coefficients: coefficients.slice(0, COLUMN_KEYS.length),
            current_player: (lastPlayer + 1) % numPlayers,
            boards,
        };
    },

    encode(pos) {
        const numPlayers = Position.numPlayersForVariant(pos.variant);
        const parts = [
            Position.MAGIC,
            pos.variant,
            pos.coefficients.join(','),
            String(pos.current_player),
        ];
        for (let p = 0; p < numPlayers; p++) {
            const grid = pos.boards['player' + p];
            const cols = [];
            for (let c = 0; c < COLUMN_KEYS.length; c++) {
                const col = grid[COLUMN_KEYS[c]];
                const cells = [];
                for (let r = 0; r < ROW_NAMES.length; r++) {
                    const v = col[ROW_NAMES[r]];
                    cells.push(v === -1 || v == null ? '-' : String(v));
                }
                cols.push(cells.join(','));
            }
            parts.push(cols.join('/'));
        }
        return parts.join(';');
    },

    // Parse a notation string. Throws an Error with a human-readable message
    // on any malformed input.
    decode(str) {
        const fields = String(str).trim().split(';');
        if (fields[0] !== Position.MAGIC) {
            throw new Error('not a Pro Yams position (missing PY1 prefix)');
        }
        const variant = fields[1];
        if (variant !== '1v1' && variant !== '2v2') {
            throw new Error(`unknown variant "${variant}"`);
        }
        const numPlayers = Position.numPlayersForVariant(variant);
        if (fields.length !== 4 + numPlayers) {
            throw new Error(`expected ${4 + numPlayers} fields, got ${fields.length}`);
        }

        const coefficients = fields[2].split(',').map(Number);
        if (coefficients.length !== COLUMN_KEYS.length || coefficients.some(isNaN)) {
            throw new Error('coefficients must be 6 numbers');
        }
        const current_player = Number(fields[3]);
        if (!Number.isInteger(current_player) ||
            current_player < 0 || current_player >= numPlayers) {
            throw new Error(`current_player out of range: ${fields[3]}`);
        }

        const boards = {};
        for (let p = 0; p < numPlayers; p++) {
            const cols = fields[4 + p].split('/');
            if (cols.length !== COLUMN_KEYS.length) {
                throw new Error(`player ${p}: expected 6 columns, got ${cols.length}`);
            }
            const grid = {};
            for (let c = 0; c < COLUMN_KEYS.length; c++) {
                const cells = cols[c].split(',');
                if (cells.length !== ROW_NAMES.length) {
                    throw new Error(`player ${p} column ${c}: expected 13 cells, got ${cells.length}`);
                }
                const col = {};
                for (let r = 0; r < ROW_NAMES.length; r++) {
                    const tok = cells[r];
                    let v;
                    if (tok === '-' || tok === '') v = -1;
                    else {
                        v = Number(tok);
                        if (!Number.isInteger(v) || v < 0 || v > 100) {
                            throw new Error(`player ${p} column ${c} row ${r}: bad cell "${tok}"`);
                        }
                    }
                    col[ROW_NAMES[r]] = v;
                }
                grid[COLUMN_KEYS[c]] = col;
            }
            boards['player' + p] = grid;
        }

        return { variant, coefficients, current_player, boards };
    },

    // Copy text to the clipboard. Resolves true on success, false otherwise.
    async copyToClipboard(text) {
        try {
            if (navigator.clipboard && navigator.clipboard.writeText) {
                await navigator.clipboard.writeText(text);
                return true;
            }
        } catch (_) { /* fall through to legacy path */ }
        try {
            const ta = document.createElement('textarea');
            ta.value = text;
            ta.style.position = 'fixed';
            ta.style.opacity = '0';
            document.body.appendChild(ta);
            ta.select();
            const ok = document.execCommand('copy');
            document.body.removeChild(ta);
            return ok;
        } catch (_) {
            return false;
        }
    },
};
