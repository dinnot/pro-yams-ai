// REST API client for the Pro Yams Play frontend.
//
// Fault tolerance: the server holds the authoritative game state, so every
// call is treated as recoverable.
//
//   * Read-only requests (info / get-game / options) are *idempotent* — they
//     are retried with capped exponential backoff on any network, timeout, or
//     5xx failure.
//   * Mutating requests (new-game / step / hold / place) are issued at most
//     once and *throw* on a hard network failure. Re-applying a mutation that
//     may already have landed server-side would corrupt the game, so the
//     caller recovers by re-syncing from GET /api/game/:id instead.
//
// Application-level errors (HTTP 4xx with an {error} body) are returned as
// parsed JSON so existing `data.error` handling keeps working; only transient
// transport failures throw.

const _sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const NET = {
    timeoutMs:     15000,  // abort a single request that hangs longer than this
    longPollMs:    35000,  // /longpoll is held open server-side (~25s) — give it
                           // room before the client aborts
    readRetries:   4,      // extra attempts for idempotent reads (total = +1).
                           // Kept modest so a recovery attempt stays bounded;
                           // play.js drives further retries on reconnect.
    newGameRetries: 3,     // new-game is safe enough to retry (worst case: an
                           // orphaned, never-played session)
    backoffBaseMs: 400,
    backoffMaxMs:  5000,
};

function _backoff(attempt) {
    const ceiling = Math.min(NET.backoffMaxMs, NET.backoffBaseMs * (2 ** attempt));
    // Half fixed, half jitter — avoids synchronized retry storms.
    return ceiling / 2 + Math.random() * (ceiling / 2);
}

// One fetch attempt with an abort timeout. Returns parsed JSON for any HTTP
// response (including 4xx, so {error} bodies pass through). Throws on network
// error, timeout, or a 5xx/429 status (treated as transient/retryable).
async function _fetchOnce(url, opts) {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), NET.timeoutMs);
    try {
        const res = await fetch(url, { ...opts, signal: ctrl.signal });
        if (res.status >= 500 || res.status === 429) {
            throw new Error(`HTTP ${res.status}`);
        }
        return await res.json();
    } finally {
        clearTimeout(timer);
    }
}

// Retry wrapper for idempotent requests (or near-idempotent ones like
// new-game where a retried duplicate is harmless).
async function _fetchRetry(url, opts, retries) {
    let lastErr;
    for (let attempt = 0; attempt <= retries; attempt++) {
        try {
            return await _fetchOnce(url, opts);
        } catch (e) {
            lastErr = e;
            if (attempt < retries) await _sleep(_backoff(attempt));
        }
    }
    throw lastErr || new Error('request failed');
}

function _get(url) {
    return _fetchRetry(url, { method: 'GET' }, NET.readRetries);
}

// At-most-once mutation: a single attempt with a timeout. Throws on failure.
function _post(url, body) {
    const opts = { method: 'POST' };
    if (body !== undefined) {
        opts.headers = { 'Content-Type': 'application/json' };
        opts.body = JSON.stringify(body);
    }
    return _fetchOnce(url, opts);
}

const API = {
    // The server's variant is fixed at launch (1v1 or 2v2). Probe it
    // before creating a game so the client sends the correct number of
    // player_types — otherwise missing seats default to heuristic.
    async info() {
        return _get('/api/info');
    },

    // playerTypes is an array of strings (one per seat: 'human', 'nn',
    // 'heuristic_v2', etc.) The server reads player0..playerN-1 keys and
    // defaults missing seats to heuristic — so the array length must
    // match the server's variant (2 entries → 1v1, 4 entries → 2v2).
    async newGame(playerTypes) {
        const body = {
            seed: Math.floor(Math.random() * 1000000),
            debug_mode: false,
        };
        for (let i = 0; i < playerTypes.length; i++) {
            body['player' + i] = playerTypes[i];
        }
        return _fetchRetry('/api/game/new', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        }, NET.newGameRetries);
    },

    // Authoritative game state — the recovery anchor. Always retried.
    async getGame(id) {
        return _get(`/api/game/${id}`);
    },

    // Long-poll for the next state change (shared games). The server blocks until
    // the version differs from `sinceVersion` or ~25s elapse, then returns the
    // full state plus its `version`. Pass null/undefined for an immediate fetch.
    // Single attempt with a long abort timeout; the caller (sharedLoop) re-issues.
    async longPoll(id, sinceVersion) {
        const q = (sinceVersion === null || sinceVersion === undefined)
            ? '' : `?since=${sinceVersion}`;
        const ctrl = new AbortController();
        const timer = setTimeout(() => ctrl.abort(), NET.longPollMs);
        try {
            const res = await fetch(`/api/game/${id}/longpoll${q}`, {
                method: 'GET', signal: ctrl.signal,
            });
            if (res.status >= 500 || res.status === 429) throw new Error(`HTTP ${res.status}`);
            return await res.json();
        } finally {
            clearTimeout(timer);
        }
    },

    // Per-seat luck for a finished game: { seat_luck: [v0, v1, ...] } where each
    // entry is a mean-zero index in [-100,100] (null for seats with no turns).
    async getLuck(id) {
        return _get(`/api/game/${id}/luck`);
    },

    async step(id) {
        return _post(`/api/game/${id}/step`);
    },

    async botStep(id) {
        return _post(`/api/game/${id}/bot_step`);
    },

    async getOptions(id) {
        return _get(`/api/game/${id}/options`);
    },

    // Advance all pending bot turns up to the next human turn (or game end).
    // Used by the shared two-human loop to push NN seats forward.
    async playAll(id) {
        return _post(`/api/game/${id}/play_all`);
    },

    // --- Matchmaking (2v2 "play with a friend") ---
    // Join the queue. Returns {status:'matched', session_id, seat} if a partner
    // was already waiting, otherwise {status:'waiting', ticket}. Retried like a
    // read: a duplicate join just creates a harmless extra waiting ticket.
    async matchmakeJoin() {
        return _fetchRetry('/api/matchmake/join', { method: 'POST' }, NET.newGameRetries);
    },
    async matchmakePoll(ticket) {
        return _get(`/api/matchmake/${ticket}`);
    },
    async matchmakeCancel(ticket) {
        try {
            return await _fetchOnce(`/api/matchmake/${ticket}/cancel`, { method: 'POST' });
        } catch (e) { return { error: String(e) }; }
    },

    // Stream the active player's tentative hold selection so a shared-game
    // teammate can watch the hold pattern form. Fire-and-forget.
    async holdPreview(id, seat, mask) {
        try {
            await _fetchOnce(`/api/game/${id}/hold_preview`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ seat, mask }),
            });
        } catch (_) { /* best-effort */ }
    },

    // Signal that the active player committed a reroll keeping `mask` (the new
    // dice are still in flight) so a spectating teammate can spin immediately.
    // Fire-and-forget.
    async rolling(id, seat, mask) {
        try {
            await _fetchOnce(`/api/game/${id}/rolling`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ seat, mask }),
            });
        } catch (_) { /* best-effort */ }
    },

    // Surviving teammate chooses to hand a disconnected seat to the NN. Returns
    // the updated game state, or {error} if the seat isn't a disconnected human.
    async takeover(id, seat) {
        return _post(`/api/game/${id}/takeover`, { seat });
    },

    // Report that the client controlling `seat` is still alive (shared games).
    // Fire-and-forget: failure just risks an early NN takeover, which recovers.
    async heartbeat(id, seat) {
        try {
            await _fetchOnce(`/api/game/${id}/heartbeat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ seat }),
            });
        } catch (_) { /* best-effort */ }
    },

    async hold(id, holdMask) {
        return _post(`/api/game/${id}/hold`, { hold_mask: holdMask });
    },

    async place(id, column, row) {
        return _post(`/api/game/${id}/place`, { column, row });
    },

    async deleteGame(id) {
        try {
            await _fetchOnce(`/api/game/${id}`, { method: 'DELETE' });
        } catch (_) { /* best-effort */ }
    },

    // Flag the just-finished game for human review (e.g. the AI played weird
    // moves). Best-effort but retried like a read, since a missed flag just
    // means the game isn't marked. Returns the parsed response or {error}.
    async flagGame(id, note) {
        try {
            return await _fetchRetry(`/api/game/${id}/flag`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ note: note || 'ai_weird_moves' }),
            }, NET.readRetries);
        } catch (e) {
            return { error: String(e) };
        }
    },
};
