// REST API client for the Pro Yams Play frontend.

const API = {
    // The server's variant is fixed at launch (1v1 or 2v2). Probe it
    // before creating a game so the client sends the correct number of
    // player_types — otherwise missing seats default to heuristic.
    async info() {
        const res = await fetch('/api/info');
        return res.json();
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
        const res = await fetch('/api/game/new', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        return res.json();
    },

    async getGame(id) {
        const res = await fetch(`/api/game/${id}`);
        return res.json();
    },

    async step(id) {
        const res = await fetch(`/api/game/${id}/step`, { method: 'POST' });
        return res.json();
    },

    async getOptions(id) {
        const res = await fetch(`/api/game/${id}/options`);
        return res.json();
    },

    async hold(id, holdMask) {
        const res = await fetch(`/api/game/${id}/hold`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ hold_mask: holdMask }),
        });
        return res.json();
    },

    async place(id, column, row) {
        const res = await fetch(`/api/game/${id}/place`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ column, row }),
        });
        return res.json();
    },

    async deleteGame(id) {
        try {
            await fetch(`/api/game/${id}`, { method: 'DELETE' });
        } catch (_) { /* best-effort */ }
    },
};
