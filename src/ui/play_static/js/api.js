// REST API client for the Pro Yams Play frontend.

const API = {
    async newGame(player0, player1) {
        const res = await fetch('/api/game/new', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                player0,
                player1,
                seed: Math.floor(Math.random() * 1000000),
                debug_mode: false,
            }),
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
