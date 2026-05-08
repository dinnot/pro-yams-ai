// REST API client for Pro Yams UI.

const API = {
    // Game management
    async newGame(player0, player1, seed, debugMode = false) {
        const body = { player0, player1, debug_mode: debugMode };
        if (seed !== undefined && seed !== null && seed !== '') body.seed = Number(seed);
        else body.seed = Math.floor(Math.random() * 1000000);
        const res = await fetch('/api/game/new', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
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

    async playAll(id) {
        const res = await fetch(`/api/game/${id}/play_all`, { method: 'POST' });
        return res.json();
    },

    async deleteGame(id) {
        const res = await fetch(`/api/game/${id}`, { method: 'DELETE' });
        return res.json();
    },

    // Human interaction
    async getOptions(id) {
        const res = await fetch(`/api/game/${id}/options`);
        return res.json();
    },

    async hold(id, holdMask) {
        const res = await fetch(`/api/game/${id}/hold`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ hold_mask: holdMask })
        });
        return res.json();
    },

    async place(id, column, row) {
        const res = await fetch(`/api/game/${id}/place`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ column, row })
        });
        return res.json();
    },

    async canReroll(id) {
        const res = await fetch(`/api/game/${id}/can_reroll`);
        return res.json();
    },

    async getTensor(id, player) {
        const url = (player === 0 || player === 1)
            ? `/api/game/${id}/tensor?player=${player}`
            : `/api/game/${id}/tensor`;
        const res = await fetch(url);
        return res.json();
    },

    // Training logs
    async getTrainingLog(dir) {
        const url = dir ? `/api/logs/training?dir=${encodeURIComponent(dir)}` : '/api/logs/training';
        const res = await fetch(url);
        if (!res.ok) return null;
        return res.text();
    },

    async getEvalLog(dir) {
        const url = dir ? `/api/logs/eval?dir=${encodeURIComponent(dir)}` : '/api/logs/eval';
        const res = await fetch(url);
        if (!res.ok) return null;
        return res.text();
    },

    async getLogList() {
        const res = await fetch('/api/logs/list');
        return res.json();
    },

    // Tournament
    async listModels(dir) {
        const url = dir ? `/api/models/list?dir=${encodeURIComponent(dir)}` : '/api/models/list';
        const res = await fetch(url);
        return res.json();
    },

    async startTournament(participants, gamesPerMatchup) {
        const res = await fetch('/api/tournament/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                participants,
                games_per_matchup: gamesPerMatchup
            })
        });
        return res.json();
    },

    async getTournamentStatus() {
        const res = await fetch('/api/tournament/status');
        return res.json();
    },

    async stopTournament() {
        const res = await fetch('/api/tournament/stop', { method: 'POST' });
        return res.json();
    }
};
