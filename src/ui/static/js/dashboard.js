// Training dashboard for Pro Yams UI.

const Dashboard = {
    charts: {},
    autoRefreshTimer: null,

    init() {
        document.getElementById('btn-load-dash').addEventListener('click', () => Dashboard.load());
        document.getElementById('dash-auto-refresh').addEventListener('change', (e) => {
            if (e.target.checked) Dashboard.startAutoRefresh();
            else Dashboard.stopAutoRefresh();
        });
    },

    async load() {
        const dir = document.getElementById('dash-log-dir').value || undefined;
        const [trainCSV, evalCSV] = await Promise.all([
            API.getTrainingLog(dir),
            API.getEvalLog(dir)
        ]);
        Dashboard.renderTrainingLog(trainCSV);
        Dashboard.renderEvalLog(evalCSV, trainCSV);
    },

    renderTrainingLog(csv) {
        if (!csv) return;

        const rows = Dashboard.parseCSV(csv);
        if (rows.length === 0) return;

        const headers = Object.keys(rows[0]);

        // Find relevant columns.
        const stepCol  = headers.find(h => /step/i.test(h));
        const lossCol  = headers.find(h => /loss/i.test(h));
        const tempCol  = headers.find(h => /temp/i.test(h));
        const gamesCol = headers.find(h => /games/i.test(h) || /completed/i.test(h));
        const gpsCol   = headers.find(h => /^gps$/i.test(h) || /games_per_second/i.test(h));
        const samplesCol = headers.find(h => /total_samples/i.test(h) || /samples_trained/i.test(h));

        if (stepCol && lossCol) {
            Dashboard.updateChart('chart-loss', 'Training Loss',
                rows.map(r => r[stepCol]),
                rows.map(r => parseFloat(r[lossCol])),
                '#e94560');
        }

        if (stepCol && tempCol) {
            Dashboard.updateChart('chart-temperature', 'Temperature',
                rows.map(r => r[stepCol]),
                rows.map(r => parseFloat(r[tempCol])),
                '#ffc107');
        }

        if (stepCol && gamesCol) {
            Dashboard.updateChart('chart-games', 'Games Completed',
                rows.map(r => r[stepCol]),
                rows.map(r => parseFloat(r[gamesCol])),
                '#64b5f6');
        }
        
        if (stepCol && gpsCol) {
            Dashboard.updateChart('chart-games-per-sec', 'Simulated Games / sec',
                rows.map(r => r[stepCol]),
                rows.map(r => parseFloat(r[gpsCol])),
                '#26c6da');
        }

        // Update stat cards.
        Dashboard.updateStats(rows, lossCol, tempCol, gamesCol, stepCol, samplesCol);
    },

    renderEvalLog(evalCSV, trainCSV) {
        if (!evalCSV) return;

        const evalRows = Dashboard.parseCSV(evalCSV);
        if (evalRows.length === 0) return;

        const eh = Object.keys(evalRows[0]);
        const stepCol   = eh.find(h => /step/i.test(h));
        const wrCol     = eh.find(h => /win.*rate/i.test(h) || /wr/i.test(h));
        const marginCol = eh.find(h => /margin/i.test(h));
        const tsCol     = eh.find(h => /timestamp/i.test(h));

        if (stepCol && wrCol) {
            Dashboard.updateChart('chart-winrate', 'Eval Win Rate',
                evalRows.map(r => r[stepCol]),
                evalRows.map(r => parseFloat(r[wrCol])),
                '#4caf50');
        }

        if (stepCol && marginCol) {
            Dashboard.updateChart('chart-win-margin', 'Eval Win Margin (avg points)',
                evalRows.map(r => r[stepCol]),
                evalRows.map(r => parseFloat(r[marginCol])),
                '#ab47bc');
        }

        // (GPS derivation removed — now using dedicated field in training log)
    },

    updateStats(rows, lossCol, tempCol, gamesCol, stepCol, samplesCol) {
        const container = document.getElementById('current-stats');
        container.innerHTML = '';
        if (rows.length === 0) return;

        const last = rows[rows.length - 1];
        const stats = [];
        if (stepCol) stats.push({ label: 'Step', value: last[stepCol] });
        if (lossCol) stats.push({ label: 'Loss', value: parseFloat(last[lossCol]).toFixed(6) });
        if (tempCol) stats.push({ label: 'Temp', value: parseFloat(last[tempCol]).toFixed(4) });
        if (gamesCol) stats.push({ label: 'Games', value: last[gamesCol] });
        if (samplesCol) stats.push({ label: 'Samples', value: parseInt(last[samplesCol], 10).toLocaleString() });

        for (const s of stats) {
            const card = document.createElement('div');
            card.className = 'stat-card';
            card.innerHTML = `<div class="stat-label">${s.label}</div><div class="stat-value">${s.value}</div>`;
            container.appendChild(card);
        }
    },

    updateChart(canvasId, title, labels, data, color) {
        if (Dashboard.charts[canvasId]) {
            const chart = Dashboard.charts[canvasId];
            chart.data.labels = labels;
            chart.data.datasets[0].data = data;
            chart.update('none');
            return;
        }

        const ctx = document.getElementById(canvasId);
        if (!ctx) return;

        Dashboard.charts[canvasId] = new Chart(ctx, {
            type: 'line',
            data: {
                labels,
                datasets: [{
                    label: title,
                    data,
                    borderColor: color,
                    backgroundColor: color + '22',
                    borderWidth: 1.5,
                    pointRadius: 0,
                    fill: true,
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: { display: true, text: title, color: '#e0e0e0' },
                    legend: { display: false }
                },
                scales: {
                    x: {
                        ticks: { color: '#888', maxTicksLimit: 10 },
                        grid: { color: '#222' }
                    },
                    y: {
                        ticks: { color: '#888' },
                        grid: { color: '#222' }
                    }
                }
            }
        });
    },

    parseCSV(text) {
        const lines = text.trim().split('\n');
        if (lines.length < 2) return [];
        const headers = lines[0].split(',').map(h => h.trim());
        const rows = [];
        for (let i = 1; i < lines.length; i++) {
            const vals = lines[i].split(',').map(v => v.trim());
            if (vals.length !== headers.length) continue;
            const row = {};
            for (let j = 0; j < headers.length; j++) row[headers[j]] = vals[j];
            rows.push(row);
        }
        return rows;
    },

    startAutoRefresh() {
        Dashboard.stopAutoRefresh();
        Dashboard.autoRefreshTimer = setInterval(() => Dashboard.load(), 5000);
    },

    stopAutoRefresh() {
        if (Dashboard.autoRefreshTimer) {
            clearInterval(Dashboard.autoRefreshTimer);
            Dashboard.autoRefreshTimer = null;
        }
    }
};
