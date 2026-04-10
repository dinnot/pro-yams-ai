// Run comparison view for Pro Yams UI.

const Comparison = {
    charts: {},

    init() {
        document.getElementById('btn-load-comparison').addEventListener('click', () => Comparison.load());
    },

    async load() {
        const text = document.getElementById('comp-dirs').value.trim();
        if (!text) return;

        const dirs = text.split('\n').map(d => d.trim()).filter(d => d.length > 0);
        if (dirs.length === 0) return;

        const colors = ['#e94560', '#4caf50', '#64b5f6', '#ffc107', '#ab47bc', '#ff7043'];

        // Load training logs for all dirs.
        const lossDatasets = [];
        const wrDatasets = [];

        for (let i = 0; i < dirs.length; i++) {
            const dir = dirs[i];
            const color = colors[i % colors.length];
            const label = dir.split('/').pop() || dir;

            // Training loss.
            const trainCsv = await API.getTrainingLog(dir);
            if (trainCsv) {
                const rows = Dashboard.parseCSV(trainCsv);
                const headers = rows.length > 0 ? Object.keys(rows[0]) : [];
                const stepCol = headers.find(h => /step/i.test(h));
                const lossCol = headers.find(h => /loss/i.test(h));
                if (stepCol && lossCol && rows.length > 0) {
                    lossDatasets.push({
                        label,
                        data: rows.map(r => ({ x: parseFloat(r[stepCol]), y: parseFloat(r[lossCol]) })),
                        borderColor: color,
                        borderWidth: 1.5,
                        pointRadius: 0,
                        fill: false,
                        tension: 0.1
                    });
                }
            }

            // Eval win rate.
            const evalCsv = await API.getEvalLog(dir);
            if (evalCsv) {
                const rows = Dashboard.parseCSV(evalCsv);
                const headers = rows.length > 0 ? Object.keys(rows[0]) : [];
                const stepCol = headers.find(h => /step/i.test(h));
                const wrCol = headers.find(h => /win.*rate/i.test(h) || /wr/i.test(h));
                if (stepCol && wrCol && rows.length > 0) {
                    wrDatasets.push({
                        label,
                        data: rows.map(r => ({ x: parseFloat(r[stepCol]), y: parseFloat(r[wrCol]) })),
                        borderColor: color,
                        borderWidth: 1.5,
                        pointRadius: 0,
                        fill: false,
                        tension: 0.1
                    });
                }
            }
        }

        Comparison.renderOverlay('chart-comparison-loss', 'Training Loss Comparison', lossDatasets);
        Comparison.renderOverlay('chart-comparison-wr', 'Win Rate Comparison', wrDatasets);
    },

    renderOverlay(canvasId, title, datasets) {
        if (Comparison.charts[canvasId]) {
            Comparison.charts[canvasId].destroy();
        }

        const ctx = document.getElementById(canvasId);
        if (!ctx) return;
        if (datasets.length === 0) return;

        Comparison.charts[canvasId] = new Chart(ctx, {
            type: 'line',
            data: { datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                parsing: false,
                plugins: {
                    title: { display: true, text: title, color: '#e0e0e0' },
                    legend: {
                        display: true,
                        labels: { color: '#ccc' }
                    }
                },
                scales: {
                    x: {
                        type: 'linear',
                        title: { display: true, text: 'Step', color: '#888' },
                        ticks: { color: '#888' },
                        grid: { color: '#222' }
                    },
                    y: {
                        ticks: { color: '#888' },
                        grid: { color: '#222' }
                    }
                }
            }
        });
    }
};
