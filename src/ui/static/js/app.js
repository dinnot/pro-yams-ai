// Main app logic — tab switching and initialization.

document.addEventListener('DOMContentLoaded', () => {
    // Tab switching.
    const tabs = document.querySelectorAll('.tab-btn');
    const sections = document.querySelectorAll('.tab-section');

    tabs.forEach(btn => {
        btn.addEventListener('click', () => {
            const target = btn.dataset.tab;

            tabs.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            sections.forEach(s => {
                s.classList.toggle('active', s.id === `tab-${target}`);
            });
        });
    });

    // Initialize modules.
    Game.init();
    Tensor.init();
    Dashboard.init();
    Comparison.init();
});
