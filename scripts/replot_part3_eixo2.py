# scripts/replot_part3_eixo2.py
import json
import numpy as np
import matplotlib.pyplot as plt
from src.viz.plot_style import apply_style

apply_style()

with open("outputs/part3_eixo2_tracking_results.json") as f:
    data = json.load(f)

per_seq = data["per_sequence"]  # lista de dicts: regime, seed, seq_id, idf1, switches
REGIMES = ["teacher_forcing", "scheduled_sampling", "free_running"]
colors = {"teacher_forcing": "#128C94", "scheduled_sampling": "#E4572E", "free_running": "#2E8B57"}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

rng = np.random.default_rng(0)  # para jitter horizontal, só estética

for i, regime in enumerate(REGIMES):
    rows = [r for r in per_seq if r["regime"] == regime]
    idf1_vals = [r["idf1"] for r in rows]
    switch_vals = [r["switches"] for r in rows]

    jitter = rng.uniform(-0.12, 0.12, size=len(rows))
    axes[0].scatter(i + jitter, idf1_vals, color=colors[regime], s=35, alpha=0.6, zorder=3)
    axes[0].scatter([i], [np.mean(idf1_vals)], color="black", marker="_", s=400, zorder=4)

    axes[1].scatter(i + jitter, switch_vals, color=colors[regime], s=35, alpha=0.6, zorder=3)
    axes[1].scatter([i], [np.mean(switch_vals)], color="black", marker="_", s=400, zorder=4)

axes[0].set_xticks(range(len(REGIMES)))
axes[0].set_xticklabels(REGIMES, rotation=15)
axes[0].set_ylabel("IDF1 por sequência (21 = 3 seeds × 7 sequências)")
axes[0].set_title("IDF1 por regime (cada ponto = 1 sequência, 1 seed)")
axes[0].set_xlim(-0.5, len(REGIMES) - 0.5)

axes[1].set_xticks(range(len(REGIMES)))
axes[1].set_xticklabels(REGIMES, rotation=15)
axes[1].set_ylabel("Switches por sequência")
axes[1].set_title("ID switches por regime")
axes[1].set_xlim(-0.5, len(REGIMES) - 0.5)

plt.tight_layout()
plt.savefig("outputs/part3_eixo2_tracking_comparison.png", dpi=150)
print("Figura salva em outputs/part3_eixo2_tracking_comparison.png")