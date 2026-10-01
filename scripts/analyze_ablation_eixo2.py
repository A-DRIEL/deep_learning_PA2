# scripts/analyze_ablation_eixo2.py
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from src.viz.plot_style import apply_style
apply_style()

Path("outputs").mkdir(exist_ok=True)

with open("outputs/ablation_eixo2_results.json") as f:
    results = json.load(f)

REGIMES = ["teacher_forcing", "scheduled_sampling", "free_running"]
COLORS = {"teacher_forcing": "#128C94", "scheduled_sampling": "#E4572E", "free_running": "#2E8B57"}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for regime in REGIMES:
    runs = results[regime]
    max_len = max(len(r["history"]["val_loss"]) for r in runs)

    val_losses = np.full((len(runs), max_len), np.nan)
    grad_norms = np.full((len(runs), max_len), np.nan)
    for i, r in enumerate(runs):
        vl = r["history"]["val_loss"]
        gn = r["history"]["grad_norm"]
        val_losses[i, :len(vl)] = vl
        grad_norms[i, :len(gn)] = gn

    mean_val = np.nanmean(val_losses, axis=0)
    std_val = np.nanstd(val_losses, axis=0)
    mean_grad = np.nanmean(grad_norms, axis=0)
    std_grad = np.nanstd(grad_norms, axis=0)

    epochs = np.arange(max_len)
    axes[0].plot(epochs, mean_val, label=regime, color=COLORS[regime])
    axes[0].fill_between(epochs, mean_val - std_val, mean_val + std_val, alpha=0.2, color=COLORS[regime])

    axes[1].plot(epochs, mean_grad, label=regime, color=COLORS[regime])
    axes[1].fill_between(epochs, mean_grad - std_grad, mean_grad + std_grad, alpha=0.2, color=COLORS[regime])

axes[0].set_xlabel("Época")
axes[0].set_ylabel("Val loss")
axes[0].set_title("Val loss por regime (média ± desvio, 3 seeds)")
axes[0].legend()
axes[0].set_yscale("log")

axes[1].set_xlabel("Época")
axes[1].set_ylabel("Norma do gradiente")
axes[1].set_title("Norma do gradiente por regime")
axes[1].legend()

plt.tight_layout()
plt.savefig("outputs/ablation_eixo2_curves.png", dpi=150)
print("Figura salva em outputs/ablation_eixo2_curves.png")

# --- comparação com/sem clipping ---
tf_clip = results["teacher_forcing"]
tf_noclip = results["teacher_forcing_no_clip"]

fig2, ax = plt.subplots(figsize=(7, 5))
for r in tf_clip:
    ax.plot(r["history"]["grad_norm"], color="#128C94", alpha=0.6,
            label="Com clipping" if r["seed"] == 0 else None)
for r in tf_noclip:
    ax.plot(r["history"]["grad_norm"], color="#E4572E", alpha=0.6,
            label="Sem clipping" if r["seed"] == 0 else None)
ax.set_xlabel("Época")
ax.set_ylabel("Norma do gradiente")
ax.set_title("Efeito do gradient clipping (teacher forcing)")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/ablation_clipping_comparison.png", dpi=150)
print("Figura salva em outputs/ablation_clipping_comparison.png")

# --- print resumo numérico ---
for regime in REGIMES + ["teacher_forcing_no_clip"]:
    finals = [r["history"]["val_loss"][-1] for r in results[regime]]
    grad_finals = [r["history"]["grad_norm"][-1] for r in results[regime]]
    grad_max = [max(r["history"]["grad_norm"]) for r in results[regime]]
    print(f"{regime}: val_loss final={np.mean(finals):.6f}±{np.std(finals):.6f}, "
          f"grad_norm final={np.mean(grad_finals):.4f}, grad_norm MÁXIMO visto={max(grad_max):.4f}")