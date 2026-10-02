# scripts/analyze_part5.py
"""
Lê outputs/part5_detector_stress.json e produz:
  - outputs/part5_main.png        : mAP e IDF1 juntos, razão de contagem, switches/ID
  - outputs/part5_per_sequence.png: queda de IDF1 por sequência (ordenadas por
                                    densidade) + fator de amplificação
  - outputs/part5_table.md        : tabela com média ± desvio (entre seeds)

Pergunta respondida: o modelo temporal ABSORVE ou AMPLIFICA a falha do detector?

Fator de amplificação (por tracker, por intensidade):
    A = queda relativa de IDF1 / queda relativa de mAP
        (relativas à configuração "limpo")
    A < 1: o tracker perde menos identidade do que o detector perdeu em
           qualidade (absorve);  A > 1: perde mais (amplifica).
Barras de erro = desvio entre seeds da degradação (o modelo é fixo);
cada ponto de seed é a média não ponderada sobre as sequências.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.datasets.detector_degradation import INTENSITIES
from src.viz.plot_style import PALETTE, apply_style

apply_style()

ORDER = ["limpo", "leve", "media", "pesada"]
TRACKERS = ["baseline", "trilha_a"]
TRACKER_LABEL = {"baseline": "Baseline (IoU)", "trilha_a": "Trilha A (LSTM)"}
TRACKER_COLOR = {"baseline": PALETTE["coral"], "trilha_a": PALETTE["teal"]}
DENSITY = {"02": 30.97, "04": 45.29, "05": 8.27, "09": 10.14,
           "10": 19.63, "11": 10.48, "13": 15.52}  # boxes/quadro (Parte 1)


def per_seed_values(rows, config, tracker, metric, seq_ids):
    """Média sobre sequências, uma por seed -> array (n_seeds,)."""
    by_seed = defaultdict(list)
    for r in rows:
        if r["config"] == config and r["tracker"] == tracker and r["seq_id"] in seq_ids:
            by_seed[r["seed"]].append(r[metric])
    return np.array([np.mean(v) for v in by_seed.values()])


def mean_std(rows, config, tracker, metric, seq_ids):
    v = per_seed_values(rows, config, tracker, metric, seq_ids)
    return float(v.mean()), float(v.std())


def amplification(idf1_clean, idf1_deg, map_clean, map_deg):
    drop_map = (map_clean - map_deg) / map_clean
    if drop_map <= 1e-9:
        return float("nan")
    return ((idf1_clean - idf1_deg) / idf1_clean) / drop_map


def build_table(rows, seq_ids, configs):
    lines = ["| config | mAP | IDF1 base | IDF1 A | razão IDs base | razão IDs A "
             "| sw/ID base | sw/ID A | amplif. base | amplif. A |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    clean = {t: mean_std(rows, "limpo", t, "idf1", seq_ids)[0] for t in TRACKERS}
    map_clean = mean_std(rows, "limpo", "baseline", "map", seq_ids)[0]
    for c in configs:
        m, ms = mean_std(rows, c, "baseline", "map", seq_ids)
        cells = [c, f"{m:.3f}±{ms:.3f}"]
        for metric in ["idf1", "count_ratio", "switches_per_id"]:
            for t in TRACKERS:
                mu, sd = mean_std(rows, c, t, metric, seq_ids)
                cells.append(f"{mu:.3f}±{sd:.3f}")
        for t in TRACKERS:
            idf1_c = mean_std(rows, c, t, "idf1", seq_ids)[0]
            cells.append("—" if c == "limpo" else f"{amplification(clean[t], idf1_c, map_clean, m):.2f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def label(c):
    cfg = INTENSITIES[c]
    if c == "limpo":
        return "limpo"
    return f"{c}\ndrop {cfg.drop_prob:.0%}\nruído {cfg.noise_rel:.2f}\nFP {cfg.fp_ratio:.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="outputs/part5_detector_stress.json")
    args = ap.parse_args()

    data = json.load(open(args.results))
    rows = data["rows"]
    all_seqs = sorted({r["seq_id"] for r in rows})
    heldout = sorted({r["seq_id"] for r in rows if r["heldout"]})
    x = np.arange(len(ORDER))

    # ---------------- Figura 1: mAP e IDF1 juntos + identidades ----------------
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    m = [mean_std(rows, c, "baseline", "map", all_seqs) for c in ORDER]
    axes[0].errorbar(x, [a for a, _ in m], yerr=[b for _, b in m], marker="s", ls="--",
                     color=PALETTE["muted"], label="mAP (detecção)")
    for t in TRACKERS:
        v = [mean_std(rows, c, t, "idf1", all_seqs) for c in ORDER]
        axes[0].errorbar(x, [a for a, _ in v], yerr=[b for _, b in v], marker="o",
                         color=TRACKER_COLOR[t], label=f"IDF1 — {TRACKER_LABEL[t]}")
    axes[0].set_title("Qualidade do detector vs. qualidade da identidade")
    axes[0].set_ylabel("Score (média das sequências)")
    axes[0].set_ylim(0, 1.05)

    for ax, metric, title, ylabel in [
        (axes[1], "count_ratio", "Identidades previstas / reais", "razão (1.0 = ideal)"),
        (axes[2], "switches_per_id", "ID switches por identidade real", "switches / ID"),
    ]:
        for t in TRACKERS:
            v = [mean_std(rows, c, t, metric, all_seqs) for c in ORDER]
            ax.errorbar(x, [a for a, _ in v], yerr=[b for _, b in v], marker="o",
                        color=TRACKER_COLOR[t], label=TRACKER_LABEL[t])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
    axes[1].axhline(1.0, color="gray", ls="--", lw=1)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([label(c) for c in ORDER], fontsize=8)
        ax.legend(fontsize=8)
    plt.suptitle(f"Parte 5 — degradação do detector ({len(all_seqs)} sequências, "
                 f"{len(data['meta']['seeds'])} seeds de degradação, modelo fixo)")
    plt.tight_layout()
    plt.savefig("outputs/part5_main.png", dpi=150)

    # ---------------- Figura 2: por sequência + amplificação ----------------
    seqs_by_density = sorted(all_seqs, key=lambda s: DENSITY.get(s, 0))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    xs = np.arange(len(seqs_by_density))
    for k, t in enumerate(TRACKERS):
        drops = []
        for s in seqs_by_density:
            clean = per_seed_values(rows, "limpo", t, "idf1", [s]).mean()
            heavy = per_seed_values(rows, "pesada", t, "idf1", [s]).mean()
            drops.append(clean - heavy)
        axes[0].bar(xs + (k - 0.5) * 0.38, drops, width=0.38,
                    color=TRACKER_COLOR[t], label=TRACKER_LABEL[t])
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels([f"{s}\n(dens={DENSITY.get(s, 0):.0f})" for s in seqs_by_density],
                            fontsize=8)
    axes[0].set_ylabel("Queda de IDF1 (limpo → pesada)")
    axes[0].set_title("Quem sofre mais, por sequência (ordenadas por densidade)")
    axes[0].legend()

    degraded = ORDER[1:]
    map_clean = mean_std(rows, "limpo", "baseline", "map", all_seqs)[0]
    for k, t in enumerate(TRACKERS):
        clean = mean_std(rows, "limpo", t, "idf1", all_seqs)[0]
        amps = [amplification(clean, mean_std(rows, c, t, "idf1", all_seqs)[0],
                              map_clean, mean_std(rows, c, "baseline", "map", all_seqs)[0])
                for c in degraded]
        axes[1].bar(np.arange(len(degraded)) + (k - 0.5) * 0.38, amps, width=0.38,
                    color=TRACKER_COLOR[t], label=TRACKER_LABEL[t])
    axes[1].axhline(1.0, color="gray", ls="--", lw=1)
    axes[1].set_xticks(np.arange(len(degraded)))
    axes[1].set_xticklabels(degraded)
    axes[1].set_ylabel("Fator de amplificação  (<1 absorve, >1 amplifica)")
    axes[1].set_title("O modelo temporal absorve ou amplifica?")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig("outputs/part5_per_sequence.png", dpi=150)

    # ---------------- Tabelas ----------------
    configs_present = [c for c in ORDER if any(r["config"] == c for r in rows)]
    md = ["## Todas as sequências\n", build_table(rows, all_seqs, configs_present)]
    if heldout:
        md += [f"\n\n## Apenas held-out ({', '.join(heldout)})\n",
               build_table(rows, heldout, configs_present)]
    isolated = [c for c in ["so_descarte", "so_ruido", "so_fp"]
                if any(r["config"] == c for r in rows)]
    if isolated:
        md += ["\n\n## Falhas isoladas (intensidade média)\n",
               build_table(rows, all_seqs, ["limpo"] + isolated)]
    Path("outputs/part5_table.md").write_text("\n".join(md))
    print("\n".join(md))
    print("\nFiguras: outputs/part5_main.png, outputs/part5_per_sequence.png")


if __name__ == "__main__":
    main()
