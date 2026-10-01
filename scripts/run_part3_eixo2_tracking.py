"""
Parte 3, Eixo 2 -- avaliação no tracker completo: pega os 9 checkpoints
(3 regimes x 3 seeds) já treinados e mede IDF1/switches nas 7 sequências
do MOT17, para ver se scheduled_sampling ou free_running reduzem o
"travamento" (lock-in) identificado na Parte 2.
"""

import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.datasets.mot17 import MOT17Sequence, SEQUENCE_IDS
from src.tracking.motion_lstm_tracker import MotionLSTMTracker
from src.models.motion_lstm import MotionLSTM
from src.metrics.tracking_metrics import compute_idf1, count_id_switches
from src.viz.plot_style import apply_style

apply_style()
Path("outputs").mkdir(exist_ok=True)

DETECTOR = "SDP"
REGIMES = ["teacher_forcing", "scheduled_sampling", "free_running"]
SEEDS = [0, 1, 2]


def build_gt_tracks(seq):
    gt_ids = np.unique(seq.gt[:, 1]).astype(int)
    return {
        gid: {int(r[0]): (r[3], r[2], r[3] + r[5], r[2] + r[4])
              for r in seq.gt[seq.gt[:, 1] == gid]}
        for gid in gt_ids
    }


def run_tracker(seq, model, device):
    detections_pixel = [seq.detections_at(t) for t in range(1, seq.info.seq_length + 1)]
    tracker = MotionLSTMTracker(model, device, seq.info.im_width, seq.info.im_height,
                                  iou_threshold=0.3, max_age=30)
    pred = tracker.run(detections_pixel)
    return {tid: {f + 1: box for f, box in frames.items()} for tid, frames in pred.items()}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path("data/raw/MOT17/train")

    # carrega GT uma vez só por sequência (reaproveitado pelos 9 modelos)
    sequences, gt_tracks_by_seq = {}, {}
    for seq_id in SEQUENCE_IDS:
        seq = MOT17Sequence(root / f"MOT17-{seq_id}-{DETECTOR}", load_gt=True)
        if seq.gt is not None:
            sequences[seq_id] = seq
            gt_tracks_by_seq[seq_id] = build_gt_tracks(seq)

    results = {regime: {"idf1": [], "switches": []} for regime in REGIMES}
    per_seq_results = []

    for regime in REGIMES:
        for seed in SEEDS:
            ckpt_path = f"outputs/ablation_checkpoints/{regime}_seed{seed}.pt"
            model = MotionLSTM(input_dim=4, hidden_dim=64)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))

            idf1_per_seq, switches_per_seq = [], []
            for seq_id in SEQUENCE_IDS:
                if seq_id not in sequences:
                    continue
                seq = sequences[seq_id]
                gt_tracks = gt_tracks_by_seq[seq_id]

                pred = run_tracker(seq, model, device)
                idf1, *_ = compute_idf1(pred, gt_tracks)
                switches = count_id_switches(pred, gt_tracks)

                idf1_per_seq.append(idf1)
                switches_per_seq.append(switches)
                per_seq_results.append({
                    "regime": regime, "seed": seed, "seq_id": seq_id,
                    "idf1": idf1, "switches": switches,
                })

            mean_idf1 = float(np.mean(idf1_per_seq))
            total_switches = int(sum(switches_per_seq))
            print(f"{regime}, seed={seed}: IDF1 médio={mean_idf1:.3f}, switches totais={total_switches}")

            results[regime]["idf1"].append(mean_idf1)
            results[regime]["switches"].append(total_switches)

    print(f"\n{'='*60}")
    for regime in REGIMES:
        idf1_vals = results[regime]["idf1"]
        switch_vals = results[regime]["switches"]
        print(f"{regime}: IDF1={np.mean(idf1_vals):.3f}±{np.std(idf1_vals):.3f}, "
              f"switches={np.mean(switch_vals):.1f}±{np.std(switch_vals):.1f}")

    with open("outputs/part3_eixo2_tracking_results.json", "w") as f:
        json.dump({"summary": results, "per_sequence": per_seq_results}, f, indent=2)
    print("\nResultados salvos em outputs/part3_eixo2_tracking_results.json")

    # --- figura: IDF1 e switches por regime, com barras de erro entre seeds ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(REGIMES))

    idf1_means = [np.mean(results[r]["idf1"]) for r in REGIMES]
    idf1_stds = [np.std(results[r]["idf1"]) for r in REGIMES]
    axes[0].bar(x, idf1_means, yerr=idf1_stds, color=["#128C94", "#E4572E", "#2E8B57"], capsize=5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(REGIMES, rotation=15)
    axes[0].set_ylabel("IDF1 médio (7 sequências)")
    axes[0].set_title("IDF1 por regime de treino")

    switch_means = [np.mean(results[r]["switches"]) for r in REGIMES]
    switch_stds = [np.std(results[r]["switches"]) for r in REGIMES]
    axes[1].bar(x, switch_means, yerr=switch_stds, color=["#128C94", "#E4572E", "#2E8B57"], capsize=5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(REGIMES, rotation=15)
    axes[1].set_ylabel("Switches totais (7 sequências)")
    axes[1].set_title("ID switches por regime de treino")

    plt.tight_layout()
    plt.savefig("outputs/part3_eixo2_tracking_comparison.png", dpi=150)
    print("Figura salva em outputs/part3_eixo2_tracking_comparison.png")


if __name__ == "__main__":
    main()