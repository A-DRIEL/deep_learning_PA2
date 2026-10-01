# scripts/compare_lockin_across_regimes.py
"""
Fecha o ciclo Parte 2 -> Parte 3: mede quantos quadros CONSECUTIVOS o
pred_id se desvia do ID dominante numa janela ao redor do evento --
métrica direta, sem inferir por adjacência entre switches (que pode
confundir "sair do ID certo" com "voltar pro ID certo", são switches
diferentes e próximos um do outro).
"""

from collections import Counter
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from src.datasets.mot17 import MOT17Sequence
from src.tracking.motion_lstm_tracker import MotionLSTMTracker
from src.models.motion_lstm import MotionLSTM
from src.analysis.find_critical_moments import assigned_pred_id_per_frame, count_switch_events

Path("outputs").mkdir(exist_ok=True)

DETECTOR = "SDP"
REGIMES = ["teacher_forcing", "scheduled_sampling", "free_running"]
TARGET_GT_ID = 3
WINDOW_CENTER = 236
ANALYSIS_WINDOW = 30  # quadros pra cada lado do evento, para achar o ID dominante


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


def longest_deviation_from_dominant(assignment, center_frame, half_window=30):
    """
    Dentro de [center_frame - half_window, center_frame + half_window]:
    acha o ID mais comum (dominante), depois mede a MAIOR sequência de
    quadros consecutivos onde o ID observado é DIFERENTE do dominante.
    Devolve (dominant_id, max_deviation_length, deviation_is_at_end).
    """
    frames = sorted(f for f in assignment.keys() if center_frame - half_window <= f <= center_frame + half_window)
    ids_in_window = [assignment[f] for f in frames if assignment[f] is not None]
    if not ids_in_window:
        return None, 0, False

    dominant_id = Counter(ids_in_window).most_common(1)[0][0]

    max_len, cur_len = 0, 0
    ends_at_last_frame = False
    for i, f in enumerate(frames):
        if assignment[f] is not None and assignment[f] != dominant_id:
            cur_len += 1
            max_len = max(max_len, cur_len)
            ends_at_last_frame = (i == len(frames) - 1)
        else:
            cur_len = 0

    return dominant_id, max_len, ends_at_last_frame


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path("data/raw/MOT17/train")

    seq = MOT17Sequence(root / f"MOT17-04-{DETECTOR}", load_gt=True)
    gt_tracks = build_gt_tracks(seq)
    gt_track = gt_tracks[TARGET_GT_ID]

    print(f"Comparando regimes no mesmo caso: MOT17-04, gt_id={TARGET_GT_ID}\n")

    results = {}
    for regime in REGIMES:
        model = MotionLSTM(input_dim=4, hidden_dim=64)
        model.load_state_dict(torch.load(f"outputs/ablation_checkpoints/{regime}_seed0.pt", map_location=device))

        pred = run_tracker(seq, model, device)
        assignment = assigned_pred_id_per_frame(pred, gt_track)
        n_switches = len(count_switch_events(assignment))

        dominant_id, max_dev, still_deviating = longest_deviation_from_dominant(
            assignment, WINDOW_CENTER, half_window=ANALYSIS_WINDOW
        )

        results[regime] = {"assignment": assignment, "switches": n_switches}

        status = " (ainda desviado no fim da janela analisada)" if still_deviating else " (recuperou dentro da janela)"
        print(f"{regime}: {n_switches} switches totais | ID dominante={dominant_id} | "
              f"maior desvio consecutivo = {max_dev} quadro(s){status}")

    # --- figura (mesma de antes) ---
    window = 6
    frames_to_show = list(range(max(1, WINDOW_CENTER - window), WINDOW_CENTER + window + 1, 2))
    n = len(frames_to_show)

    fig, axes = plt.subplots(len(REGIMES), n, figsize=(3 * n, 3 * len(REGIMES)))
    colors_by_regime = {"teacher_forcing": "tab:blue", "scheduled_sampling": "tab:orange", "free_running": "tab:green"}

    for row, regime in enumerate(REGIMES):
        assignment = results[regime]["assignment"]
        for col, f in enumerate(frames_to_show):
            img_path = root / f"MOT17-04-{DETECTOR}" / "img1" / f"{f:06d}.jpg"
            ax = axes[row, col]
            if not img_path.exists():
                ax.axis("off")
                continue
            img = Image.open(img_path)
            ax.imshow(img)
            if f in gt_track:
                y0, x0, y1, x1 = gt_track[f]
                pid = assignment.get(f)
                rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                      edgecolor=colors_by_regime[regime], linewidth=3)
                ax.add_patch(rect)
                ax.set_title(f"t={f}  ID={pid}", fontsize=9)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(regime, fontsize=11)

    plt.tight_layout()
    plt.savefig("outputs/part3_lockin_comparison.png", dpi=130)
    print("\nFigura salva em outputs/part3_lockin_comparison.png")


if __name__ == "__main__":
    main()