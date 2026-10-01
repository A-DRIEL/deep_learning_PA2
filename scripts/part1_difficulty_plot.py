"""
Parte 1, item 5: gráfico obrigatório do "descolamento" -- dois painéis,
sequências ordenadas por densidade (eixo de dificuldade escolhido,
justificado empiricamente: correlaciona fortemente com a razão de
contagem em 5 das 7 sequências -- MOT17-02 e MOT17-04 são exceções
documentadas, não escondidas).

Painel de cima: mAP por quadro (qualidade de detecção, independente de
identidade) e IDF1 (qualidade de identidade) -- mostra que uma coisa
não implica a outra.
Painel de baixo: razão de contagem (IDs previstos / IDs reais) e
switches por identidade real.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.datasets.mot17 import MOT17Sequence, SEQUENCE_IDS
from src.tracking.naive_tracker import NaiveTracker
from src.metrics.tracking_metrics import compute_idf1, count_id_switches, compute_iou

Path("outputs").mkdir(exist_ok=True)

DETECTOR = "SDP"
IOU_THRESHOLD = 0.3
MAX_AGE = 30

# densidade oficial (boxes/quadro), calculada a partir do próprio gt.txt
DENSITY = {
    "02": 30.97, "04": 45.29, "05": 8.27, "09": 10.14,
    "10": 19.63, "11": 10.48, "13": 15.52,
}


def left_top_wh_to_yxyx(box):
    left, top, w, h = box
    return (top, left, top + h, left + w)


def frame_map(pred_boxes, gt_boxes, thresholds=np.arange(0.5, 1.0, 0.05)):
    """mAP num único quadro: matching guloso por IoU, por limiar, média."""
    if len(gt_boxes) == 0:
        return 1.0 if len(pred_boxes) == 0 else 0.0
    if len(pred_boxes) == 0:
        return 0.0

    aps = []
    for t in thresholds:
        pairs = [
            (compute_iou(pb, gb), i, j)
            for i, pb in enumerate(pred_boxes) for j, gb in enumerate(gt_boxes)
            if compute_iou(pb, gb) >= t
        ]
        pairs.sort(key=lambda p: p[0], reverse=True)
        matched_p, matched_g = set(), set()
        tp = 0
        for iou, i, j in pairs:
            if i in matched_p or j in matched_g:
                continue
            matched_p.add(i)
            matched_g.add(j)
            tp += 1
        fp = len(pred_boxes) - tp
        fn = len(gt_boxes) - tp
        denom = tp + fp + fn
        aps.append(tp / denom if denom > 0 else 1.0)
    return float(np.mean(aps))


def run_sequence(seq_path):
    seq = MOT17Sequence(seq_path, load_gt=True)
    if seq.gt is None:
        return None

    detections_by_frame = []
    frame_maps = []
    for t in range(1, seq.info.seq_length + 1):
        dets = seq.detections_at(t)
        dets_yxyx = [left_top_wh_to_yxyx(d) for d in dets] if len(dets) else []
        detections_by_frame.append(np.array(dets_yxyx) if dets_yxyx else np.zeros((0, 4)))

        gt_rows = seq.gt[seq.gt[:, 0] == t]
        gt_yxyx = [left_top_wh_to_yxyx((r[2], r[3], r[4], r[5])) for r in gt_rows]
        frame_maps.append(frame_map(dets_yxyx, gt_yxyx))

    tracker = NaiveTracker(iou_threshold=IOU_THRESHOLD, max_age=MAX_AGE)
    pred_tracks = tracker.run(detections_by_frame)
    pred_tracks = {tid: {f + 1: box for f, box in frames.items()} for tid, frames in pred_tracks.items()}

    gt_ids = np.unique(seq.gt[:, 1]).astype(int)
    gt_tracks = {
        gid: {int(r[0]): left_top_wh_to_yxyx((r[2], r[3], r[4], r[5]))
              for r in seq.gt[seq.gt[:, 1] == gid]}
        for gid in gt_ids
    }

    idf1, tp, fp, fn, mapping = compute_idf1(pred_tracks, gt_tracks)
    switches = count_id_switches(pred_tracks, gt_tracks)
    n_gt_ids = len(gt_tracks)
    n_pred_ids = len(pred_tracks)

    return {
        "seq_id": seq.info.name.split("-")[1],
        "mean_frame_map": float(np.mean(frame_maps)),
        "idf1": idf1,
        "count_ratio": n_pred_ids / n_gt_ids if n_gt_ids else np.nan,
        "switches_per_id": switches / n_gt_ids if n_gt_ids else np.nan,
    }


def main():
    root = Path("data/raw/MOT17/train")
    results = []
    for seq_id in SEQUENCE_IDS:
        print(f"Processando MOT17-{seq_id}-{DETECTOR}...")
        r = run_sequence(root / f"MOT17-{seq_id}-{DETECTOR}")
        if r:
            results.append(r)

    # ordena pela densidade (eixo de dificuldade escolhido)
    results.sort(key=lambda r: DENSITY[r["seq_id"]])
    labels = [f"MOT17-{r['seq_id']}\n(dens={DENSITY[r['seq_id']]:.0f})" for r in results]

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    x = np.arange(len(results))
    axes[0].plot(x, [r["mean_frame_map"] for r in results], "o-", color="#128C94", label="mAP por quadro")
    axes[0].plot(x, [r["idf1"] for r in results], "o-", color="#E4572E", label="IDF1")
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()
    axes[0].set_title("Qualidade de detecção (mAP) vs. qualidade de identidade (IDF1)")
    axes[0].grid(alpha=0.3)

    ax2b = axes[1].twinx()
    axes[1].bar(x - 0.2, [r["count_ratio"] for r in results], width=0.4, color="#E4572E", label="Razão de contagem")
    ax2b.bar(x + 0.2, [r["switches_per_id"] for r in results], width=0.4, color="#5B6B70", label="Switches/ID")
    axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Razão de contagem (previsto/real)")
    ax2b.set_ylabel("Switches por identidade real")
    axes[1].set_title("Identidades fantasmas e trocas, por sequência (ordenado por densidade)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=8)

    lines1, labs1 = axes[1].get_legend_handles_labels()
    lines2, labs2 = ax2b.get_legend_handles_labels()
    axes[1].legend(lines1 + lines2, labs1 + labs2, loc="upper left")

    plt.tight_layout()
    plt.savefig("outputs/part1_difficulty_plot.png", dpi=150)
    print("\nFigura salva em outputs/part1_difficulty_plot.png")

    for r in results:
        print(r)


if __name__ == "__main__":
    main()