"""
Parte 1, itens 2-3: associação ingênua por IoU em todas as 7 sequências
do MOT17 (fonte: SDP, decidido empiricamente -- ver compare_detectors.py),
avaliada com as métricas implementadas na Parte 0.

Regra de associação (documentada, como o enunciado exige):
  - IoU entre detecções do quadro t e tracks vivas do quadro t-1
  - matching guloso por IoU decrescente, limiar fixo = 0.3
  - ID novo quando nenhuma detecção casa com nenhuma track viva
  - track morta após max_age=30 quadros consecutivos sem observação
    (= 1 segundo a 30fps -- tolera oclusões breves sem explodir o
    número de tracks fantasmas vivas indefinidamente)
"""

from pathlib import Path

import numpy as np

from src.datasets.mot17 import MOT17Sequence, SEQUENCE_IDS
from src.tracking.naive_tracker import NaiveTracker
from src.metrics.tracking_metrics import compute_idf1, count_id_switches, count_fragmentations

IOU_THRESHOLD = 0.3
MAX_AGE = 30
DETECTOR = "SDP"


def left_top_wh_to_yxyx(box):
    left, top, w, h = box
    return (top, left, top + h, left + w)


def run_sequence(seq_path):
    seq = MOT17Sequence(seq_path, load_gt=True)
    if seq.gt is None:
        return None

    detections_by_frame = []
    for t in range(1, seq.info.seq_length + 1):
        dets = seq.detections_at(t)  # left, top, w, h
        dets_yxyx = np.array([left_top_wh_to_yxyx(d) for d in dets]) if len(dets) else np.zeros((0, 4))
        detections_by_frame.append(dets_yxyx)

    tracker = NaiveTracker(iou_threshold=IOU_THRESHOLD, max_age=MAX_AGE)
    pred_tracks = tracker.run(detections_by_frame)
    # pred_tracks está indexado por frame 0-based; realinha pra 1-based como o GT
    pred_tracks = {
        tid: {f + 1: box for f, box in frames.items()}
        for tid, frames in pred_tracks.items()
    }

    gt_ids = np.unique(seq.gt[:, 1]).astype(int)
    gt_tracks = {}
    for gid in gt_ids:
        rows = seq.gt[seq.gt[:, 1] == gid]
        gt_tracks[gid] = {
            int(r[0]): left_top_wh_to_yxyx((r[2], r[3], r[4], r[5]))
            for r in rows
        }

    idf1, tp, fp, fn, mapping = compute_idf1(pred_tracks, gt_tracks)
    switches = count_id_switches(pred_tracks, gt_tracks)
    frags = count_fragmentations(pred_tracks, gt_tracks)

    n_gt_ids = len(gt_tracks)
    n_pred_ids = len(pred_tracks)
    count_error = abs(n_pred_ids - n_gt_ids)
    count_ratio = n_pred_ids / n_gt_ids if n_gt_ids > 0 else float("nan")

    return {
        "seq": seq.info.name, "idf1": idf1, "switches": switches,
        "fragmentations": frags, "n_gt_ids": n_gt_ids, "n_pred_ids": n_pred_ids,
        "count_error": count_error, "count_ratio": count_ratio,
    }


def main():
    root = Path("data/raw/MOT17/train")
    results = []

    for seq_id in SEQUENCE_IDS:
        seq_path = root / f"MOT17-{seq_id}-{DETECTOR}"
        print(f"Rodando MOT17-{seq_id}-{DETECTOR}...")
        r = run_sequence(seq_path)
        if r is None:
            continue
        results.append(r)
        print(f"  IDF1={r['idf1']:.3f}  switches={r['switches']}  frags={r['fragmentations']}  "
              f"IDs: {r['n_gt_ids']} real -> {r['n_pred_ids']} previsto "
              f"(razão={r['count_ratio']:.2f})")

    print(f"\n{'='*60}")
    print(f"Resumo (SDP, iou_threshold={IOU_THRESHOLD}, max_age={MAX_AGE})")
    print(f"{'='*60}")
    print(f"IDF1 médio: {np.mean([r['idf1'] for r in results]):.3f}")
    print(f"Switches totais: {sum(r['switches'] for r in results)}")
    print(f"Fragmentações totais: {sum(r['fragmentations'] for r in results)}")
    print(f"Razão de contagem média: {np.mean([r['count_ratio'] for r in results]):.3f}")


if __name__ == "__main__":
    main()