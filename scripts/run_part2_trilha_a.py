# scripts/run_part2_trilha_a.py
from pathlib import Path

import numpy as np
import torch

from src.datasets.mot17 import MOT17Sequence, SEQUENCE_IDS
from src.tracking.naive_tracker import NaiveTracker
from src.tracking.motion_lstm_tracker import MotionLSTMTracker
from src.models.motion_lstm import MotionLSTM
from src.metrics.tracking_metrics import compute_idf1, count_id_switches, count_fragmentations

DETECTOR = "SDP"
IOU_THRESHOLD = 0.3
MAX_AGE = 30


def left_top_wh_to_yxyx(box):
    left, top, w, h = box
    return (top, left, top + h, left + w)


def build_gt_tracks(seq):
    gt_ids = np.unique(seq.gt[:, 1]).astype(int)
    return {
        gid: {int(r[0]): left_top_wh_to_yxyx((r[2], r[3], r[4], r[5]))
              for r in seq.gt[seq.gt[:, 1] == gid]}
        for gid in gt_ids
    }


def run_naive(seq):
    detections_by_frame = []
    for t in range(1, seq.info.seq_length + 1):
        dets = seq.detections_at(t)
        dets_yxyx = np.array([left_top_wh_to_yxyx(d) for d in dets]) if len(dets) else np.zeros((0, 4))
        detections_by_frame.append(dets_yxyx)

    tracker = NaiveTracker(iou_threshold=IOU_THRESHOLD, max_age=MAX_AGE)
    pred = tracker.run(detections_by_frame)
    return {tid: {f + 1: box for f, box in frames.items()} for tid, frames in pred.items()}


def run_trilha_a(seq, model, device):
    detections_by_frame = [seq.detections_at(t) for t in range(1, seq.info.seq_length + 1)]

    tracker = MotionLSTMTracker(model, device, seq.info.im_width, seq.info.im_height,
                                  iou_threshold=IOU_THRESHOLD, max_age=MAX_AGE)
    pred = tracker.run(detections_by_frame)
    return {tid: {f + 1: box for f, box in frames.items()} for tid, frames in pred.items()}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MotionLSTM(input_dim=4, hidden_dim=64)
    model.load_state_dict(torch.load("outputs/motion_lstm.pt", map_location=device))

    root = Path("data/raw/MOT17/train")
    results = []

    for seq_id in SEQUENCE_IDS:
        seq = MOT17Sequence(root / f"MOT17-{seq_id}-{DETECTOR}", load_gt=True)
        if seq.gt is None:
            continue

        gt_tracks = build_gt_tracks(seq)

        pred_naive = run_naive(seq)
        idf1_naive, *_ = compute_idf1(pred_naive, gt_tracks)
        switches_naive = count_id_switches(pred_naive, gt_tracks)
        frags_naive = count_fragmentations(pred_naive, gt_tracks)

        pred_a = run_trilha_a(seq, model, device)
        idf1_a, *_ = compute_idf1(pred_a, gt_tracks)
        switches_a = count_id_switches(pred_a, gt_tracks)
        frags_a = count_fragmentations(pred_a, gt_tracks)

        print(f"MOT17-{seq_id}: baseline IDF1={idf1_naive:.3f} switches={switches_naive} frags={frags_naive} | "
              f"Trilha A IDF1={idf1_a:.3f} switches={switches_a} frags={frags_a}")
        results.append((seq_id, idf1_naive, idf1_a, switches_naive, switches_a, frags_naive, frags_a))

    print(f"\nIDF1 médio -- baseline: {np.mean([r[1] for r in results]):.3f}, "
          f"Trilha A: {np.mean([r[2] for r in results]):.3f}")
    print(f"Switches totais -- baseline: {sum(r[3] for r in results)}, "
          f"Trilha A: {sum(r[4] for r in results)}")
    print(f"Fragmentações totais -- baseline: {sum(r[5] for r in results)}, "
          f"Trilha A: {sum(r[6] for r in results)}")


import json

def save_results(results, path="outputs/part2_results.json"):
    data = [
        {
            "seq_id": r[0], "idf1_baseline": r[1], "idf1_trilha_a": r[2],
            "switches_baseline": r[3], "switches_trilha_a": r[4],
            "frags_baseline": r[5], "frags_trilha_a": r[6],
        }
        for r in results
    ]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResultados salvos em {path}")

if __name__ == "__main__":
    main()