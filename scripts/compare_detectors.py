"""
Compara os 3 detectores públicos do MOT17 (DPM, FRCNN, SDP) nas 7
sequências de treino, pra decidir qual usar como fonte padrão do PA.

Métrica simples, sem nada de tracking ainda: recall (quantas caixas
reais cada detector encontra, a IoU>=0.5) e precisão (quantas detecções
batem com algo real vs. são ruído).
"""

from pathlib import Path

import numpy as np

from src.datasets.mot17 import MOT17Sequence, SEQUENCE_IDS
from src.metrics.tracking_metrics import compute_iou


def evaluate_detector_on_sequence(seq_path, iou_threshold=0.5):
    seq = MOT17Sequence(seq_path, load_gt=True)
    if seq.gt is None:
        return None

    total_gt, total_det = 0, 0
    matched_gt, matched_det = 0, 0

    for t in range(1, seq.info.seq_length + 1):
        gt_boxes = seq.gt_at(t)  # [id, left, top, w, h]
        det_boxes = seq.detections_at(t)  # [left, top, w, h]

        total_gt += len(gt_boxes)
        total_det += len(det_boxes)

        if len(gt_boxes) == 0 or len(det_boxes) == 0:
            continue

        # converte left,top,w,h -> y0,x0,y1,x1 para usar compute_iou
        gt_yxyx = [(b[2], b[1], b[2] + b[4], b[1] + b[3]) for b in gt_boxes]
        det_yxyx = [(b[1], b[0], b[1] + b[3], b[0] + b[2]) for b in det_boxes]

        gt_matched_this_frame = set()
        det_matched_this_frame = set()
        for i, gb in enumerate(gt_yxyx):
            for j, db in enumerate(det_yxyx):
                if j in det_matched_this_frame:
                    continue
                if compute_iou(gb, db) >= iou_threshold:
                    gt_matched_this_frame.add(i)
                    det_matched_this_frame.add(j)
                    break

        matched_gt += len(gt_matched_this_frame)
        matched_det += len(det_matched_this_frame)

    recall = matched_gt / total_gt if total_gt > 0 else 0.0
    precision = matched_det / total_det if total_det > 0 else 0.0
    return {"recall": recall, "precision": precision,
            "total_gt": total_gt, "total_det": total_det}


def main():
    root = Path("data/raw/MOT17/train")

    for detector in ["DPM", "FRCNN", "SDP"]:
        print(f"\n=== Detector: {detector} ===")
        recalls, precisions = [], []
        for seq_id in SEQUENCE_IDS:
            seq_path = root / f"MOT17-{seq_id}-{detector}"
            result = evaluate_detector_on_sequence(seq_path)
            if result is None:
                continue
            recalls.append(result["recall"])
            precisions.append(result["precision"])
            print(f"  MOT17-{seq_id}: recall={result['recall']:.3f}, "
                  f"precision={result['precision']:.3f}, "
                  f"gt={result['total_gt']}, det={result['total_det']}")

        print(f"  MÉDIA: recall={np.mean(recalls):.3f}, precision={np.mean(precisions):.3f}")


if __name__ == "__main__":
    main()