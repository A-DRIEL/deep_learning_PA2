"""
Funções utilitárias para rodar os dois trackers (baseline ingênuo e
Trilha A) a partir de listas de detecções por quadro, e para montar as
trajetórias GT. Consolida o que os scripts anteriores repetiam
(left_top_wh_to_yxyx, build_gt_tracks, realinhamento 0-based -> 1-based).

Entrada: lista (n_frames) de arrays (N_t, >=4) em (left, top, w, h, ...).
Saída:   dict id -> {frame (1-based): caixa (y0, x0, y1, x1)}.
"""

import numpy as np

from src.tracking.naive_tracker import NaiveTracker
from src.tracking.motion_lstm_tracker import MotionLSTMTracker

IOU_THRESHOLD = 0.3  # mesmos valores das Partes 1 e 2
MAX_AGE = 30


def _to_yxyx(d):
    d = np.asarray(d, dtype=np.float64)
    if len(d) == 0:
        return np.zeros((0, 4))
    return np.stack([d[:, 1], d[:, 0], d[:, 1] + d[:, 3], d[:, 0] + d[:, 2]], axis=1)


def _shift_to_one_based(pred):
    return {tid: {f + 1: box for f, box in frames.items()} for tid, frames in pred.items()}


def build_gt_tracks(seq):
    gt_ids = np.unique(seq.gt[:, 1]).astype(int)
    tracks = {}
    for gid in gt_ids:
        rows = seq.gt[seq.gt[:, 1] == gid]
        tracks[gid] = {
            int(r[0]): (r[3], r[2], r[3] + r[5], r[2] + r[4]) for r in rows
        }
    return tracks


def run_naive(dets_by_frame, iou_threshold=IOU_THRESHOLD, max_age=MAX_AGE):
    tracker = NaiveTracker(iou_threshold=iou_threshold, max_age=max_age)
    pred = tracker.run([_to_yxyx(d) for d in dets_by_frame])
    return _shift_to_one_based(pred)


def run_trilha_a(dets_by_frame, model, device, img_w, img_h,
                 iou_threshold=IOU_THRESHOLD, max_age=MAX_AGE):
    tracker = MotionLSTMTracker(model, device, img_w, img_h,
                                iou_threshold=iou_threshold, max_age=max_age)
    dets = [np.asarray(d, dtype=np.float64)[:, :4] for d in dets_by_frame]
    pred = tracker.run(dets)
    return _shift_to_one_based(pred)
