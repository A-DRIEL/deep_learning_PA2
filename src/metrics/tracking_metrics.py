"""
IDF1 e ID switches -- implementação própria, do zero.

IDF1 exige uma correspondência GLOBAL (não por quadro) entre IDs
previstos e IDs reais, resolvida com o algoritmo Húngaro sobre uma
matriz de "quantos quadros cada par de IDs coincide o suficiente".
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_iou(box_a, box_b):
    """box = (y0, x0, y1, x1)"""
    y0 = max(box_a[0], box_b[0])
    x0 = max(box_a[1], box_b[1])
    y1 = min(box_a[2], box_b[2])
    x1 = min(box_a[3], box_b[3])
    if y1 <= y0 or x1 <= x0:
        return 0.0
    intersection = (y1 - y0) * (x1 - x0)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return intersection / (area_a + area_b - intersection)


def build_cooccurrence_matrix(pred_tracks, gt_tracks, iou_threshold=0.5):
    """
    pred_tracks: dict id_previsto -> {frame: box}
    gt_tracks:   dict id_real -> {frame: box}

    Devolve matriz (n_pred, n_gt) com a contagem de quadros em que
    aquele par de IDs tem IoU >= iou_threshold -- essa contagem é o
    que o Húngaro vai maximizar.
    """
    pred_ids = list(pred_tracks.keys())
    gt_ids = list(gt_tracks.keys())
    matrix = np.zeros((len(pred_ids), len(gt_ids)), dtype=np.int64)

    for i, pid in enumerate(pred_ids):
        for j, gid in enumerate(gt_ids):
            common_frames = set(pred_tracks[pid]) & set(gt_tracks[gid])
            count = sum(
                1 for f in common_frames
                if compute_iou(pred_tracks[pid][f], gt_tracks[gid][f]) >= iou_threshold
            )
            matrix[i, j] = count

    return matrix, pred_ids, gt_ids


def compute_idf1(pred_tracks, gt_tracks, iou_threshold=0.5):
    """
    pred_tracks / gt_tracks: dict id -> {frame: box}

    Devolve (idf1, idtp, idfp, idfn, id_mapping)
    id_mapping: dict id_previsto -> id_real (só os pares emparelhados)
    """
    matrix, pred_ids, gt_ids = build_cooccurrence_matrix(pred_tracks, gt_tracks, iou_threshold)

    if matrix.size == 0:
        total_gt_frames = sum(len(v) for v in gt_tracks.values())
        total_pred_frames = sum(len(v) for v in pred_tracks.values())
        return 0.0, 0, total_pred_frames, total_gt_frames, {}

    # Húngaro maximiza -- linear_sum_assignment só minimiza, então negamos
    row_idx, col_idx = linear_sum_assignment(-matrix)

    idtp = 0
    id_mapping = {}
    for r, c in zip(row_idx, col_idx):
        if matrix[r, c] > 0:  # só conta como emparelhado se coincidiu ALGUMA vez
            idtp += matrix[r, c]
            id_mapping[pred_ids[r]] = gt_ids[c]

    total_pred_frames = sum(len(v) for v in pred_tracks.values())
    total_gt_frames = sum(len(v) for v in gt_tracks.values())

    idfp = total_pred_frames - idtp
    idfn = total_gt_frames - idtp

    denom = 2 * idtp + idfp + idfn
    idf1 = (2 * idtp / denom) if denom > 0 else 1.0

    return idf1, idtp, idfp, idfn, id_mapping


def count_id_switches(pred_tracks, gt_tracks, iou_threshold=0.5):
    """
    Conta switches: mudança de ID previsto associado a um GT entre dois
    quadros CONSECUTIVOS em que o GT estava sendo rastreado (sem lacuna).
    """
    gt_id_order = sorted(gt_tracks.keys())
    all_frames = sorted(set().union(*[set(v) for v in gt_tracks.values()]))

    last_pred_for_gt = {gid: None for gid in gt_id_order}
    last_frame_matched = {gid: None for gid in gt_id_order}
    switches = 0

    for f in all_frames:
        for gid in gt_id_order:
            if f not in gt_tracks[gid]:
                continue
            gt_box = gt_tracks[gid][f]

            best_pid, best_iou = None, iou_threshold
            for pid, frames in pred_tracks.items():
                if f in frames:
                    iou = compute_iou(frames[f], gt_box)
                    if iou >= best_iou:
                        best_pid, best_iou = pid, iou

            is_consecutive = (last_frame_matched[gid] == f - 1)
            if is_consecutive and last_pred_for_gt[gid] is not None and best_pid is not None:
                if best_pid != last_pred_for_gt[gid]:
                    switches += 1

            if best_pid is not None:
                last_pred_for_gt[gid] = best_pid
                last_frame_matched[gid] = f

    return switches


def count_fragmentations(pred_tracks, gt_tracks, iou_threshold=0.5):
    """
    Conta fragmentações: quadros em que o GT estava sendo rastreado,
    depois passa por uma LACUNA (1+ quadros sem correspondência), e
    depois volta a ser rastreado -- independente do ID usado ao retomar.
    """
    gt_id_order = sorted(gt_tracks.keys())
    all_frames = sorted(set().union(*[set(v) for v in gt_tracks.values()]))

    was_matched = {gid: False for gid in gt_id_order}
    had_gap_since_last_match = {gid: False for gid in gt_id_order}
    fragmentations = 0

    for f in all_frames:
        for gid in gt_id_order:
            if f not in gt_tracks[gid]:
                continue
            gt_box = gt_tracks[gid][f]

            matched_now = any(
                f in frames and compute_iou(frames[f], gt_box) >= iou_threshold
                for frames in pred_tracks.values()
            )

            if matched_now:
                if was_matched[gid] is False and had_gap_since_last_match[gid]:
                    fragmentations += 1
                was_matched[gid] = True
                had_gap_since_last_match[gid] = False
            else:
                if was_matched[gid]:
                    had_gap_since_last_match[gid] = True
                was_matched[gid] = False

    return fragmentations