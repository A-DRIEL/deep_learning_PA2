"""
mAP de detecção (classe única: pessoa), implementação própria.

Protocolo (estilo COCO, uma classe):
  - detecções de TODOS os quadros da sequência são ordenadas por confiança
    decrescente
  - para cada limiar de IoU em {0.50, 0.55, ..., 0.95}: cada detecção tenta
    casar com o GT AINDA LIVRE do mesmo quadro de maior IoU; se IoU >= limiar
    é TP, senão FP; cada GT casa no máximo uma vez
  - AP = área sob a curva precisão x revocação (interpolação "all-point")
  - mAP = média dos APs sobre os limiares; AP50 é reportado à parte

Atenção: isto NÃO é a métrica "por quadro" (tp/(tp+fp+fn)) do
part1_difficulty_plot.py; aquela ignora a confiança. Aqui a confiança
importa, então FPs com score alto doem mais que FPs com score baixo.
"""

import numpy as np


def iou_matrix_xywh(a, b):
    """a: (N,4), b: (M,4), ambos em (left, top, w, h). Devolve (N,M)."""
    ax0, ay0 = a[:, 0:1], a[:, 1:2]
    ax1, ay1 = ax0 + a[:, 2:3], ay0 + a[:, 3:4]
    bx0, by0 = b[:, 0][None, :], b[:, 1][None, :]
    bx1, by1 = bx0 + b[:, 2][None, :], by0 + b[:, 3][None, :]

    iw = np.clip(np.minimum(ax1, bx1) - np.maximum(ax0, bx0), 0, None)
    ih = np.clip(np.minimum(ay1, by1) - np.maximum(ay0, by0), 0, None)
    inter = iw * ih
    union = a[:, 2:3] * a[:, 3:4] + (b[:, 2] * b[:, 3])[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def _ap_from_pr(recall, precision):
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]  # envelope monotônico
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def detection_map(dets_by_frame, gt_by_frame, thresholds=None):
    """
    dets_by_frame: lista de arrays (N_t, 5) = [left, top, w, h, conf]
    gt_by_frame:   lista de arrays (M_t, 4)  = [left, top, w, h]
    Devolve dict com 'map', 'ap50' e 'ap_per_threshold'.
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.5, 1.0, 0.05), 2)

    n_gt_total = sum(len(g) for g in gt_by_frame)
    if n_gt_total == 0:
        return {"map": float("nan"), "ap50": float("nan"), "ap_per_threshold": []}

    ious, conf_l, frame_l, idx_l = [], [], [], []
    for f, (d, g) in enumerate(zip(dets_by_frame, gt_by_frame)):
        n = len(d)
        if n == 0:
            ious.append(None)
            continue
        ious.append(iou_matrix_xywh(d[:, :4], g) if len(g) else np.zeros((n, 0)))
        conf_l.append(d[:, 4])
        frame_l.append(np.full(n, f, dtype=np.int64))
        idx_l.append(np.arange(n))

    if not conf_l:
        return {"map": 0.0, "ap50": 0.0, "ap_per_threshold": [0.0] * len(thresholds)}

    conf = np.concatenate(conf_l)
    order = np.argsort(-conf, kind="stable")
    frames = np.concatenate(frame_l)[order]
    idxs = np.concatenate(idx_l)[order]

    aps = []
    for thr in thresholds:
        matched = [np.zeros(len(g), dtype=bool) for g in gt_by_frame]
        tp = np.zeros(len(order))
        for k in range(len(order)):
            f, i = int(frames[k]), int(idxs[k])
            row = ious[f][i]
            if row.size == 0:
                continue
            cand = np.where(matched[f], -1.0, row)
            j = int(cand.argmax())
            if cand[j] >= thr:
                matched[f][j] = True
                tp[k] = 1.0
        ctp = np.cumsum(tp)
        cfp = np.cumsum(1.0 - tp)
        recall = ctp / n_gt_total
        precision = ctp / np.maximum(ctp + cfp, 1e-9)
        aps.append(_ap_from_pr(recall, precision))

    return {
        "map": float(np.mean(aps)),
        "ap50": float(aps[0]),
        "ap_per_threshold": [float(a) for a in aps],
    }
