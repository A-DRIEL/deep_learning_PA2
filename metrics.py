"""
metrics.py -- IDF1, ID switches e fragmentações, implementação própria.

Sem motmetrics / TrackEval / py-motmetrics (proibidos pelo enunciado): só
numpy e scipy (o scipy entra apenas com `linear_sum_assignment`, o algoritmo
Húngaro, para a atribuição global do IDF1).

Convenções (as mesmas do resto do repositório)
----------------------------------------------
    caixa  = (y0, x0, y1, x1), em pixels
    tracks = dict  id -> {frame: caixa}
    O mesmo formato serve para predição (`pred_tracks`) e para o ground truth
    (`gt_tracks`). Os quadros só precisam estar na mesma indexação nos dois.

Definições
----------
IDF1
    Atribuição GLOBAL um-para-um entre identidades previstas e reais, na
    sequência inteira (não por quadro). A matriz de custo é "em quantos quadros
    o par (pred_id, gt_id) tem IoU >= limiar"; o Húngaro maximiza a soma.
        IDTP = soma da matriz nos pares escolhidos
        IDFP = (quadros previstos) - IDTP
        IDFN = (quadros reais)     - IDTP
        IDF1 = 2*IDTP / (2*IDTP + IDFP + IDFN)

ID switch
    Em cada quadro, cada GT é associado ao pred_id de maior IoU (>= limiar).
    Há um switch quando esse pred_id muda entre dois quadros CONSECUTIVOS em
    que o GT estava sendo rastreado. Se existe uma lacuna entre os dois
    (1+ quadros sem correspondência), a troca de ID NÃO é switch: é contada
    como fragmentação. (O CLEAR-MOT clássico conta também a troca depois de
    uma lacuna; para reproduzir essa variante use `require_consecutive=False`.)

Fragmentação
    O GT estava sendo rastreado, passa por uma LACUNA (1+ quadros sem nenhuma
    predição com IoU >= limiar) e volta a ser rastreado -- qualquer que seja o ID
    usado ao retomar. Cada retomada conta 1.

Uso
---
    from metrics import compute_idf1, count_id_switches, count_fragmentations
    idf1, idtp, idfp, idfn, mapping = compute_idf1(pred_tracks, gt_tracks)

    python metrics.py        # roda os autotestes (casos construídos à mão)
"""

from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

__all__ = [
    "compute_iou", "iou_matrix", "build_cooccurrence_matrix", "compute_idf1",
    "count_id_switches", "count_fragmentations", "identity_count_error",
    "evaluate_tracking",
]


# ----------------------------------------------------------------------
# geometria
# ----------------------------------------------------------------------
def compute_iou(box_a, box_b):
    """IoU entre duas caixas (y0, x0, y1, x1)."""
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


def iou_matrix(boxes_a, boxes_b):
    """IoU par-a-par, vetorizado. boxes_a: (N,4), boxes_b: (M,4) -> (N,M)."""
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    ih = np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0])
    iw = np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1])
    inter = np.maximum(ih, 0.0) * np.maximum(iw, 0.0)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    out = np.zeros_like(inter)
    np.divide(inter, union, out=out, where=inter > 0)
    return out


def _index_by_frame(tracks):
    """tracks -> {frame: (ids, caixas (k,4))}, preservando a ordem de inserção dos ids."""
    ids_at, boxes_at = defaultdict(list), defaultdict(list)
    for tid, frames in tracks.items():
        for f, box in frames.items():
            ids_at[f].append(tid)
            boxes_at[f].append(box)
    return {f: (ids_at[f], np.asarray(boxes_at[f], dtype=np.float64)) for f in ids_at}


# ----------------------------------------------------------------------
# IDF1
# ----------------------------------------------------------------------
def build_cooccurrence_matrix(pred_tracks, gt_tracks, iou_threshold=0.5):
    """
    Devolve (matriz, pred_ids, gt_ids): matriz[i, j] = nº de quadros em que
    pred_ids[i] e gt_ids[j] têm IoU >= iou_threshold. É o que o Húngaro maximiza.
    """
    pred_ids, gt_ids = list(pred_tracks.keys()), list(gt_tracks.keys())
    p_index = {pid: i for i, pid in enumerate(pred_ids)}
    g_index = {gid: j for j, gid in enumerate(gt_ids)}
    matrix = np.zeros((len(pred_ids), len(gt_ids)), dtype=np.int64)

    pred_by_frame, gt_by_frame = _index_by_frame(pred_tracks), _index_by_frame(gt_tracks)
    for f in pred_by_frame.keys() & gt_by_frame.keys():
        p_ids, p_boxes = pred_by_frame[f]
        g_ids, g_boxes = gt_by_frame[f]
        rows, cols = np.nonzero(iou_matrix(p_boxes, g_boxes) >= iou_threshold)
        if len(rows) == 0:
            continue
        pi = np.array([p_index[p_ids[r]] for r in rows])
        gj = np.array([g_index[g_ids[c]] for c in cols])
        matrix[pi, gj] += 1      # um par (pred, gt) aparece no máximo uma vez por quadro
    return matrix, pred_ids, gt_ids


def compute_idf1(pred_tracks, gt_tracks, iou_threshold=0.5):
    """
    Devolve (idf1, idtp, idfp, idfn, id_mapping).
    id_mapping: dict pred_id -> gt_id, só dos pares emparelhados (que coincidiram
    em pelo menos um quadro).
    """
    total_pred = sum(len(v) for v in pred_tracks.values())
    total_gt = sum(len(v) for v in gt_tracks.values())
    matrix, pred_ids, gt_ids = build_cooccurrence_matrix(pred_tracks, gt_tracks, iou_threshold)

    if matrix.size == 0:
        # sem predição ou sem GT: nada a emparelhar. (Vazio vs. vazio é 1.0 por convenção.)
        return (1.0 if total_pred + total_gt == 0 else 0.0), 0, total_pred, total_gt, {}

    # linear_sum_assignment minimiza, e queremos maximizar a coocorrência -> nega
    row_idx, col_idx = linear_sum_assignment(-matrix)

    idtp, id_mapping = 0, {}
    for r, c in zip(row_idx, col_idx):
        if matrix[r, c] > 0:
            idtp += int(matrix[r, c])
            id_mapping[pred_ids[r]] = gt_ids[c]

    idfp = total_pred - idtp
    idfn = total_gt - idtp
    denom = 2 * idtp + idfp + idfn
    idf1 = (2 * idtp / denom) if denom > 0 else 1.0
    return idf1, idtp, idfp, idfn, id_mapping


# ----------------------------------------------------------------------
# ID switches e fragmentações
# ----------------------------------------------------------------------
def count_id_switches(pred_tracks, gt_tracks, iou_threshold=0.5, require_consecutive=True):
    """
    Nº de ID switches (ver definição no topo do arquivo). A associação GT -> pred_id é
    POR QUADRO (maior IoU >= limiar; em empate, vale o pred_id inserido por último),
    independente do emparelhamento global do IDF1.
    """
    pred_by_frame = _index_by_frame(pred_tracks)
    gt_by_frame = _index_by_frame(gt_tracks)

    last_pred, last_frame, switches = {}, {}, 0
    for f in sorted(gt_by_frame):
        g_ids, g_boxes = gt_by_frame[f]
        if f in pred_by_frame:
            p_ids, p_boxes = pred_by_frame[f]
            ious = iou_matrix(g_boxes, p_boxes)
        else:
            p_ids, ious = [], None

        for row, gid in enumerate(g_ids):
            best_pid = None
            if ious is not None:
                best = ious[row].max()
                if best >= iou_threshold:
                    best_pid = p_ids[int(np.flatnonzero(ious[row] == best)[-1])]
            if best_pid is None:
                continue

            consecutive = (last_frame.get(gid) == f - 1) or not require_consecutive
            if consecutive and gid in last_pred and best_pid != last_pred[gid]:
                switches += 1
            last_pred[gid] = best_pid
            last_frame[gid] = f
    return switches


def count_fragmentations(pred_tracks, gt_tracks, iou_threshold=0.5):
    """Nº de fragmentações (ver definição no topo do arquivo)."""
    pred_by_frame = _index_by_frame(pred_tracks)
    gt_by_frame = _index_by_frame(gt_tracks)

    was_matched = defaultdict(bool)
    gap_since_match = defaultdict(bool)
    fragmentations = 0
    for f in sorted(gt_by_frame):
        g_ids, g_boxes = gt_by_frame[f]
        if f in pred_by_frame:
            matched = iou_matrix(g_boxes, pred_by_frame[f][1]).max(axis=1) >= iou_threshold
        else:
            matched = np.zeros(len(g_ids), dtype=bool)

        for gid, ok in zip(g_ids, matched):
            if ok:
                if not was_matched[gid] and gap_since_match[gid]:
                    fragmentations += 1
                was_matched[gid] = True
                gap_since_match[gid] = False
            else:
                if was_matched[gid]:
                    gap_since_match[gid] = True
                was_matched[gid] = False
    return fragmentations


# ----------------------------------------------------------------------
# contagem de identidades únicas + atalho
# ----------------------------------------------------------------------
def identity_count_error(pred_tracks, gt_tracks):
    """Erro de contagem de identidades únicas (análogo temporal do erro de contagem do PA1)."""
    n_pred, n_gt = len(pred_tracks), len(gt_tracks)
    return {
        "n_pred_ids": n_pred, "n_gt_ids": n_gt,
        "count_error": abs(n_pred - n_gt),
        "count_ratio": (n_pred / n_gt) if n_gt else float("nan"),
    }


def evaluate_tracking(pred_tracks, gt_tracks, iou_threshold=0.5):
    """Todas as métricas de uma vez, num dict."""
    idf1, idtp, idfp, idfn, _ = compute_idf1(pred_tracks, gt_tracks, iou_threshold)
    n_gt = len(gt_tracks)
    switches = count_id_switches(pred_tracks, gt_tracks, iou_threshold)
    return {
        "idf1": float(idf1), "idtp": idtp, "idfp": idfp, "idfn": idfn,
        "id_switches": switches,
        "switches_per_id": (switches / n_gt) if n_gt else float("nan"),
        "fragmentations": count_fragmentations(pred_tracks, gt_tracks, iou_threshold),
        **identity_count_error(pred_tracks, gt_tracks),
    }


# ----------------------------------------------------------------------
# autotestes: casos construídos à mão, com o resultado esperado calculado
# no papel (rodar: python metrics.py)
# ----------------------------------------------------------------------
def _selftest():
    near = lambda a, b: abs(a - b) < 1e-9

    # (a) predição == GT  =>  IDF1 = 1 e zero switches
    gt = {1: {0: (0, 0, 10, 10), 1: (1, 1, 11, 11), 2: (2, 2, 12, 12)}}
    pred = {7: dict(gt[1])}                      # o NOME do id não importa, só a consistência
    idf1, idtp, idfp, idfn, mapping = compute_idf1(pred, gt)
    sw, fr = count_id_switches(pred, gt), count_fragmentations(pred, gt)
    print(f"(a) pred = gt:             IDF1={idf1:.3f} (esp. 1.000)  switches={sw} (esp. 0)  frags={fr} (esp. 0)")
    assert near(idf1, 1.0) and sw == 0 and fr == 0 and mapping == {7: 1}

    # (b) duas identidades trocam a partir do quadro k=2
    #     IDTP=4 de 8 quadros -> IDF1 = 8/(8+4+4) = 0.5; cada GT sofre 1 troca -> 2 switches
    gt = {
        1: {0: (0, 0, 10, 10), 1: (1, 1, 11, 11), 2: (2, 2, 12, 12), 3: (3, 3, 13, 13)},
        2: {0: (50, 50, 60, 60), 1: (51, 51, 61, 61), 2: (52, 52, 62, 62), 3: (53, 53, 63, 63)},
    }
    pred = {
        10: {0: (0, 0, 10, 10), 1: (1, 1, 11, 11), 2: (52, 52, 62, 62), 3: (53, 53, 63, 63)},
        20: {0: (50, 50, 60, 60), 1: (51, 51, 61, 61), 2: (2, 2, 12, 12), 3: (3, 3, 13, 13)},
    }
    idf1_b = compute_idf1(pred, gt)[0]
    sw_b, fr_b = count_id_switches(pred, gt), count_fragmentations(pred, gt)
    print(f"(b) troca de ids em k=2:   IDF1={idf1_b:.3f} (esp. 0.500)  switches={sw_b} (esp. 2)  frags={fr_b} (esp. 0)")
    assert near(idf1_b, 0.5) and sw_b == 2 and fr_b == 0

    # (c) uma track partida em duas, COM lacuna no quadro 2
    #     IDTP=2; pred=4 quadros, GT=5 -> IDF1 = 4/(4+2+3) = 0.444
    #     troca depois de lacuna NÃO é switch (é fragmentação): 0 switches, 1 fragmentação
    gt = {1: {i: (i, i, 10 + i, 10 + i) for i in range(5)}}
    pred = {
        100: {0: gt[1][0], 1: gt[1][1]},
        200: {3: gt[1][3], 4: gt[1][4]},
    }
    idf1_c = compute_idf1(pred, gt)[0]
    sw_c, fr_c = count_id_switches(pred, gt), count_fragmentations(pred, gt)
    print(f"(c) partida com lacuna:    IDF1={idf1_c:.3f} (esp. 0.444)  switches={sw_c} (esp. 0)  frags={fr_c} (esp. 1)")
    assert near(idf1_c, 4 / 9) and sw_c == 0 and fr_c == 1

    # (d) mesma track partida em duas, SEM lacuna (troca direta no quadro 2)
    #     IDTP=3; pred=5, GT=5 -> IDF1 = 6/10 = 0.600; 1 switch, 0 fragmentações
    pred = {100: {i: gt[1][i] for i in (0, 1)}, 200: {i: gt[1][i] for i in (2, 3, 4)}}
    idf1_d = compute_idf1(pred, gt)[0]
    sw_d, fr_d = count_id_switches(pred, gt), count_fragmentations(pred, gt)
    print(f"(d) partida sem lacuna:    IDF1={idf1_d:.3f} (esp. 0.600)  switches={sw_d} (esp. 1)  frags={fr_d} (esp. 0)")
    assert near(idf1_d, 0.6) and sw_d == 1 and fr_d == 0

    # (e) com lacuna, a variante CLEAR-MOT (require_consecutive=False) conta 1 switch
    pred = {100: {0: gt[1][0], 1: gt[1][1]}, 200: {3: gt[1][3], 4: gt[1][4]}}
    sw_e = count_id_switches(pred, gt, require_consecutive=False)
    print(f"(e) (c) com CLEAR-MOT:     switches={sw_e} (esp. 1)")
    assert sw_e == 1

    # (f) bordas: sem predição / vazio vs. vazio
    assert compute_idf1({}, gt)[0] == 0.0 and compute_idf1({}, {})[0] == 1.0
    assert count_id_switches({}, gt) == 0 and count_fragmentations({}, gt) == 0

    # (g) (b) e (c) NÃO podem dar o mesmo IDF1 -- cada falha pesa de um jeito
    assert not near(idf1_b, idf1_c)
    print("\nTodos os autotestes passaram.")


if __name__ == "__main__":
    _selftest()
