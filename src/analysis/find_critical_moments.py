# src/analysis/find_critical_moments.py
"""
Encontra, para uma sequência e um par de trackers, o objeto real (gt_id)
e a janela de quadros onde a diferença entre os dois trackers é mais
ilustrativa -- usado para escolher O QUE mostrar na figura comparativa,
em vez de pegar quadros aleatórios.
"""

import numpy as np

from src.metrics.tracking_metrics import compute_iou


def assigned_pred_id_per_frame(pred_tracks, gt_track, iou_threshold=0.3):
    """Para cada quadro do gt_track, descobre qual pred_id está cobrindo
    aquele objeto (ou None se nenhum cobre o suficiente)."""
    assignment = {}
    for f, gt_box in gt_track.items():
        best_pid, best_iou = None, iou_threshold
        for pid, frames in pred_tracks.items():
            if f in frames:
                iou = compute_iou(frames[f], gt_box)
                if iou >= best_iou:
                    best_pid, best_iou = pid, iou
        assignment[f] = best_pid
    return assignment


def count_switch_events(assignment):
    """Lista os quadros onde o pred_id atribuído MUDA em relação ao anterior."""
    frames = sorted(assignment.keys())
    events = []
    last_pid = None
    for f in frames:
        pid = assignment[f]
        if last_pid is not None and pid is not None and pid != last_pid:
            events.append(f)
        if pid is not None:
            last_pid = pid
    return events


def find_recovery_case(pred_naive, pred_trilha_a, gt_tracks, iou_threshold=0.3, min_length=15):
    """
    Caso de SUCESSO: gt_id onde a Trilha A tem MENOS switches que o
    baseline (não precisa ser zero), priorizando a maior melhora absoluta.
    """
    best = None
    for gid, gt_track in gt_tracks.items():
        if len(gt_track) < min_length:
            continue
        assign_naive = assigned_pred_id_per_frame(pred_naive, gt_track, iou_threshold)
        assign_a = assigned_pred_id_per_frame(pred_trilha_a, gt_track, iou_threshold)

        naive_events = count_switch_events(assign_naive)
        a_events = count_switch_events(assign_a)
        improvement = len(naive_events) - len(a_events)

        if improvement > 0 and len(naive_events) > 0:
            event_frame = naive_events[0]
            if best is None or improvement > best[3]:
                best = (gid, event_frame, len(gt_track), improvement)

    return best


def find_lock_in_failure_case(pred_naive, pred_trilha_a, gt_tracks, iou_threshold=0.3, min_length=15):
    """
    Caso de FALHA: Trilha A tem menos switches que o baseline, mas fica
    MUITO TEMPO presa a um pred_id, numa sequência que NÃO começa logo
    no primeiro quadro observado do objeto (para não contar "nunca trocou
    porque é o início" como se fosse um evento de lock-in real).
    """
    best = None
    for gid, gt_track in gt_tracks.items():
        if len(gt_track) < min_length:
            continue
        frames = sorted(gt_track.keys())
        first_frame = frames[0]

        assign_naive = assigned_pred_id_per_frame(pred_naive, gt_track, iou_threshold)
        assign_a = assigned_pred_id_per_frame(pred_trilha_a, gt_track, iou_threshold)

        naive_events = count_switch_events(assign_naive)
        a_events = count_switch_events(assign_a)
        if len(a_events) >= len(naive_events):
            continue

        run_start_frame, run_len, max_run, max_run_start = frames[0], 1, 1, frames[0]
        for i in range(1, len(frames)):
            if assign_a[frames[i]] == assign_a[frames[i - 1]] and assign_a[frames[i]] is not None:
                run_len += 1
                if run_len > max_run:
                    max_run = run_len
                    max_run_start = run_start_frame
            else:
                run_len = 1
                run_start_frame = frames[i]

        if max_run > 15 and max_run_start != first_frame:
            if best is None or max_run > best[2]:
                best = (gid, max_run_start, max_run)

    return best