"""
Parte 4 -- ferramentas para medir o horizonte de memória efetivo.

Duas medidas, como pede o enunciado:

  1. ANALÍTICA: ||dL_t/dh_{t-k}|| em função de k (`gradient_horizon`),
     medida em janelas MAIORES que a do treino (a janela de 8 só permite
     k<=7, enquanto as oclusões chegam a centenas de quadros).

  2. EMPÍRICA: para cada oclusão (visibilidade < 0.1 no gt.txt) em que a
     track JÁ seguia o objeto logo antes, o que aconteceu com o estado?
        recovered          -> mesmo ID quando o objeto reaparece
        expired            -> a track foi removida (max_age) antes da volta
        stolen             -> a track passou a seguir OUTRO objeto
        alive_not_matched  -> track viva, mas a caixa extrapolada não
                              sobrepunha o objeto (deriva) -> ID novo
        no_prior_track     -> ninguém seguia o objeto antes: NÃO é falha
                              de memória, e fica FORA das estatísticas.
     A "sobrevivência" é medida em quadros desde a última observação, SEM
     truncar pela duração da oclusão, e resumida por Kaplan-Meier
     (eventos recuperados/vivos são censurados).
"""

import numpy as np
import torch
import torch.nn.functional as F

from src.metrics.tracking_metrics import (
    compute_iou, compute_idf1, count_id_switches, count_fragmentations,
)
from src.tracking.motion_lstm_tracker import MotionLSTMTracker
from src.tracking.naive_tracker import NaiveTracker

IOU_ASSIGN = 0.30
VISIBILITY_THRESHOLD = 0.10
PRE_LIMIT = 5      # a track precisa estar seguindo o objeto até 5 quadros antes da oclusão
POST_LIMIT = 10    # janela para procurar o objeto de volta depois da oclusão
FAILURE_CATEGORIES = ("expired", "stolen", "alive_not_matched")


# ----------------------------------------------------------------------
# dados / execução
# ----------------------------------------------------------------------
def left_top_wh_to_yxyx(box):
    left, top, w, h = box
    return (top, left, top + h, left + w)


def build_gt_tracks(seq):
    tracks = {}
    for row in seq.gt:
        frame, gid = int(row[0]), int(row[1])
        tracks.setdefault(gid, {})[frame] = left_top_wh_to_yxyx(row[2:6])
    return tracks


def build_gt_by_frame(gt_tracks):
    by_frame = {}
    for gid, frames in gt_tracks.items():
        for f, box in frames.items():
            by_frame.setdefault(f, []).append((gid, box))
    return by_frame


def build_visibility(seq):
    vis = {}
    for row in seq.gt:
        vis.setdefault(int(row[1]), {})[int(row[0])] = float(row[8])
    return vis


def run_lstm_tracker(seq, model, device, max_age, legacy=False):
    tracker = MotionLSTMTracker(
        model, device, seq.info.im_width, seq.info.im_height,
        iou_threshold=IOU_ASSIGN, max_age=max_age, legacy_state_update=legacy,
    )
    detections = [seq.detections_at(t) for t in range(1, seq.info.seq_length + 1)]
    tracker.run(detections, first_frame=1)   # já em quadros 1-based
    return tracker


def run_naive_tracker(seq, max_age):
    dets = []
    for t in range(1, seq.info.seq_length + 1):
        d = seq.detections_at(t)
        dets.append(np.array([left_top_wh_to_yxyx(x) for x in d]) if len(d) else np.zeros((0, 4)))
    history = NaiveTracker(iou_threshold=IOU_ASSIGN, max_age=max_age).run(dets)
    return {tid: {f + 1: b for f, b in fr.items()} for tid, fr in history.items()}


def evaluate_history(history, gt_tracks):
    idf1, idtp, idfp, idfn, _ = compute_idf1(history, gt_tracks)
    return {
        "idf1": float(idf1),
        "id_switches": int(count_id_switches(history, gt_tracks)),
        "fragmentations": int(count_fragmentations(history, gt_tracks)),
        "predicted_identities": len(history),
        "ground_truth_identities": len(gt_tracks),
    }


# ----------------------------------------------------------------------
# medida analítica
# ----------------------------------------------------------------------
def gradient_horizon(model, trajectories, device, window, max_samples=256, seed=17):
    """
    Norma de dL_t/dh_{t-k}, k = 0..window-1, com L_t a perda (smooth-L1) da
    ÚLTIMA previsão de uma janela rodada em free-running (as entradas após
    o 1º passo são as próprias previsões, destacadas do grafo -- então o
    único caminho de gradiente até h_{t-k} é a recorrência).

    `window` pode ser maior que a janela de treino: mede uma propriedade
    da célula treinada, não do treino.
    """
    eligible = [t for t in trajectories if len(t) >= window + 1]
    if not eligible:
        raise ValueError(f"Nenhuma trajetória contínua com >= {window + 1} quadros.")
    rng = np.random.default_rng(seed)
    lag_values = [[] for _ in range(window)]

    was_training = model.training
    model.eval()
    # cuDNN não permite backward de RNN em modo eval (erro em GPU)
    with torch.backends.cudnn.flags(enabled=False):
        for _ in range(max_samples):
            traj = eligible[int(rng.integers(len(eligible)))]
            start = int(rng.integers(len(traj) - window))
            first = torch.as_tensor(traj[start], dtype=torch.float32, device=device).view(1, 1, 4)
            target = torch.as_tensor(traj[start + window], dtype=torch.float32, device=device).view(1, 1, 4)

            model.zero_grad(set_to_none=True)
            hidden, current, steps, prediction = None, first, [], None
            for step in range(window):
                _, hidden = model.lstm(current, hidden)
                h = hidden[0]
                h.retain_grad()
                steps.append(h)
                prediction = current + model.head(h[-1].unsqueeze(1))
                if step < window - 1:
                    current = prediction.detach()
            F.smooth_l1_loss(prediction, target).backward()
            for lag in range(window):
                g = steps[window - 1 - lag].grad[-1, 0]
                lag_values[lag].append(float(torch.linalg.vector_norm(g).item()))
    model.train(was_training)

    med = [float(np.median(v)) for v in lag_values]
    return {
        "window": window,
        "samples": max_samples,
        "lags_frames": list(range(window)),
        "median_gradient_norm": med,
        "mean_gradient_norm": [float(np.mean(v)) for v in lag_values],
        "q25_gradient_norm": [float(np.quantile(v, 0.25)) for v in lag_values],
        "q75_gradient_norm": [float(np.quantile(v, 0.75)) for v in lag_values],
        "relative_to_lag0": [m / med[0] if med[0] > 0 else 0.0 for m in med],
    }


def summarize_gradient_curve(curve, train_window):
    """Números que entram no diagnóstico: k_eff, atenuação, decaimento/passo."""
    med = np.asarray(curve["median_gradient_norm"])
    ratio = med / med[0] if med[0] > 0 else np.zeros_like(med)
    out = {}
    for thr in (0.1, 0.01):
        idx = np.where(ratio < thr)[0]
        out[f"k_eff_{int(thr * 100)}pct"] = int(idx[0]) if len(idx) else None
    k = min(train_window, len(ratio)) - 1
    out["attenuation_at_train_window"] = float(1 / ratio[k]) if ratio[k] > 0 else float("inf")
    out["train_window"] = train_window
    if len(med) >= 5:
        lags = np.arange(len(med))[2:]
        slope = np.polyfit(lags, np.log(np.maximum(med[2:], 1e-30)), 1)[0]
        out["decay_per_step"] = float(np.exp(slope))   # fator multiplicativo médio por quadro (k>=2)
    return out


# ----------------------------------------------------------------------
# medida empírica
# ----------------------------------------------------------------------
def visibility_episodes(seq, threshold=VISIBILITY_THRESHOLD, min_len=2):
    """Intervalos consecutivos do GT com visibilidade < threshold."""
    episodes = []
    for gid in np.unique(seq.gt[:, 1]).astype(int):
        rows = seq.gt[seq.gt[:, 1] == gid]
        rows = rows[np.argsort(rows[:, 0])]
        low = rows[rows[:, 8] < threshold, 0].astype(int)
        if len(low) == 0:
            continue
        start = prev = int(low[0])
        for fr in list(low[1:]) + [None]:
            if fr is None or int(fr) != prev + 1:
                if prev - start + 1 >= min_len:
                    episodes.append({"gt_id": int(gid), "start": start, "end": prev,
                                     "duration": prev - start + 1})
                if fr is not None:
                    start = int(fr)
            if fr is not None:
                prev = int(fr)
    return episodes


def assigned_id_at(pred_tracks, gt_track, frame, threshold=IOU_ASSIGN):
    if frame not in gt_track:
        return None
    best_id, best_iou = None, threshold
    for pid, frames in pred_tracks.items():
        if frame in frames:
            iou = compute_iou(frames[frame], gt_track[frame])
            if iou >= best_iou:
                best_id, best_iou = pid, iou
    return best_id


def nearest_assigned_id(pred_tracks, gt_track, frame, direction, limit):
    for offset in range(limit + 1):
        candidate = frame + direction * offset
        pid = assigned_id_at(pred_tracks, gt_track, candidate)
        if pid is not None:
            return pid, candidate
    return None, None


def _center_error(pred_box, gt_box):
    """Distância entre centros, em alturas do GT (invariante à escala)."""
    pc = ((pred_box[1] + pred_box[3]) / 2, (pred_box[0] + pred_box[2]) / 2)
    gc = ((gt_box[1] + gt_box[3]) / 2, (gt_box[0] + gt_box[2]) / 2)
    return float(np.hypot(pc[0] - gc[0], pc[1] - gc[1]) / max(gt_box[2] - gt_box[0], 1.0))


def analyze_event(ep, seq_id, gt_track, gt_by_frame, tracker):
    start, end, gid = ep["start"], ep["end"], ep["gt_id"]
    hist = tracker.history
    before_id, last_obs = nearest_assigned_id(hist, gt_track, start - 1, -1, PRE_LIMIT)
    after_id, first_reobs = nearest_assigned_id(hist, gt_track, end + 1, +1, POST_LIMIT)

    ev = dict(ep)
    ev.update(sequence=seq_id, before_pred_id=before_id, last_obs_frame=last_obs,
              after_pred_id=after_id, first_reobs_frame=first_reobs)
    if before_id is None:
        ev.update(category="no_prior_track", blind_gap=None, survival_frames=None,
                  survival_censored=None, death_frame=None, stolen_frame=None, gate_loss_frame=None,
                  mean_pred_iou=None, center_err_last=None, pred_frames=0)
        return ev

    blind_gap = end + 1 - last_obs      # quadros sem observação até o objeto voltar a ser visível
    death = tracker.death_frame.get(before_id)
    expired = death is not None and death <= end
    ref_hist = hist.get(before_id, {})
    pred = tracker.predicted_history.get(before_id, {})

    # "roubada": a caixa OBSERVADA da track passa a explicar MELHOR outro GT do que o próprio alvo.
    # (Só exigir IoU>=0.3 com outro GT marcaria quase toda oclusão como roubo, pois o oclusor
    # sobrepõe o alvo por definição.)
    stolen_frame = None
    for f in range(last_obs + 1, end + 1):
        box = ref_hist.get(f)
        if box is None:
            continue
        own = compute_iou(box, gt_track[f]) if f in gt_track else 0.0
        other = max((compute_iou(box, gb) for g, gb in gt_by_frame.get(f, []) if g != gid), default=0.0)
        if other >= IOU_ASSIGN and other > own:
            stolen_frame = f
            break

    # perda do portão: 1º quadro em que a caixa PREVISTA deixa de sobrepor o alvo (IoU < 0.3)
    gate_loss = None
    for f in range(last_obs + 1, end + 1):
        if f in pred and f in gt_track and compute_iou(pred[f], gt_track[f]) < IOU_ASSIGN:
            gate_loss = f
            break

    end_events = []
    if expired:
        end_events.append((death, "expired"))
    if stolen_frame is not None:
        end_events.append((stolen_frame, "stolen"))

    if after_id == before_id:
        category = "recovered"
    elif end_events:
        category = min(end_events)[1]
    else:
        category = "alive_not_matched"

    # sobrevivência do estado = quadros desde a última observação até o 1º de:
    # track removida, track roubada, previsão fora do portão. Recuperados são censurados.
    failure_times = [t for t, _ in end_events] + ([gate_loss] if gate_loss is not None else [])
    failed = category != "recovered" and bool(failure_times)
    survival = (min(failure_times) - last_obs) if failed else blind_gap

    ious, errs = [], []
    for f in range(start, end + 1):
        if f in pred and f in gt_track:
            ious.append(compute_iou(pred[f], gt_track[f]))
            errs.append(_center_error(pred[f], gt_track[f]))

    ev.update(category=category, blind_gap=blind_gap, survival_frames=int(survival),
              survival_censored=not failed, death_frame=death, stolen_frame=stolen_frame,
              gate_loss_frame=gate_loss,
              mean_pred_iou=float(np.mean(ious)) if ious else None,
              center_err_last=errs[-1] if errs else None, pred_frames=len(ious))
    return ev


def collect_events(seq_id, seq, gt_tracks, tracker):
    gt_by_frame = build_gt_by_frame(gt_tracks)
    return [analyze_event(ep, seq_id, gt_tracks[ep["gt_id"]], gt_by_frame, tracker)
            for ep in visibility_episodes(seq)]


def kaplan_meier(times, observed):
    """Curva de sobrevivência do estado. observed=True: falha; False: censurado."""
    times = np.asarray(times)
    observed = np.asarray(observed, dtype=bool)
    ts, ss, surv = [0], [1.0], 1.0
    for t in np.unique(times[observed]):
        at_risk = (times >= t).sum()
        deaths = ((times == t) & observed).sum()
        surv *= 1 - deaths / at_risk
        ts.append(int(t))
        ss.append(float(surv))
    return np.array(ts), np.array(ss)


def recovery_table(events, edges=(0, 5, 10, 20, 30, 60, 10 ** 6)):
    """Taxa de recuperação e causas de falha por faixa de quadros sem observação."""
    tracked = [e for e in events if e["category"] != "no_prior_track"]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [e for e in tracked if lo < e["blind_gap"] <= hi]
        counts = {c: sum(e["category"] == c for e in sel) for c in ("recovered",) + FAILURE_CATEGORIES}
        rows.append({"bin": f"{lo + 1}-{hi}" if hi < 10 ** 6 else f">{lo}", "n": len(sel), **counts,
                     "recovery_rate": counts["recovered"] / len(sel) if sel else None})
    return rows


def overall_recovery(events):
    tracked = [e for e in events if e["category"] != "no_prior_track"]
    if not tracked:
        return {"n_tracked": 0, "recovery_rate": None}
    return {"n_tracked": len(tracked),
            "n_no_prior_track": len(events) - len(tracked),
            "recovery_rate": sum(e["category"] == "recovered" for e in tracked) / len(tracked),
            **{c: sum(e["category"] == c for e in tracked) for c in FAILURE_CATEGORIES}}