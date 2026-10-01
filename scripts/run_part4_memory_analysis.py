"""Parte 4: galeria de falhas e medidas do horizonte de memória.

Usa o checkpoint free-running da Parte 3, as anotações visibility do MOT17
e o tracker implementado no projeto. Gera curvas, tabelas e três galerias em
outputs/part4/. Nenhum rastreador ou métrica externa é usado.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
import torch.nn.functional as F

from src.datasets.mot17 import MOT17Sequence, SEQUENCE_IDS
from src.datasets.trajectory_dataset import extract_trajectories
from src.metrics.tracking_metrics import compute_idf1, count_id_switches, count_fragmentations, compute_iou
from src.models.motion_lstm import MotionLSTM
from src.tracking.motion_lstm_tracker import MotionLSTMTracker


DETECTOR = "SDP"
CHECKPOINT = Path("outputs/ablation_checkpoints/free_running_seed0.pt")
OUTPUT = Path("outputs/part4")
WINDOW_SIZE = 8  # janela de BPTT efetivamente usada na Parte 3
VISIBILITY_THRESHOLD = 0.10
BASE_MAX_AGE = 30
CORRECTED_MAX_AGE = 60
IOU_THRESHOLD = 0.30


def build_gt_tracks(seq):
    tracks = {}
    for row in seq.gt:
        frame, gid = int(row[0]), int(row[1])
        left, top, width, height = row[2:6]
        tracks.setdefault(gid, {})[frame] = (top, left, top + height, left + width)
    return tracks


def run_tracker(seq, model, device, max_age):
    tracker = MotionLSTMTracker(
        model, device, seq.info.im_width, seq.info.im_height,
        iou_threshold=IOU_THRESHOLD, max_age=max_age,
    )
    detections = [seq.detections_at(frame) for frame in range(1, seq.info.seq_length + 1)]
    tracker.run(detections)
    # MOT17 é 1-based; o enumerate interno do tracker começa em zero.
    tracker.history = {
        tid: {frame + 1: box for frame, box in frames.items()}
        for tid, frames in tracker.history.items()
    }
    tracker.predicted_history = {
        tid: {frame + 1: box for frame, box in frames.items()}
        for tid, frames in tracker.predicted_history.items()
    }
    return tracker


def visibility_occlusions(seq):
    """Intervalos anotados consecutivos abaixo do limiar de visibilidade."""
    episodes = []
    for gid in np.unique(seq.gt[:, 1]).astype(int):
        rows = seq.gt[seq.gt[:, 1] == gid]
        rows = rows[np.argsort(rows[:, 0])]
        low = rows[rows[:, 8] < VISIBILITY_THRESHOLD, 0].astype(int)
        if len(low) == 0:
            continue
        start = previous = int(low[0])
        for frame in low[1:]:
            frame = int(frame)
            if frame != previous + 1:
                if previous - start + 1 >= 2:
                    episodes.append({"gt_id": int(gid), "start": start, "end": previous,
                                     "duration": previous - start + 1})
                start = frame
            previous = frame
        if previous - start + 1 >= 2:
            episodes.append({"gt_id": int(gid), "start": start, "end": previous,
                             "duration": previous - start + 1})
    return episodes


def assigned_id_at(pred_tracks, gt_track, frame, threshold=IOU_THRESHOLD):
    if frame not in gt_track:
        return None
    best_id, best_iou = None, threshold
    for pid, frames in pred_tracks.items():
        if frame in frames:
            iou = compute_iou(frames[frame], gt_track[frame])
            if iou >= best_iou:
                best_id, best_iou = pid, iou
    return best_id


def nearest_assigned_id(pred_tracks, gt_track, frame, direction, limit=10):
    for offset in range(limit + 1):
        candidate = frame + direction * offset
        pid = assigned_id_at(pred_tracks, gt_track, candidate)
        if pid is not None:
            return pid, candidate
    return None, None


def most_relevant_prediction_id(gt_track, predicted_history, start):
    """Escolhe a previsão mais próxima da trajetória GT no começo do intervalo."""
    candidate_frames = [
        f for f in range(max(1, start - 5), start + BASE_MAX_AGE + 1)
        if f in gt_track
    ]
    best_id, best_score = None, None
    for pid, frames in predicted_history.items():
        overlaps, distances = [], []
        for frame in candidate_frames:
            if frame not in frames:
                continue
            pred = frames[frame]
            gt = gt_track[frame]
            overlaps.append(compute_iou(pred, gt))
            pred_cx, pred_cy = (pred[1] + pred[3]) / 2, (pred[0] + pred[2]) / 2
            gt_cx, gt_cy = (gt[1] + gt[3]) / 2, (gt[0] + gt[2]) / 2
            scale = max(1.0, np.hypot(gt[3] - gt[1], gt[2] - gt[0]))
            distances.append(float(np.hypot(pred_cx - gt_cx, pred_cy - gt_cy) / scale))
        if not overlaps:
            continue
        score = (max(overlaps), -min(distances), len(overlaps))
        if best_score is None or score > best_score:
            best_id, best_score = pid, score
    return best_id


def analyze_occlusion_events(seq_id, seq, gt_tracks, tracker):
    events = []
    for episode in visibility_occlusions(seq):
        track = gt_tracks[episode["gt_id"]]
        before_id, before_frame = nearest_assigned_id(
            tracker.history, track, episode["start"] - 1, -1, limit=60
        )
        after_id, after_frame = nearest_assigned_id(
            tracker.history, track, episode["end"] + 1, 1, limit=10
        )
        reference_id = before_id
        if reference_id is None:
            reference_id = most_relevant_prediction_id(
                track, tracker.predicted_history, episode["start"]
            )
        predicted_boxes = tracker.predicted_history.get(reference_id, {}) if reference_id is not None else {}
        prediction_ious = [
            compute_iou(predicted_boxes[frame], track[frame])
            if frame in predicted_boxes and frame in track else 0.0
            for frame in range(episode["start"], episode["end"] + 1)
        ]
        state_available = 0
        for frame in range(episode["start"], episode["end"] + 1):
            if frame not in predicted_boxes:
                break
            state_available += 1
        episode.update({
            "sequence": seq_id, "before_pred_id": before_id, "before_frame": before_frame,
            "after_pred_id": after_id, "after_frame": after_frame,
            "reference_prediction_id": reference_id,
            "same_id_recovered": before_id is not None and before_id == after_id,
            "lost_or_switched": before_id is None or after_id != before_id,
            "mean_predicted_iou_during_occlusion": float(np.mean(prediction_ious)) if prediction_ious else 0.0,
            "state_available_frames": state_available,
        })
        events.append(episode)
    return events


def gradient_horizon(model, trajectories, device, max_samples=256):
    """Mede dL(final forecast)/dh[t-k] em janelas completas do treino free-running."""
    eligible = [traj for traj in trajectories if len(traj) >= WINDOW_SIZE + 1]
    if not eligible:
        raise ValueError(
            f"Não há trajetórias contínuas com pelo menos {WINDOW_SIZE + 1} quadros "
            "para estimar a curva do gradiente."
        )
    rng = np.random.default_rng(17)
    samples = []
    for _ in range(max_samples):
        traj = eligible[int(rng.integers(len(eligible)))]
        start = int(rng.integers(len(traj) - WINDOW_SIZE))
        samples.append((
            traj[start:start + WINDOW_SIZE],
            traj[start + 1:start + WINDOW_SIZE + 1],
        ))

    lag_values = [[] for _ in range(WINDOW_SIZE)]
    model.eval()
    for inputs_np, targets_np in samples:
        x = torch.as_tensor(inputs_np, dtype=torch.float32, device=device).unsqueeze(0)
        y = torch.as_tensor(targets_np, dtype=torch.float32, device=device).unsqueeze(0)
        model.zero_grad(set_to_none=True)
        hidden = None
        current_input = x[:, 0:1, :]
        hidden_steps = []
        final_prediction = None

        # Reproduz o free-running do treino: após o primeiro passo,
        # realimenta a previsão destacada do grafo, mantendo o estado recorrente.
        for step in range(WINDOW_SIZE):
            _, hidden = model.lstm(current_input, hidden)
            hidden_state = hidden[0]
            hidden_state.retain_grad()
            final_prediction = current_input + model.head(hidden_state[-1].unsqueeze(1))
            hidden_steps.append(hidden_state)
            if step < WINDOW_SIZE - 1:
                current_input = final_prediction.detach()

        loss = F.smooth_l1_loss(final_prediction, y[:, -1:, :])
        loss.backward()
        for lag in range(WINDOW_SIZE):
            grad = hidden_steps[WINDOW_SIZE - 1 - lag].grad[-1, 0]
            lag_values[lag].append(float(torch.linalg.vector_norm(grad).item()))

    means = [float(np.mean(values)) for values in lag_values]
    stds = [float(np.std(values)) for values in lag_values]
    medians = [float(np.median(values)) for values in lag_values]
    q25 = [float(np.quantile(values, 0.25)) for values in lag_values]
    q75 = [float(np.quantile(values, 0.75)) for values in lag_values]
    baseline = medians[0]
    ratios = [value / baseline if baseline > 0 else 0.0 for value in medians]
    return {
        "lags_frames": list(range(WINDOW_SIZE)),
        "mean_gradient_norm": means,
        "std_gradient_norm": stds,
        "median_gradient_norm": medians,
        "q25_gradient_norm": q25,
        "q75_gradient_norm": q75,
        "relative_to_lag0": ratios,
        "samples": len(samples),
        "samples_per_lag": [len(values) for values in lag_values],
        "window_size": WINDOW_SIZE,
    }

def stable_color(identity):
    return plt.get_cmap("tab20")(int(identity) % 20)


def gallery(event, seq, gt_tracks, tracker, destination):
    gid, start, end = event["gt_id"], event["start"], event["end"]
    candidates = [event.get("before_frame"), start, (start + end) // 2, end, event.get("after_frame")]
    frames = list(dict.fromkeys(int(f) for f in candidates if f is not None))
    if len(frames) < 2:
        frames = sorted(set([max(1, start - 1), start, end, min(len(seq), end + 1)]))
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV é necessário para carregar img1 e desenhar as galerias.") from exc

    fig = plt.figure(figsize=(15, 8))
    grid = fig.add_gridspec(2, len(frames), height_ratios=[3, 1.2])
    gt_track = gt_tracks[gid]
    for col, frame in enumerate(frames):
        image = cv2.imread(str(seq.frame_path(frame)))
        if image is None:
            raise FileNotFoundError(f"Quadro {seq.frame_path(frame)} ausente. Baixe o pacote MOT17 completo, com img1.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax = fig.add_subplot(grid[0, col])
        ax.imshow(image)
        if frame in gt_track:
            y0, x0, y1, x1 = gt_track[frame]
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   linewidth=2.5, edgecolor=stable_color(gid), linestyle="-"))
        for pid, track in tracker.history.items():
            if frame not in track:
                continue
            y0, x0, y1, x1 = track[frame]
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   linewidth=1.7, linestyle="--", edgecolor=stable_color(pid)))
        ax.set_title(f"quadro {frame}")
        ax.axis("off")

    # Mapa intermediário: previsão recorrente da track associada (ou da mais
    # próxima previsão disponível quando nenhuma identidade GT casou antes).
    map_ax = fig.add_subplot(grid[1, :])
    pid = event.get("reference_prediction_id")
    predicted = tracker.predicted_history.get(pid, {}) if pid is not None else {}
    xs, ys = [], []
    for frame, box in sorted(predicted.items()):
        if start - 3 <= frame <= min(end + 3, start + BASE_MAX_AGE + 3):
            y0, x0, y1, x1 = box
            xs.append((x0 + x1) / 2)
            ys.append((y0 + y1) / 2)
    if xs:
        map_ax.plot(xs, ys, "o--", color="#E4572E",
                    label=f"centro previsto pela LSTM (ID {pid})")
    gt_frames = [f for f in sorted(gt_track) if start - 3 <= f <= end + 3]
    if gt_frames:
        centers = [((gt_track[f][1] + gt_track[f][3]) / 2,
                    (gt_track[f][0] + gt_track[f][2]) / 2) for f in gt_frames]
        map_ax.plot([p[0] for p in centers], [p[1] for p in centers], "o-",
                    color=stable_color(gid), label=f"centro GT {gid}")
    map_ax.invert_yaxis()
    map_ax.set_xlabel("x (pixels)")
    map_ax.set_ylabel("y (pixels)")
    map_ax.set_title("Mapa intermediário: centros previstos e do ground truth")
    map_ax.legend(loc="best")
    state = "recuperou o mesmo ID" if event["same_id_recovered"] else "ID perdido ou trocado"
    fig.suptitle(
        f"MOT17-{event['sequence']} · GT {gid} · baixa visibilidade (<0,1) "
        f"por {event['duration']} quadros · {state}"
    )
    fig.tight_layout()
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_gradient_plot(curve, path):
    lag = np.asarray(curve["lags_frames"])
    median = np.asarray(curve["median_gradient_norm"])
    q25 = np.asarray(curve["q25_gradient_norm"])
    q75 = np.asarray(curve["q75_gradient_norm"])
    positive = median[median > 0]
    floor = max(float(positive.min()) * 0.5, 1e-16) if positive.size else 1e-16
    median = np.maximum(median, floor)
    q25 = np.maximum(q25, floor)
    q75 = np.maximum(q75, floor)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lag, median, marker="o", color="#128C94", label="mediana")
    ax.fill_between(lag, q25, q75, color="#128C94", alpha=.2, label="intervalo interquartil")
    ax.set_yscale("log")
    ax.set_xticks(lag)
    ax.set_xlabel("k: passos para trás a partir da previsão supervisionada")
    ax.set_ylabel(r"Norma de $\partial L_t / \partial h_{t-k}$")
    ax.set_title(
        f"Horizonte analítico de memória · free-running · janela {WINDOW_SIZE} "
        f"({curve['samples']} janelas completas)"
    )
    ax.grid(True, which="both", alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

def save_occlusion_plot(events, path):
    durations = [e["duration"] for e in events]
    state_lifetimes = [e["state_available_frames"] for e in events]
    fig, ax = plt.subplots(figsize=(8, 5))
    if durations:
        bins = np.arange(0, max(max(durations), max(state_lifetimes, default=0)) + 6, 5)
        ax.hist(durations, bins=bins, alpha=.5, label="intervalo GT com visibilidade < 0,1", color="#128C94")
        ax.hist(state_lifetimes, bins=bins, alpha=.55,
                label="quadros em que a track manteve estado ativo", color="#E4572E")
        recovered = [e["duration"] for e in events if e["same_id_recovered"]]
        lost = [e["duration"] for e in events if not e["same_id_recovered"]]
        if recovered:
            ax.hist(recovered, bins=bins, histtype="step", linewidth=2,
                    label="oclusão com mesmo ID recuperado", color="#2E8B57")
        if lost:
            ax.hist(lost, bins=bins, histtype="step", linewidth=2,
                    label="oclusão com ID perdido/trocado", color="#7A5195")
    ax.axvline(BASE_MAX_AGE, color="#555555", linestyle="--", label=f"max_age atual = {BASE_MAX_AGE}")
    ax.set_xlabel("Duração da oclusão (quadros)")
    ax.set_ylabel("Número de eventos")
    ax.set_title("Horizonte empírico: baixa visibilidade e sobrevivência da identidade")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def summarize_tracker(sequences, trackers, max_age):
    per_sequence = []
    for seq_id, seq, gt_tracks in sequences:
        tracker = trackers[seq_id]
        idf1, idtp, idfp, idfn, _ = compute_idf1(tracker.history, gt_tracks)
        per_sequence.append({"sequence": seq_id, "idf1": float(idf1), "idtp": int(idtp),
                             "idfp": int(idfp), "idfn": int(idfn),
                             "id_switches": count_id_switches(tracker.history, gt_tracks),
                             "fragmentations": count_fragmentations(tracker.history, gt_tracks),
                             "predicted_identities": len(tracker.history),
                             "ground_truth_identities": len(gt_tracks)})
    return {"max_age": max_age, "mean_idf1": float(np.mean([r["idf1"] for r in per_sequence])),
            "total_id_switches": int(sum(r["id_switches"] for r in per_sequence)),
            "total_fragmentations": int(sum(r["fragmentations"] for r in per_sequence)),
            "per_sequence": per_sequence}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/MOT17/train"))
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--max-samples", type=int, default=256,
                        help="janelas reais amostradas para estimar a curva do gradiente")
    args = parser.parse_args()

    missing = [str(p) for p in [args.checkpoint] if not p.exists()]
    missing.extend(str(args.data_root / f"MOT17-{sid}-{DETECTOR}") for sid in SEQUENCE_IDS
                   if not (args.data_root / f"MOT17-{sid}-{DETECTOR}" / "gt" / "gt.txt").exists())
    if missing:
        raise FileNotFoundError("Arquivos necessários ausentes:\n  " + "\n  ".join(missing) +
                                "\nExtraia o MOT17 train (com gt/det; img1 é necessário para as galerias).")

    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MotionLSTM(input_dim=4, hidden_dim=64).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()

    sequences, sequences_info = [], []
    trackers_by_age = {BASE_MAX_AGE: {}, CORRECTED_MAX_AGE: {}}
    all_events = []
    for seq_id in SEQUENCE_IDS:
        seq = MOT17Sequence(args.data_root / f"MOT17-{seq_id}-{DETECTOR}", load_gt=True)
        if seq.gt is None or len(seq.gt) == 0:
            continue
        gt_tracks = build_gt_tracks(seq)
        sequences.append((seq_id, seq, gt_tracks))
        sequences_info.append((seq.info.name, seq.gt, seq.info.im_width, seq.info.im_height))
        original = run_tracker(seq, model, device, BASE_MAX_AGE)
        corrected = run_tracker(seq, model, device, CORRECTED_MAX_AGE)
        trackers_by_age[BASE_MAX_AGE][seq_id] = original
        trackers_by_age[CORRECTED_MAX_AGE][seq_id] = corrected
        all_events.extend(analyze_occlusion_events(seq_id, seq, gt_tracks, original))
        print(f"MOT17-{seq_id}: avaliadas as duas políticas max_age={BASE_MAX_AGE}/{CORRECTED_MAX_AGE}")

    if not sequences:
        raise RuntimeError("Nenhuma sequência MOT17 com ground truth válido foi encontrada.")

    trajectories = extract_trajectories(sequences_info)
    gradient = gradient_horizon(model, trajectories, device, args.max_samples)
    save_gradient_plot(gradient, args.output / "gradient_horizon.png")
    save_occlusion_plot(all_events, args.output / "occlusion_survival.png")

    # Seleciona erros em oclusões, favorecendo ID perdido/trocado e trechos longos,
    # em sequências/IDs distintos para a galeria não repetir o mesmo caso.
    ranked = sorted(all_events,
                    key=lambda e: (e["before_pred_id"] is not None, e["lost_or_switched"],
                                   1.0 - e["mean_predicted_iou_during_occlusion"], e["duration"]),
                    reverse=True)
    selected, used = [], set()
    for event in ranked:
        key = (event["sequence"], event["gt_id"])
        if key in used:
            continue
        selected.append(event)
        used.add(key)
        if len(selected) == 3:
            break
    if len(selected) < 3:
        raise RuntimeError(f"Só foram encontrados {len(selected)} eventos de oclusão distintos; são necessários 3.")

    current_gallery_names = {
        f"failure_{index}_{event['sequence']}_gt{event['gt_id']}_f{event['start']}.png"
        for index, event in enumerate(selected, start=1)
    }
    for stale_gallery in args.output.glob("failure_*.png"):
        if stale_gallery.name not in current_gallery_names:
            stale_gallery.unlink()

    gallery_diagnostics = []
    seq_lookup = {sid: (seq, gt) for sid, seq, gt in sequences}
    for index, event in enumerate(selected, start=1):
        filename = f"failure_{index}_{event['sequence']}_gt{event['gt_id']}_f{event['start']}.png"
        gallery(event, *seq_lookup[event["sequence"]],
                trackers_by_age[BASE_MAX_AGE][event["sequence"]], args.output / filename)
        tail_ratio = gradient["relative_to_lag0"][-1]
        attenuation = (f" A mediana em k={WINDOW_SIZE - 1} é {tail_ratio:.4g} da mediana em k=0 "
                       f"(atenuação de {1.0 / tail_ratio:.1f}x)." if tail_ratio > 0 else
                       f" A mediana do gradiente em k={WINDOW_SIZE - 1} caiu abaixo de zero numérico.")
        gallery_diagnostics.append({**event, "figure": filename,
            "diagnosis": (f"O GT {event['gt_id']} ficou com visibilidade abaixo de {VISIBILITY_THRESHOLD:.2f} "
                          f"por {event['duration']} quadros; a janela de BPTT tem {WINDOW_SIZE} passos. "
                          f"A IoU média da caixa extrapolada nesse intervalo foi "
                          f"{event['mean_predicted_iou_during_occlusion']:.3f}. "
                          f"O mesmo ID foi {'recuperado' if event['same_id_recovered'] else 'perdido ou trocado'} "
                          f"após o intervalo. A medida analítica dá:{attenuation}")})

    summaries = {str(age): summarize_tracker(sequences, trackers_by_age[age], age)
                 for age in (BASE_MAX_AGE, CORRECTED_MAX_AGE)}
    result = {
        "model": str(args.checkpoint), "regime": "free_running", "seed": 0,
        "detector": DETECTOR, "bptt_window": WINDOW_SIZE,
        "analytical_horizon": gradient,
        "empirical_occlusion": {
            "visibility_threshold": VISIBILITY_THRESHOLD,
            "events_count": len(all_events),
            "duration_frames": [e["duration"] for e in all_events],
            "state_available_frames": [e["state_available_frames"] for e in all_events],
            "same_id_recovered": sum(e["same_id_recovered"] for e in all_events),
            "lost_or_switched": sum(e["lost_or_switched"] for e in all_events),
            "events": all_events,
        },
        "correction": {
            "hypothesis": "tracks morrem ao exceder max_age mesmo quando o objeto reaparece após oclusão longa",
            "change": f"max_age {BASE_MAX_AGE} -> {CORRECTED_MAX_AGE}",
            "before_after": summaries,
        },
        "failure_gallery": gallery_diagnostics,
        "figures": ["gradient_horizon.png", "occlusion_survival.png"] +
                   [item["figure"] for item in gallery_diagnostics],
    }
    with (args.output / "part4_memory_results.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
    print(f"Análise completa; resultados e figuras em {args.output}")
    print(f"IDF1 médio max_age={BASE_MAX_AGE}: {summaries[str(BASE_MAX_AGE)]['mean_idf1']:.4f}")
    print(f"IDF1 médio max_age={CORRECTED_MAX_AGE}: {summaries[str(CORRECTED_MAX_AGE)]['mean_idf1']:.4f}")


if __name__ == "__main__":
    main()
