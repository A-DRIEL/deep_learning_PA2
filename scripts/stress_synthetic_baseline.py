"""
Parte 0, passo final: gira os botões do gerador sintético e mostra onde
o tracker ingênuo (baseline) começa a quebrar. Esse gráfico é o "ensaio"
da Parte 1 (mesma lógica, aplicado depois no MOT17 real).
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.datasets.synthetic_video import generate_video
from src.tracking.naive_tracker import NaiveTracker
from src.metrics.tracking_metrics import compute_idf1, count_id_switches, compute_iou
from src.datasets.detector_simulator import simulate_detections

Path("outputs").mkdir(exist_ok=True)


def run_once(seed, n_objects, typical_speed, occlusion_duration, n_frames=40,
             visibility_threshold=0.3):
    rng = np.random.default_rng(seed)
    frames, boxes, visibility, on_screen, objects = generate_video(
        rng, n_frames=n_frames, n_objects=n_objects,
        typical_speed=typical_speed, occlusion_duration=occlusion_duration,
    )

    detections_by_frame = []
    for t in range(n_frames):
        # só inclui a caixa se o objeto estiver suficientemente visível --
        # um detector real não enxerga o que está quase totalmente coberto
        true_boxes_t = [
            boxes[oid][t] for oid in boxes
            if boxes[oid][t] is not None and visibility[oid][t] >= visibility_threshold
        ]
        dets = simulate_detections(true_boxes_t, rng, drop_prob=0.05,
                                     noise_std=1.0, false_positive_rate=0.05)
        detections_by_frame.append(dets)

    tracker = NaiveTracker(iou_threshold=0.3, max_age=5)
    pred_tracks = tracker.run(detections_by_frame)

    gt_tracks = {
        oid: {t: boxes[oid][t] for t in range(n_frames) if boxes[oid][t] is not None}
        for oid in boxes
    }

    idf1, tp, fp, fn, mapping = compute_idf1(pred_tracks, gt_tracks)
    switches = count_id_switches(pred_tracks, gt_tracks)
    return idf1, switches

def sweep(param_name, values, n_seeds=3, **fixed_kwargs):
    idf1_means, idf1_stds = [], []
    for v in values:
        kwargs = {**fixed_kwargs, param_name: v}
        results = [run_once(seed=s, **kwargs) for s in range(n_seeds)]
        idf1s = [r[0] for r in results]
        idf1_means.append(np.mean(idf1s))
        idf1_stds.append(np.std(idf1s))
    return np.array(idf1_means), np.array(idf1_stds)


# --- eixo 1: número de objetos (densidade) ---
n_objects_values = [3, 6, 9, 12, 15]
idf1_density, std_density = sweep(
    "n_objects", n_objects_values,
    typical_speed=3.5, occlusion_duration=0,  
)

# --- eixo 2: velocidade ---
speed_values = [0.5, 1.5, 3.0, 5.0, 8.0]
idf1_speed, std_speed = sweep(
    "typical_speed", speed_values,
    n_objects=6, occlusion_duration=0,
)

# --- eixo 3: duração de oclusão ---
def filter_relevant_tracks(pred_tracks, gt_track, iou_threshold=0.01):
    """
    Mantém só as tracks previstas que, em ALGUM quadro, têm alguma
    sobreposição com o objeto de interesse -- descarta tracks de outros
    objetos da cena que nunca interagem com ele.
    """
    relevant = {}
    for tid, frames in pred_tracks.items():
        for f, box in frames.items():
            if f in gt_track and compute_iou(box, gt_track[f]) > iou_threshold:
                relevant[tid] = frames
                break
    return relevant


def run_once_occlusion(seed, occlusion_duration, n_frames=40):
    rng = np.random.default_rng(seed)
    frames, boxes, visibility, on_screen, objects = generate_video(
        rng, n_frames=n_frames, n_objects=2,
        typical_speed=1.0, occlusion_duration=occlusion_duration,
    )
    detections_by_frame = []
    for t in range(n_frames):
        true_boxes_t = [
            boxes[oid][t] for oid in boxes
            if boxes[oid][t] is not None and visibility[oid][t] >= 0.3
        ]
        dets = simulate_detections(true_boxes_t, rng, drop_prob=0.05,
                                     noise_std=1.0, false_positive_rate=0.05)
        detections_by_frame.append(dets)

    tracker = NaiveTracker(iou_threshold=0.3, max_age=5)
    pred_tracks = tracker.run(detections_by_frame)

    gt_track_0 = {t: boxes[0][t] for t in range(n_frames) if boxes[0][t] is not None}
    relevant_pred_tracks = filter_relevant_tracks(pred_tracks, gt_track_0)

    idf1, tp, fp, fn, mapping = compute_idf1(relevant_pred_tracks, {0: gt_track_0})
    return idf1


occlusion_values = [0, 4, 8, 15, 25]
idf1_occlusion = []
std_occlusion = []
for occ in occlusion_values:
    results = [run_once_occlusion(seed=s, occlusion_duration=occ) for s in range(3)]
    idf1_occlusion.append(np.mean(results))
    std_occlusion.append(np.std(results))
idf1_occlusion = np.array(idf1_occlusion)
std_occlusion = np.array(std_occlusion)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

axes[0].errorbar(n_objects_values, idf1_density, yerr=std_density, marker="o", color="#E4572E")
axes[0].set_xlabel("Número de objetos")
axes[0].set_ylabel("IDF1")
axes[0].set_title("Densidade")
axes[0].set_ylim(0, 1.05)

axes[1].errorbar(speed_values, idf1_speed, yerr=std_speed, marker="o", color="#128C94")
axes[1].set_xlabel("Velocidade típica (px/quadro)")
axes[1].set_title("Velocidade")
axes[1].set_ylim(0, 1.05)

axes[2].errorbar(occlusion_values, idf1_occlusion, yerr=std_occlusion, marker="o", color="#2E8B57")
axes[2].set_xlabel("Duração da oclusão (quadros)")
axes[2].set_title("Oclusão")
axes[2].set_ylim(0, 1.05)

for ax in axes:
    ax.grid(alpha=0.3)

plt.suptitle("Onde o baseline (associação ingênua por IoU) começa a quebrar")
plt.tight_layout()
plt.savefig("outputs/synthetic_stress_baseline.png", dpi=150)

print("Densidade:", list(zip(n_objects_values, idf1_density.round(3))))
print("Velocidade:", list(zip(speed_values, idf1_speed.round(3))))
print("Oclusão:", list(zip(occlusion_values, idf1_occlusion.round(3))))