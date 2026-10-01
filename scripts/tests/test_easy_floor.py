# scripts/test_easy_floor.py
import numpy as np
from src.datasets.synthetic_video import generate_video
from src.tracking.naive_tracker import NaiveTracker
from src.metrics.tracking_metrics import compute_idf1, count_id_switches

# piso fácil: poucos objetos, lentos, SEM oclusão forçada
# (usamos n_objects baixo o suficiente pra colisão de trajetória ser rara)
rng = np.random.default_rng(1)
frames, boxes, visibility, on_screen, objects = generate_video(
    rng, n_frames=30, n_objects=3, typical_speed=0.5, occlusion_duration=0
)

n_frames = len(frames)
detections_by_frame = []
for t in range(n_frames):
    dets = [boxes[obj_id][t] for obj_id in boxes if boxes[obj_id][t] is not None]
    detections_by_frame.append(np.array(dets) if dets else np.zeros((0, 4)))

tracker = NaiveTracker(iou_threshold=0.3, max_age=5)
pred_tracks = tracker.run(detections_by_frame)

gt_tracks = {}
for obj_id in boxes:
    gt_tracks[obj_id] = {t: boxes[obj_id][t] for t in range(n_frames) if boxes[obj_id][t] is not None}

idf1, tp, fp, fn, mapping = compute_idf1(pred_tracks, gt_tracks)
switches = count_id_switches(pred_tracks, gt_tracks)

print(f"Piso fácil: {len(gt_tracks)} objetos reais, {len(pred_tracks)} tracks previstas")
print(f"IDF1={idf1:.3f} (esperado próximo de 1.0), switches={switches} (esperado 0 ou muito baixo)")