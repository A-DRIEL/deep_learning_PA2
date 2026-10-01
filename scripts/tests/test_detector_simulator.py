# scripts/test_detector_simulator.py
import numpy as np
from src.datasets.synthetic_video import generate_video
from src.datasets.detector_simulator import simulate_detections

rng = np.random.default_rng(0)
frames, boxes, visibility, on_screen, objects = generate_video(rng, n_frames=20, n_objects=6)

frame_t = 5
true_boxes_t = [boxes[obj_id][frame_t] for obj_id in boxes]
n_real = sum(1 for b in true_boxes_t if b is not None)

detected = simulate_detections(true_boxes_t, rng, drop_prob=0.2,
                                 noise_std=1.5, false_positive_rate=0.3)

print(f"Quadro {frame_t}: {n_real} objetos reais na tela, {len(detected)} detecções simuladas")
print(detected)