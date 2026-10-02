# scripts/tests/test_part5_components.py
"""
Testes de sanidade (sem MOT17, sem GPU) do degradador e do mAP:
  1. GT usado como detecção (conf=1)  =>  mAP = 1
  2. cfg "limpo" não altera nada
  3. drop_prob=0.5 mantém ~50% das caixas
  4. ruído pequeno: IoU médio com o original alto mas < 1
  5. FP: média de FPs por quadro ~ fp_ratio * n_original
  6. mAP decresce monotonicamente: limpo >= leve >= media >= pesada
"""
import numpy as np

from src.datasets.detector_degradation import (
    INTENSITIES, DegradationConfig, degrade_sequence, split_by_frame,
)
from src.metrics.detection_metrics import detection_map, iou_matrix_xywh

rng = np.random.default_rng(0)
N_FRAMES, N_OBJ, W, H = 60, 20, 1920, 1080

# --- cena sintética no formato det.txt/gt.txt: frame,id,l,t,w,h,conf,x,y,z ---
rows = []
for t in range(1, N_FRAMES + 1):
    for k in range(N_OBJ):
        w, h = 40 + 3 * k, 100 + 6 * k
        left = 50 + 80 * k + 2 * t
        top = 200 + 15 * k
        rows.append([t, k, left, top, w, h, rng.uniform(0.3, 1.0), -1, -1, -1])
det = np.array(rows, dtype=np.float64)
gt_frames = [g[:, :4] for g in split_by_frame(det, N_FRAMES)]

# 1) GT como detecção, conf = 1
perfect = [np.concatenate([g, np.ones((len(g), 1))], axis=1) for g in gt_frames]
r = detection_map(perfect, gt_frames)
print(f"[1] mAP(GT como detecção) = {r['map']:.4f} (esperado 1.0)")
assert abs(r["map"] - 1.0) < 1e-9

# 2) limpo não altera
clean = degrade_sequence(det, N_FRAMES, INTENSITIES["limpo"], 0, W, H)
orig = split_by_frame(det, N_FRAMES)
assert all(np.allclose(a, b) for a, b in zip(clean, orig))
print("[2] cfg limpo: detecções idênticas às originais")

# 3) descarte
cfg = DegradationConfig("t", drop_prob=0.5)
out = degrade_sequence(det, N_FRAMES, cfg, 0, W, H)
frac = sum(len(o) for o in out) / sum(len(o) for o in orig)
print(f"[3] fração mantida com drop=0.5: {frac:.3f} (esperado ~0.5)")
assert 0.43 < frac < 0.57

# 4) ruído
cfg = DegradationConfig("t", noise_rel=0.05)
out = degrade_sequence(det, N_FRAMES, cfg, 0, W, H)
ious = [np.diag(iou_matrix_xywh(o[:, :4], g[:, :4])).mean() for o, g in zip(out, orig)]
print(f"[4] IoU médio com ruído 0.05: {np.mean(ious):.3f} (esperado entre 0.7 e 1.0)")
assert 0.7 < np.mean(ious) < 1.0

# 5) falsos positivos
cfg = DegradationConfig("t", fp_ratio=0.3)
out = degrade_sequence(det, N_FRAMES, cfg, 0, W, H)
mean_fp = np.mean([len(o) - N_OBJ for o in out])
print(f"[5] FPs/quadro com fp_ratio=0.3: {mean_fp:.2f} (esperado ~{0.3 * N_OBJ:.1f})")
assert 0.7 * 0.3 * N_OBJ < mean_fp < 1.3 * 0.3 * N_OBJ

# 6) monotonicidade do mAP
maps = []
for name in ["limpo", "leve", "media", "pesada"]:
    dets = degrade_sequence(det, N_FRAMES, INTENSITIES[name], 0, W, H)
    maps.append(detection_map(dets, gt_frames)["map"])
    print(f"[6] mAP {name:7s} = {maps[-1]:.3f}")
assert all(maps[i] >= maps[i + 1] for i in range(3))
print("\nTodos os testes passaram.")
