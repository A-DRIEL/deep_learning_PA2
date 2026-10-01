# scripts/test_tracking_metrics.py
from src.metrics.tracking_metrics import compute_idf1, count_id_switches, count_fragmentations

def make_track(id_, boxes_by_frame):
    return {f: b for f, b in boxes_by_frame.items()}

# --- Caso (a) ---
gt = {1: make_track(1, {0: (0, 0, 10, 10), 1: (1, 1, 11, 11), 2: (2, 2, 12, 12)})}
pred_a = {1: make_track(1, {0: (0, 0, 10, 10), 1: (1, 1, 11, 11), 2: (2, 2, 12, 12)})}
idf1_a, tp, fp, fn, mapping_a = compute_idf1(pred_a, gt)
switches_a = count_id_switches(pred_a, gt)  # <- sem mapping_a
print(f"(a) pred=gt exato: IDF1={idf1_a:.3f} (esperado 1.0), switches={switches_a} (esperado 0)")

# --- Caso (b) ---
gt_b = {
    1: {0: (0, 0, 10, 10), 1: (1, 1, 11, 11), 2: (2, 2, 12, 12), 3: (3, 3, 13, 13)},
    2: {0: (50, 50, 60, 60), 1: (51, 51, 61, 61), 2: (52, 52, 62, 62), 3: (53, 53, 63, 63)},
}
pred_b = {
    10: {0: (0, 0, 10, 10), 1: (1, 1, 11, 11), 2: (52, 52, 62, 62), 3: (53, 53, 63, 63)},
    20: {0: (50, 50, 60, 60), 1: (51, 51, 61, 61), 2: (2, 2, 12, 12), 3: (3, 3, 13, 13)},
}
idf1_b, tp_b, fp_b, fn_b, mapping_b = compute_idf1(pred_b, gt_b)
switches_b = count_id_switches(pred_b, gt_b)  # <- sem mapping_b
print(f"(b) troca no quadro k=2: IDF1={idf1_b:.3f}, switches={switches_b} (esperado 2)")

# --- Caso (c), com lacuna real ---
gt_c = {1: {0: (0,0,10,10), 1: (1,1,11,11), 2: (2,2,12,12), 3: (3,3,13,13), 4: (4,4,14,14)}}
pred_c = {
    100: {0: (0,0,10,10), 1: (1,1,11,11)},
    200: {3: (3,3,13,13), 4: (4,4,14,14)},
}
idf1_c, tp_c, fp_c, fn_c, mapping_c = compute_idf1(pred_c, gt_c)
switches_c = count_id_switches(pred_c, gt_c)
frags_c = count_fragmentations(pred_c, gt_c)
print(f"(c) track partida com lacuna: IDF1={idf1_c:.3f}, switches={switches_c} (esperado 0), fragmentações={frags_c} (esperado 1)")