# scripts/visualize_part2_cases.py
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from src.datasets.mot17 import MOT17Sequence
from src.tracking.naive_tracker import NaiveTracker
from src.tracking.motion_lstm_tracker import MotionLSTMTracker
from src.models.motion_lstm import MotionLSTM
from src.analysis.find_critical_moments import (
    find_recovery_case, find_lock_in_failure_case, assigned_pred_id_per_frame,
)

Path("outputs").mkdir(exist_ok=True)

DETECTOR = "SDP"
COLORS = plt.cm.tab20(np.linspace(0, 1, 20))


def left_top_wh_to_yxyx(box):
    left, top, w, h = box
    return (top, left, top + h, left + w)


def build_gt_tracks(seq):
    gt_ids = np.unique(seq.gt[:, 1]).astype(int)
    return {
        gid: {int(r[0]): left_top_wh_to_yxyx((r[2], r[3], r[4], r[5]))
              for r in seq.gt[seq.gt[:, 1] == gid]}
        for gid in gt_ids
    }


def run_both_trackers(seq, model, device):
    detections_pixel = [seq.detections_at(t) for t in range(1, seq.info.seq_length + 1)]
    detections_yxyx = [
        np.array([left_top_wh_to_yxyx(d) for d in dets]) if len(dets) else np.zeros((0, 4))
        for dets in detections_pixel
    ]

    naive = NaiveTracker(iou_threshold=0.3, max_age=30)
    pred_naive = naive.run(detections_yxyx)
    pred_naive = {tid: {f + 1: box for f, box in frames.items()} for tid, frames in pred_naive.items()}

    trilha_a = MotionLSTMTracker(model, device, seq.info.im_width, seq.info.im_height,
                                   iou_threshold=0.3, max_age=30)
    pred_a = trilha_a.run(detections_pixel)
    pred_a = {tid: {f + 1: box for f, box in frames.items()} for tid, frames in pred_a.items()}

    return pred_naive, pred_a


def build_local_color_map(assign_naive, assign_a):
    """Mapeia cada ID distinto que aparece NESSA figura para uma cor
    única, evitando colisão de módulo entre IDs de trackers diferentes
    que não têm relação nenhuma entre si (ex.: ID=1 do baseline e
    ID=81 da Trilha A não devem compartilhar cor só por coincidência
    de 81 % 20 == 1 % 20)."""
    all_ids = sorted(set(
        [pid for pid in assign_naive.values() if pid is not None] +
        [pid for pid in assign_a.values() if pid is not None]
    ))
    return {pid: COLORS[i % 20] for i, pid in enumerate(all_ids)}


def draw_case(seq_path, seq, gt_track, assign_naive, assign_a, center_frame,
              window=6, out_path="outputs/case.png"):
    color_map = build_local_color_map(assign_naive, assign_a)
    gray = (0.5, 0.5, 0.5, 1.0)

    frames_to_show = list(range(max(1, center_frame - window), center_frame + window + 1, 2))
    n = len(frames_to_show)

    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))

    for col, f in enumerate(frames_to_show):
        img_path = seq_path / "img1" / f"{f:06d}.jpg"
        if not img_path.exists():
            continue
        img = Image.open(img_path)

        for row, (label, assignment) in enumerate([("Baseline", assign_naive), ("Trilha A", assign_a)]):
            ax = axes[row, col]
            ax.imshow(img)
            if f in gt_track:
                y0, x0, y1, x1 = gt_track[f]
                pid = assignment.get(f)
                color = color_map.get(pid, gray)
                rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                      edgecolor=color, linewidth=3)
                ax.add_patch(rect)
                ax.set_title(f"t={f}  ID={pid}", fontsize=9, color=color)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(label, fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    print(f"Salvo em {out_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MotionLSTM(input_dim=4, hidden_dim=64)
    model.load_state_dict(torch.load("outputs/motion_lstm.pt", map_location=device))

    root = Path("data/raw/MOT17/train")

    # --- caso de sucesso: busca em várias sequências ---
    case_success = None
    success_seq_id = None

    for candidate_seq_id in ["09", "10", "13", "02", "04", "05", "11"]:
        seq_path = root / f"MOT17-{candidate_seq_id}-{DETECTOR}"
        seq = MOT17Sequence(seq_path, load_gt=True)
        if seq.gt is None:
            continue
        gt_tracks = build_gt_tracks(seq)
        pred_naive, pred_a = run_both_trackers(seq, model, device)

        case_success = find_recovery_case(pred_naive, pred_a, gt_tracks)
        if case_success:
            success_seq_id = candidate_seq_id
            success_seq_path, success_seq = seq_path, seq
            success_gt_tracks = gt_tracks
            print(f"Caso de sucesso encontrado em MOT17-{success_seq_id}")
            break

    if case_success:
        gid, event_frame, _, improvement = case_success
        assign_naive = assigned_pred_id_per_frame(pred_naive, success_gt_tracks[gid])
        assign_a = assigned_pred_id_per_frame(pred_a, success_gt_tracks[gid])
        print(f"  gt_id={gid}, evento no quadro {event_frame}, melhora={improvement} switches a menos")
        draw_case(success_seq_path, success_seq, success_gt_tracks[gid], assign_naive, assign_a,
                  event_frame, out_path="outputs/part2_case_success.png")
    else:
        print("Nenhum caso de sucesso claro encontrado em nenhuma sequência.")

    # --- caso de falha: MOT17-04 ---
    seq_path_04 = root / f"MOT17-04-{DETECTOR}"
    seq_04 = MOT17Sequence(seq_path_04, load_gt=True)
    gt_tracks_04 = build_gt_tracks(seq_04)
    pred_naive_04, pred_a_04 = run_both_trackers(seq_04, model, device)

    case_failure = find_lock_in_failure_case(pred_naive_04, pred_a_04, gt_tracks_04)
    if case_failure:
        gid, event_frame, _ = case_failure
        assign_naive = assigned_pred_id_per_frame(pred_naive_04, gt_tracks_04[gid])
        assign_a = assigned_pred_id_per_frame(pred_a_04, gt_tracks_04[gid])
        print(f"Caso de falha: MOT17-04, gt_id={gid}, evento no quadro {event_frame}")
        draw_case(seq_path_04, seq_04, gt_tracks_04[gid], assign_naive, assign_a,
                  event_frame, out_path="outputs/part2_case_failure.png")
    else:
        print("Nenhum caso de falha claro encontrado em MOT17-04.")


if __name__ == "__main__":
    main()