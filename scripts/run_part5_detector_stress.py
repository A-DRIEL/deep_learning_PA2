# scripts/run_part5_detector_stress.py
"""
Parte 5 (opção: qualidade do detector). SEM RETREINAR.

Para cada sequência do MOT17, cada intensidade de degradação
(limpo / leve / media / pesada, + opcionalmente cada falha isolada) e
cada seed de degradação:
  1. degrada det.txt (descarte + ruído + falsos positivos)
  2. mede mAP da detecção degradada vs GT
  3. roda baseline (Parte 1) e Trilha A (Parte 2) sobre as detecções
     degradadas e mede IDF1, switches, fragmentações, razão de contagem
Grava tudo (uma linha por sequência x config x seed x tracker) em JSON.
O JSON é incremental: se o script for interrompido, rodar de novo retoma.

Uso:
  python -m scripts.run_part5_detector_stress \
      --checkpoint outputs/motion_lstm.pt --seeds 0 1 2 --heldout 09
"""

import argparse
import json
from pathlib import Path

import torch

from src.datasets.detector_degradation import (
    INTENSITIES, ISOLATED, degrade_sequence, split_by_frame,
)
from src.datasets.mot17 import MOT17Sequence, SEQUENCE_IDS
from src.metrics.detection_metrics import detection_map
from src.metrics.tracking_metrics import (
    compute_idf1, count_fragmentations, count_id_switches,
)
from src.models.motion_lstm import MotionLSTM
from src.tracking.runners import build_gt_tracks, run_naive, run_trilha_a


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="outputs/motion_lstm.pt",
                   help="pesos do modelo FINAL da Trilha A (não é retreinado)")
    p.add_argument("--detector", default="SDP")
    p.add_argument("--root", default="data/raw/MOT17/train")
    p.add_argument("--sequences", nargs="+", default=SEQUENCE_IDS)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
                   help="seeds da DEGRADAÇÃO (o modelo é fixo)")
    p.add_argument("--heldout", nargs="*", default=[],
                   help="ids de sequências nunca vistas no treino (ex.: 09)")
    p.add_argument("--isolate", action="store_true",
                   help="roda também cada falha isolada (intensidade média)")
    p.add_argument("--out", default="outputs/part5_detector_stress.json")
    p.add_argument("--restart", action="store_true", help="ignora resultados anteriores")
    return p.parse_args()


def evaluate(seq, gt_tracks, gt_frames, dets_by_frame, model, device):
    """Devolve uma lista de dicts (um por tracker) com todas as métricas."""
    det_res = detection_map(dets_by_frame, gt_frames)
    n_dets = int(sum(len(d) for d in dets_by_frame))
    n_gt_ids = len(gt_tracks)

    preds = {
        "baseline": run_naive(dets_by_frame),
        "trilha_a": run_trilha_a(dets_by_frame, model, device,
                                 seq.info.im_width, seq.info.im_height),
    }

    rows = []
    for tracker_name, pred in preds.items():
        idf1, *_ = compute_idf1(pred, gt_tracks)
        switches = count_id_switches(pred, gt_tracks)
        frags = count_fragmentations(pred, gt_tracks)
        n_pred_ids = len(pred)
        rows.append({
            "tracker": tracker_name,
            "map": det_res["map"], "ap50": det_res["ap50"], "n_dets": n_dets,
            "idf1": float(idf1), "switches": int(switches), "frags": int(frags),
            "n_gt_ids": n_gt_ids, "n_pred_ids": n_pred_ids,
            "count_ratio": n_pred_ids / n_gt_ids,
            "switches_per_id": switches / n_gt_ids,
        })
    return rows


def save(path, meta, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"meta": meta, "rows": rows}, f, indent=1)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = MotionLSTM(input_dim=4, hidden_dim=64)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device).eval()

    configs = dict(INTENSITIES)
    if args.isolate:
        configs.update(ISOLATED)

    meta = {"checkpoint": args.checkpoint, "detector": args.detector,
            "seeds": args.seeds, "heldout": args.heldout,
            "configs": {k: vars(v) for k, v in configs.items()}}

    rows, done = [], set()
    out_path = Path(args.out)
    if out_path.exists() and not args.restart:
        rows = json.load(open(out_path))["rows"]
        done = {(r["seq_id"], r["config"], r["seed"]) for r in rows}
        print(f"Retomando: {len(done)} (seq, config, seed) já feitos")

    root = Path(args.root)
    for seq_id in args.sequences:
        seq = MOT17Sequence(root / f"MOT17-{seq_id}-{args.detector}", load_gt=True)
        n = seq.info.seq_length
        gt_tracks = build_gt_tracks(seq)
        gt_frames = [g[:, :4] for g in split_by_frame(seq.gt, n)]

        for cname, cfg in configs.items():
            seeds = [0] if cname == "limpo" else args.seeds  # limpo é determinístico
            for seed in seeds:
                if (seq_id, cname, seed) in done:
                    continue
                dets = degrade_sequence(seq.det, n, cfg, seed,
                                        seq.info.im_width, seq.info.im_height)
                new_rows = evaluate(seq, gt_tracks, gt_frames, dets, model, device)
                for r in new_rows:
                    r.update(seq_id=seq_id, config=cname, seed=seed,
                             heldout=seq_id in args.heldout)
                    rows.append(r)
                save(out_path, meta, rows)
                b, a = new_rows
                print(f"MOT17-{seq_id} {cname:11s} seed={seed}: mAP={b['map']:.3f} | "
                      f"IDF1 base={b['idf1']:.3f} A={a['idf1']:.3f} | "
                      f"ratio base={b['count_ratio']:.2f} A={a['count_ratio']:.2f}")

    print(f"\nResultados em {out_path}")


if __name__ == "__main__":
    main()
