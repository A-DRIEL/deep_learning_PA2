"""Parte 4 -- correção: implementa a mudança que o diagnóstico de GRADIENTE sugere.

Diagnóstico: o BPTT de treino (8 quadros, todos observados) nunca expõe o LSTM a
um buraco de observação, e o gradiente decai rápido demais para cruzá-lo.
Mudança: treinar com janela longa (T=32) e "oclusões simuladas" -- num trecho
aleatório da janela as entradas deixam de ser as observações e passam a ser as
próprias previsões do modelo, enquanto a perda continua supervisionando o GT.

Antes/depois, com CONTROLES para não atribuir ao LSTM o que é efeito de max_age:
    naive            max_age 30 / 60   (Parte 1)
    lstm_before      max_age 30 / 60   (checkpoint free-running original)
    lstm_gap_aug     max_age 30 / 60   (modelo corrigido)
Saídas em outputs/part4/: correction_results.json, correction_gradient.png,
correction_summary.png e uma tabela markdown no terminal.

ATENÇÃO (split): ambos os modelos são treinados nas mesmas 7 sequências, para a
comparação ser justa entre si. Para afirmar generalização, retreine os dois
deixando uma sequência inteira de fora (ver --holdout).
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.analysis.memory_horizon import (
    build_gt_tracks, collect_events, evaluate_history, gradient_horizon, overall_recovery,
    run_lstm_tracker, run_naive_tracker, summarize_gradient_curve,
)
from src.datasets.mot17 import MOT17Sequence, SEQUENCE_IDS
from src.datasets.trajectory_dataset import extract_trajectories
from src.models.motion_lstm import MotionLSTM
from src.viz.plot_style import PALETTE, apply_style

apply_style()
DETECTOR = "SDP"


class StridedWindowDataset(Dataset):
    def __init__(self, trajectories, window, stride):
        self.windows = [traj[s:s + window + 1]
                        for traj in trajectories for s in range(0, len(traj) - window, stride)]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        return torch.from_numpy(self.windows[i])      # (window+1, 4)


def sample_gap_mask(batch, window, p_gap, gap_min, gap_max):
    """mask[b, t] = True -> a entrada do passo t é a PRÓPRIA previsão (oclusão simulada)."""
    mask = torch.zeros(batch, window, dtype=torch.bool)
    gap_max = min(gap_max, window - 3)
    for b in range(batch):
        if random.random() < p_gap:
            g = random.randint(gap_min, gap_max)
            s = random.randint(1, window - g - 1)
            mask[b, s:s + g] = True
    return mask


def train_gap_model(trajectories, device, window, stride, epochs, seed, p_gap=0.7, gap_min=4, gap_max=24):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    data = StridedWindowDataset(trajectories, window, stride)
    loader = DataLoader(data, batch_size=128, shuffle=True, drop_last=True)
    model = MotionLSTM(input_dim=4, hidden_dim=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    print(f"Treino com oclusões simuladas: {len(data)} janelas de {window + 1} quadros, {epochs} épocas")
    for epoch in range(epochs):
        model.train()
        total, n = 0.0, 0
        for batch in loader:
            batch = batch.to(device)
            inputs, targets = batch[:, :-1], batch[:, 1:]
            mask = sample_gap_mask(batch.size(0), window, p_gap, gap_min, gap_max).to(device)
            opt.zero_grad()
            hidden, inp, loss = None, inputs[:, 0], 0.0
            for t in range(window):
                out, hidden = model.lstm(inp.unsqueeze(1), hidden)
                pred = inp + model.head(out[:, 0])
                loss = loss + F.smooth_l1_loss(pred, targets[:, t])
                if t < window - 1:
                    inp = torch.where(mask[:, t + 1].unsqueeze(1), pred.detach(), inputs[:, t + 1])
            loss = loss / window
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * batch.size(0)
            n += batch.size(0)
        print(f"  época {epoch + 1}/{epochs}: loss={total / n:.6f}")
    return model


def load_model(path, device):
    m = MotionLSTM(input_dim=4, hidden_dim=64).to(device)
    m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return m.eval()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=Path("data/raw/MOT17/train"))
    ap.add_argument("--before", type=Path, default=Path("outputs/ablation_checkpoints/free_running_seed0.pt"))
    ap.add_argument("--output", type=Path, default=Path("outputs/part4"))
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--holdout", nargs="*", default=[],
                    help="IDs de sequência (ex.: 09 13) excluídos do TREINO do modelo corrigido e "
                         "usados na avaliação. Só é limpo se o modelo 'before' também for retreinado sem elas.")
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seqs, info_train = {}, []
    for sid in SEQUENCE_IDS:
        seq = MOT17Sequence(args.data_root / f"MOT17-{sid}-{DETECTOR}", load_gt=True)
        if seq.gt is None or len(seq.gt) == 0:
            continue
        seqs[sid] = (seq, build_gt_tracks(seq))
        if sid not in args.holdout:
            info_train.append((seq.info.name, seq.gt, seq.info.im_width, seq.info.im_height))
    eval_ids = args.holdout or list(seqs)
    trajectories = extract_trajectories(info_train)

    after_path = args.output / f"motion_lstm_gap_T{args.window}.pt"
    if args.retrain or not after_path.exists():
        model_after = train_gap_model(trajectories, device, args.window, args.stride, args.epochs, args.seed)
        torch.save(model_after.state_dict(), after_path)
    model_after = load_model(after_path, device)
    model_before = load_model(args.before, device)

    # --- 1) gradiente analítico, mesma janela longa para os dois ---
    curves = {name: gradient_horizon(m, trajectories, device, args.window, 256)
              for name, m in (("before", model_before), ("after", model_after))}
    grad_summary = {name: summarize_gradient_curve(c, 8 if name == "before" else args.window)
                    for name, c in curves.items()}
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, color, label in (("before", PALETTE["coral"], "antes (BPTT 8, sem oclusão simulada)"),
                               ("after", PALETTE["teal"], f"depois (BPTT {args.window} + oclusão simulada)")):
        ax.plot(curves[name]["lags_frames"], np.maximum(curves[name]["relative_to_lag0"], 1e-30),
                "o-", markersize=3, color=color, label=label)
    ax.axvline(7, color="#555555", linestyle=":", label="limite do BPTT original")
    ax.set_yscale("log")
    ax.set_xlabel("k (passos para trás)")
    ax.set_ylabel(r"$\|\partial L_t/\partial h_{t-k}\|$ relativo a k=0")
    ax.set_title("Correção: o gradiente atravessa mais quadros?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / "correction_gradient.png", dpi=160)
    plt.close(fig)

    # --- 2) tracker: antes/depois + controles de max_age ---
    configs = [("naive", None, 30), ("naive", None, 60),
               ("lstm_before", model_before, 30), ("lstm_before", model_before, 60),
               ("lstm_gap_aug", model_after, 30), ("lstm_gap_aug", model_after, 60)]
    results = []
    for name, model, max_age in configs:
        per_seq, events = [], []
        for sid in eval_ids:
            seq, gt_tracks = seqs[sid]
            if model is None:
                history = run_naive_tracker(seq, max_age)
            else:
                tracker = run_lstm_tracker(seq, model, device, max_age)
                history = tracker.history
                events += collect_events(sid, seq, gt_tracks, tracker)
            m = evaluate_history(history, gt_tracks)
            per_seq.append({"sequence": sid, **m})
            print(f"{name:13s} max_age={max_age:2d} MOT17-{sid}: IDF1={m['idf1']:.3f} "
                  f"sw={m['id_switches']} frag={m['fragmentations']}")
        row = {"config": name, "max_age": max_age,
               "mean_idf1": float(np.mean([r["idf1"] for r in per_seq])),
               "total_id_switches": int(sum(r["id_switches"] for r in per_seq)),
               "total_fragmentations": int(sum(r["fragmentations"] for r in per_seq)),
               "id_count_ratio": float(np.mean([r["predicted_identities"] / r["ground_truth_identities"]
                                                for r in per_seq])),
               "recovery": overall_recovery(events) if events else None, "per_sequence": per_seq}
        results.append(row)

    print("\n| config | max_age | IDF1 médio | switches | fragm. | IDs prev./reais | recuperação em oclusão |")
    print("|---|---|---|---|---|---|---|")
    for r in results:
        rec = r["recovery"]["recovery_rate"] if r["recovery"] else None
        print(f"| {r['config']} | {r['max_age']} | {r['mean_idf1']:.3f} | {r['total_id_switches']} | "
              f"{r['total_fragmentations']} | {r['id_count_ratio']:.2f} | "
              f"{'-' if rec is None else format(rec, '.3f')} |")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    labels = [f"{r['config']}\nmax_age={r['max_age']}" for r in results]
    colors = [PALETTE["muted"]] * 2 + [PALETTE["coral"]] * 2 + [PALETTE["teal"]] * 2
    axes[0].bar(range(len(results)), [r["mean_idf1"] for r in results], color=colors)
    axes[0].set_title("IDF1 médio")
    axes[1].bar(range(len(results)), [r["total_id_switches"] for r in results], color=colors)
    axes[1].set_title("ID switches (total)")
    for a in axes:
        a.set_xticks(range(len(results)))
        a.set_xticklabels(labels, fontsize=7)
    fig.tight_layout()
    fig.savefig(args.output / "correction_summary.png", dpi=160)
    plt.close(fig)

    with (args.output / "correction_results.json").open("w") as f:
        json.dump({"gradient_summary": grad_summary, "tracker_results": results,
                   "evaluated_sequences": eval_ids, "holdout": args.holdout},
                  f, indent=2)
    print(f"\nSaídas em {args.output}")


if __name__ == "__main__":
    main()