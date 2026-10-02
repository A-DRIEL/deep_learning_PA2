"""Parte 4: galeria de falhas e horizonte de memória efetivo.

Gera em outputs/part4/:
  gradient_horizon.png      curva analítica ||dL_t/dh_{t-k}|| (k até --grad-window)
  memory_survival.png       (a) oferta (sobrevivência do estado, Kaplan-Meier) vs
                                demanda (duração das oclusões do dataset)
                            (b) taxa de recuperação e causas de falha por faixa de oclusão
  failure_*.png             3 falhas de CAUSAS diferentes, com diagnóstico numérico
  part4_memory_results.json

A correção (antes/depois) está em scripts/run_part4_correction.py.
"""

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import torch
from PIL import Image

from src.analysis.memory_horizon import (
    FAILURE_CATEGORIES, IOU_ASSIGN, VISIBILITY_THRESHOLD, build_gt_tracks, build_visibility,
    collect_events, gradient_horizon, kaplan_meier, overall_recovery, recovery_table,
    run_lstm_tracker, summarize_gradient_curve,
)
from src.datasets.mot17 import MOT17Sequence, SEQUENCE_IDS
from src.datasets.trajectory_dataset import extract_trajectories
from src.metrics.tracking_metrics import compute_iou
from src.models.motion_lstm import MotionLSTM
from src.viz.plot_style import PALETTE, apply_style

DETECTOR = "SDP"
TRAIN_WINDOW = 8          # janela de BPTT usada no treino do checkpoint
apply_style()
GT_COLOR = "#FFD400"
CAT_COLORS = {"recovered": "#2E8B57", "alive_not_matched": "#7A5195",
              "stolen": "#E4572E", "expired": "#5B6B70"}
CAT_LABELS = {"recovered": "mesmo ID recuperado", "alive_not_matched": "track viva, sem casar na volta (deriva)",
              "stolen": "track passou a seguir outro objeto", "expired": "track removida (max_age)"}


# ----------------------------------------------------------------------
# figuras agregadas
# ----------------------------------------------------------------------
def save_gradient_plot(curve, summary, path):
    lag = np.asarray(curve["lags_frames"])
    med = np.asarray(curve["median_gradient_norm"])
    q25 = np.asarray(curve["q25_gradient_norm"])
    q75 = np.asarray(curve["q75_gradient_norm"])
    floor = max(float(med[med > 0].min()) * 0.5, 1e-30) if (med > 0).any() else 1e-30
    med, q25, q75 = (np.maximum(a, floor) for a in (med, q25, q75))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(lag, med, "o-", color=PALETTE["teal"], label="mediana")
    ax.fill_between(lag, q25, q75, color=PALETTE["teal"], alpha=0.2, label="intervalo interquartil")
    ax.axvline(TRAIN_WINDOW - 1, color=PALETTE["muted"], linestyle="--", label=f"limite do BPTT de treino (k={TRAIN_WINDOW - 1})")
    for key, ls in (("k_eff_10pct", ":"), ("k_eff_1pct", "-.")):
        k = summary.get(key)
        if k is not None:
            ax.axvline(k, color=PALETTE["coral"], linestyle=ls, label=f"{key.replace('k_eff_', 'k_eff(')[:-3]}% do k=0) = {k}")
    ax.set_yscale("log")
    ax.set_xlabel("k: passos para trás a partir da previsão supervisionada")
    ax.set_ylabel(r"Norma de $\partial L_t / \partial h_{t-k}$")
    ax.set_title(f"Horizonte analítico de memória · free-running · janela {curve['window']} "
                 f"({curve['samples']} janelas)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_survival_plot(events, path, max_age):
    tracked = [e for e in events if e["category"] != "no_prior_track"]
    n_excluded = len(events) - len(tracked)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4))
    title_kw = dict(loc="left", fontsize=12, fontweight="bold", pad=62)
    legend_kw = dict(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, fontsize=9,
                     borderaxespad=0.0, columnspacing=1.8, handlelength=2.4, frameon=False)

    # (a) demanda x oferta
    ax = axes[0]
    xmax = 120
    ts = np.arange(0, xmax + 1)
    dur_all = np.array([e["duration"] for e in events])
    gap_tr = np.array([e["blind_gap"] for e in tracked])
    demand_all = np.array([(dur_all > t).mean() for t in ts])
    demand_tr = np.array([(gap_tr > t).mean() for t in ts])
    km_t, km_s = kaplan_meier([e["survival_frames"] for e in tracked],
                              [not e["survival_censored"] for e in tracked])
    km_t = np.append(km_t, xmax)
    km_s = np.append(km_s, km_s[-1])
    ax.step(ts, demand_all, where="post", color=PALETTE["muted"], linewidth=1.8,
            label=f"Demanda: duração da oclusão (todas, n={len(events)})")
    ax.step(ts, demand_tr, where="post", color=PALETTE["teal"], linewidth=1.8,
            label=f"Demanda: quadros sem observação (com track prévia, n={len(tracked)})")
    ax.step(km_t, km_s, where="post", color=PALETTE["coral"], linewidth=2.8,
            label="Oferta: estado íntegro (Kaplan-Meier)")
    ax.axvline(max_age, color="#333333", linestyle="--", linewidth=1.2, label=f"max_age = {max_age}")
    ax.axvline(TRAIN_WINDOW, color="#333333", linestyle=":", linewidth=1.4, label=f"janela de BPTT = {TRAIN_WINDOW}")
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("quadros desde a última observação da track")
    ax.set_ylabel("fração")
    ax.set_title("(a) O que o dataset exige vs. quanto o estado sobrevive", **title_kw)
    ax.legend(**legend_kw)

    # (b) taxa de recuperação por faixa
    ax = axes[1]
    table = recovery_table(events)
    bottom = np.zeros(len(table))
    for cat in ("recovered", "alive_not_matched", "stolen", "expired"):
        frac = np.array([(r[cat] / r["n"]) if r["n"] else 0.0 for r in table])
        ax.bar(range(len(table)), frac, bottom=bottom, color=CAT_COLORS[cat],
               label=CAT_LABELS[cat], width=0.72)
        bottom += frac
    for i, r in enumerate(table):
        ax.text(i, 1.015, f"n={r['n']}", ha="center", fontsize=9, color=PALETTE["muted"])
    ax.set_xticks(range(len(table)))
    ax.set_xticklabels([r["bin"] for r in table])
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("quadros sem observação (última observação -> objeto visível de novo)")
    ax.set_ylabel("fração dos eventos")
    ax.set_title("(b) O que acontece com a identidade", **title_kw)
    ax.legend(**legend_kw)

    fig.text(0.995, 0.0, f"{n_excluded} eventos sem track prévia foram excluídos (não testam memória)",
             ha="right", va="top", fontsize=8, color=PALETTE["muted"])
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# seleção e diagnóstico das falhas
# ----------------------------------------------------------------------
def select_gallery(events, n=3):
    """Uma falha de cada CAUSA (expired / stolen / alive_not_matched), preferindo
    oclusões de 10-100 quadros; completa com as mais longas se faltar categoria."""
    failures = [e for e in events if e["category"] in FAILURE_CATEGORIES]
    key = lambda e: (10 <= e["duration"] <= 100, e["duration"])
    picked, used = [], set()

    def add(pool):
        for e in sorted(pool, key=key, reverse=True):
            k = (e["sequence"], e["gt_id"])
            if k not in used:
                picked.append(e)
                used.add(k)
                return

    for cat in FAILURE_CATEGORIES:
        add([e for e in failures if e["category"] == cat])
    while len(picked) < n:
        before = len(picked)
        add(failures)
        if len(picked) == before:
            break
    return picked[:n]


def fmt(x, spec=".2f", none="n/d"):
    return none if x is None else format(x, spec)


def make_diagnosis(ev, grad_summary, max_age):
    cat = ev["category"]
    base = (f"GT {ev['gt_id']} (MOT17-{ev['sequence']}): visibilidade < {VISIBILITY_THRESHOLD:.1f} por "
            f"{ev['duration']} quadros; a track {ev['before_pred_id']} ficou {ev['blind_gap']} quadros sem "
            f"observação (último quadro rastreado: {ev['last_obs_frame']}). ")
    cause = {
        "expired": (f"A track foi removida por max_age={max_age} no quadro {ev['death_frame']}, antes do objeto "
                    f"reaparecer: a falha é de gestão de tracks, nenhuma memória do LSTM salvaria a identidade "
                    f"com essa regra. "),
        "stolen": (f"A track passou a seguir OUTRO objeto a partir do quadro {ev['stolen_frame']}: a caixa "
                   f"extrapolada colidiu com outra pessoa e o portão de IoU aceitou a troca. "),
        "alive_not_matched": ("A track ainda estava viva, mas a caixa extrapolada já não sobrepunha o objeto ao "
                              "reaparecer (IoU < 0,3): o estado derivou e nasceu uma identidade nova. "),
    }[cat]
    quality = (f"IoU médio da caixa prevista durante a oclusão = {fmt(ev['mean_pred_iou'])}; erro de centro no "
               f"último quadro ocluído = {fmt(ev['center_err_last'])} alturas do objeto. ")
    gate = ""
    if ev.get("gate_loss_frame") is not None:
        gate = (f"A caixa prevista saiu do portão de IoU {ev['gate_loss_frame'] - ev['last_obs_frame']} quadros "
                f"após a última observação. ")
    k10 = grad_summary.get("k_eff_10pct")
    memory = (f"Memória: BPTT de treino = {grad_summary['train_window']} passos; o gradiente cai "
              f"{grad_summary['attenuation_at_train_window']:.0f}x até k={grad_summary['train_window'] - 1} "
              f"e passa de 10% para menos em k={k10}. ")
    if ev.get("gate_loss_frame") is not None and ev["blind_gap"] > grad_summary["train_window"]:
        memory += (f"Como {ev['blind_gap']} > {grad_summary['train_window']}, o treino nunca deu supervisão "
                   f"que atravessasse esse buraco.")
    return base + cause + quality + gate + memory


# ----------------------------------------------------------------------
# galeria
# ----------------------------------------------------------------------
def _box_patch(box, color, ls="-", lw=2.5):
    y0, x0, y1, x1 = box
    return Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linestyle=ls, linewidth=lw)


def crop_limits(boxes, img_w, img_h, aspect=1.0, min_h=300.0):
    """Recorte com proporção fixa (w/h = aspect) centrado nas caixas GT mostradas."""
    y0 = min(b[0] for b in boxes)
    y1 = max(b[2] for b in boxes)
    x0 = min(b[1] for b in boxes)
    x1 = max(b[3] for b in boxes)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ch = max(min_h, 2.5 * (y1 - y0), 1.4 * (x1 - x0) / aspect)
    cw = ch * aspect
    if cw > img_w:
        cw, ch = img_w, img_w / aspect
    if ch > img_h:
        ch, cw = img_h, img_h * aspect
    left = min(max(cx - cw / 2, 0), img_w - cw)
    top = min(max(cy - ch / 2, 0), img_h - ch)
    return left, left + cw, top, top + ch


def _mark_events(ax, ev, label=False):
    """Faixa de oclusão + marcadores verticais (último quadro rastreado, perda do portão, morte)."""
    ax.axvspan(ev["start"], ev["end"], color="gray", alpha=0.13, linewidth=0)
    marks = [(ev["last_obs_frame"], PALETTE["muted"], ":", "último quadro rastreado"),
             (ev.get("gate_loss_frame"), "#E08A00", "-.", "saiu do portão de IoU"),
             (ev.get("death_frame"), "#C62828", "-", "track removida (max_age)")]
    for x, color, ls, text in marks:
        if x is None:
            continue
        ax.axvline(x, color=color, linestyle=ls, linewidth=1.5)
        if label:
            ax.text(x, 0.97, " " + text, rotation=90, ha="left", va="top", fontsize=8, color=color,
                    transform=ax.get_xaxis_transform())
    if label:
        ax.text((ev["start"] + ev["end"]) / 2, 0.03, "oclusão (vis < 0,1)", ha="center", va="bottom",
                fontsize=8, color=PALETTE["muted"], transform=ax.get_xaxis_transform())


def gallery(ev, seq, gt_track, vis_track, tracker, destination, subtitle=True):
    gid, start, end = ev["gt_id"], ev["start"], ev["end"]
    ref, after = ev["before_pred_id"], ev["after_pred_id"]
    L = seq.info.seq_length
    cand = [ev["last_obs_frame"], start, (start + end) // 2, end, ev["first_reobs_frame"] or end + 1]
    frames = sorted({int(f) for f in cand if f is not None and 1 <= f <= L})
    if len(frames) < 2:
        frames = sorted({max(1, start - 1), start, min(L, end + 1)})

    # IDs preditos que aparecem sobre o alvo -> mesmo mapa de cores em todos os quadros
    overlapping = {}
    for f in frames:
        if f in gt_track:
            overlapping[f] = [pid for pid, fr in tracker.history.items()
                              if f in fr and compute_iou(fr[f], gt_track[f]) >= IOU_ASSIGN]
    ids = sorted({p for p in (ref, after) if p is not None} | {p for v in overlapping.values() for p in v})
    cmap = plt.get_cmap("tab10")
    color_of = {pid: cmap(i % 10) for i, pid in enumerate(ids)}

    cx0, cx1, cy0, cy1 = crop_limits([gt_track[f] for f in frames if f in gt_track],
                                     seq.info.im_width, seq.info.im_height)

    n = len(frames)
    fig = plt.figure(figsize=(3.1 * n + 0.4, 7.6), facecolor="white")
    grid = fig.add_gridspec(2, n, height_ratios=[2.0, 1.0], hspace=0.42, wspace=0.06,
                            left=0.05, right=0.985, top=0.80, bottom=0.15)

    for col, f in enumerate(frames):
        path = seq.frame_path(f)
        if not path.exists():
            raise FileNotFoundError(f"Quadro {path} ausente. Baixe o pacote completo do MOT17 (img1).")
        ax = fig.add_subplot(grid[0, col])
        ax.imshow(np.asarray(Image.open(path).convert("RGB")))
        ax.set_xlim(cx0, cx1)
        ax.set_ylim(cy1, cy0)
        if f in gt_track:
            ax.add_patch(_box_patch(gt_track[f], GT_COLOR, "-", 3.0))
        predicted = tracker.predicted_history.get(ref, {}).get(f) if ref is not None else None
        if predicted is not None:
            ax.add_patch(_box_patch(predicted, color_of[ref], ":", 2.4))      # previsão da recorrência
        for pid in overlapping.get(f, []):
            b = tracker.history[pid][f]
            ax.add_patch(_box_patch(b, color_of[pid], "--", 2.2))
            ax.text(b[1], b[0] - 4, f"ID {pid}", color="white", fontsize=8, weight="bold",
                    bbox=dict(facecolor=color_of[pid], alpha=0.9, pad=1.2, edgecolor="none"))
        vis = vis_track.get(f)
        hidden = vis is not None and vis < VISIBILITY_THRESHOLD
        ax.set_title(f"quadro {f}", fontsize=11, weight="bold")
        ax.text(0.5, 0.025, f"{'oculto' if hidden else 'visível'} · vis {fmt(vis)}",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5, color="white",
                bbox=dict(facecolor="#C62828" if hidden else "#2E7D32", edgecolor="none",
                          boxstyle="round,pad=0.3", alpha=0.92))
        ax.axis("off")

    # --- painéis temporais ---
    k = max(1, (n + 1) // 2)
    sub = grid[1, :].subgridspec(1, 2, width_ratios=[k, n - k] if n > k else [1, 1], wspace=0.28)
    ax_pos = fig.add_subplot(sub[0, 0])
    ax_iou = fig.add_subplot(sub[0, 1])
    ref_frame = ev["last_obs_frame"]
    t0 = max(1, ref_frame - 5)
    t1 = min(L, max(end, ev["first_reobs_frame"] or end) + 10)
    pred = tracker.predicted_history.get(ref, {}) if ref is not None else {}
    gt_f = [f for f in range(t0, t1 + 1) if f in gt_track]
    pr_f = [f for f in range(t0, t1 + 1) if f in pred]

    def center(b):
        return (b[1] + b[3]) / 2, (b[0] + b[2]) / 2

    ox, oy = center(gt_track[ref_frame] if ref_frame in gt_track else gt_track[gt_f[0]])
    x_color, y_color = PALETTE["teal"], PALETTE["coral"]
    ax_pos.plot(gt_f, [center(gt_track[f])[0] - ox for f in gt_f], "-", color=x_color, lw=2, label="Δx GT")
    ax_pos.plot(gt_f, [center(gt_track[f])[1] - oy for f in gt_f], "-", color=y_color, lw=2, label="Δy GT")
    ax_pos.plot(pr_f, [center(pred[f])[0] - ox for f in pr_f], "--", color=x_color, lw=2, label="Δx previsto")
    ax_pos.plot(pr_f, [center(pred[f])[1] - oy for f in pr_f], "--", color=y_color, lw=2, label="Δy previsto")
    ax_pos.axhline(0, color="#999999", lw=0.8)
    ax_pos.set_ylabel("deslocamento (px)")
    ax_pos.set_title("Deslocamento do centro desde o último quadro rastreado", fontsize=10, weight="bold", loc="left")
    ax_pos.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=4, fontsize=8.5, frameon=False)

    both = [f for f in pr_f if f in gt_track]
    ax_iou.plot(both, [compute_iou(pred[f], gt_track[f]) for f in both], "-", color=PALETTE["muted"], lw=2.2)
    ax_iou.axhline(IOU_ASSIGN, color="#333333", linestyle="--", linewidth=1.1)
    ax_iou.text(t1, IOU_ASSIGN + 0.02, f"portão IoU = {IOU_ASSIGN}", ha="right", va="bottom", fontsize=8)
    ax_iou.set_ylim(0, 1)
    ax_iou.set_ylabel("IoU previsão × GT")
    ax_iou.set_title("A caixa prevista ainda sobrepõe o objeto?", fontsize=10, weight="bold", loc="left")

    for a, lab in ((ax_pos, False), (ax_iou, True)):
        _mark_events(a, ev, label=lab)
        a.set_xlim(t0, t1)
        a.set_xlabel("quadro")

    fig.suptitle(f"MOT17-{ev['sequence']} · GT {gid} · {CAT_LABELS[ev['category']]}",
                 fontsize=15, weight="bold", y=0.985)
    if subtitle:
        iou_txt = fmt(ev["mean_pred_iou"])
        fig.text(0.5, 0.935, f"oclusão de {ev['duration']} quadros · {ev['blind_gap']} quadros sem observação "
                 f"da track · IoU médio da previsão durante a oclusão = {iou_txt}",
                 ha="center", fontsize=10.5, color=PALETTE["muted"])
    handles = [Line2D([0], [0], color=GT_COLOR, lw=3, label="ground truth"),
               Line2D([0], [0], color="#444444", lw=2.4, ls=":", label="previsão da recorrência"),
               Line2D([0], [0], color="#444444", lw=2.2, ls="--", label="track casada (cor = ID)")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.905), ncol=3,
               frameon=False, fontsize=10)
    fig.savefig(destination, dpi=170, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/MOT17/train"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/ablation_checkpoints/free_running_seed0.pt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/part4"))
    parser.add_argument("--max-age", type=int, default=30)
    parser.add_argument("--grad-window", type=int, default=32,
                        help="comprimento da janela para a curva analítica (>= janela de treino)")
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--no-subtitle", action="store_true",
                        help="remove também a linha de fatos abaixo do título das figuras de falha")
    parser.add_argument("--legacy-tracker", action="store_true",
                        help="reproduz o protocolo de estado antigo (caixa consumida duas vezes)")
    args = parser.parse_args()

    missing = [str(args.checkpoint)] if not args.checkpoint.exists() else []
    missing += [str(args.data_root / f"MOT17-{s}-{DETECTOR}") for s in SEQUENCE_IDS
                if not (args.data_root / f"MOT17-{s}-{DETECTOR}" / "gt" / "gt.txt").exists()]
    if missing:
        raise FileNotFoundError("Arquivos ausentes:\n  " + "\n  ".join(missing))

    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MotionLSTM(input_dim=4, hidden_dim=64).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()

    sequences, seq_info, all_events, trackers = {}, [], [], {}
    for sid in SEQUENCE_IDS:
        seq = MOT17Sequence(args.data_root / f"MOT17-{sid}-{DETECTOR}", load_gt=True)
        if seq.gt is None or len(seq.gt) == 0:
            continue
        gt_tracks = build_gt_tracks(seq)
        tracker = run_lstm_tracker(seq, model, device, args.max_age, legacy=args.legacy_tracker)
        sequences[sid] = (seq, gt_tracks, build_visibility(seq))
        trackers[sid] = tracker
        seq_info.append((seq.info.name, seq.gt, seq.info.im_width, seq.info.im_height))
        events = collect_events(sid, seq, gt_tracks, tracker)
        all_events.extend(events)
        print(f"MOT17-{sid}: {len(events)} oclusões analisadas")
    if not sequences:
        raise RuntimeError("Nenhuma sequência com GT válido.")

    # --- medida analítica ---
    trajectories = extract_trajectories(seq_info)
    curve = gradient_horizon(model, trajectories, device, args.grad_window, args.max_samples)
    grad_summary = summarize_gradient_curve(curve, TRAIN_WINDOW)
    save_gradient_plot(curve, grad_summary, args.output / "gradient_horizon.png")

    # --- medida empírica ---
    save_survival_plot(all_events, args.output / "memory_survival.png", args.max_age)
    table = recovery_table(all_events)
    overall = overall_recovery(all_events)

    # --- galeria ---
    selected = select_gallery(all_events)
    if len(selected) < 3:
        print(f"AVISO: só {len(selected)} falhas distintas encontradas.")
    for stale in args.output.glob("failure_*.png"):
        stale.unlink()
    gallery_out = []
    for i, ev in enumerate(selected, start=1):
        seq, gt_tracks, vis = sequences[ev["sequence"]]
        name = f"failure_{i}_{ev['category']}_{ev['sequence']}_gt{ev['gt_id']}_f{ev['start']}.png"
        diagnosis = make_diagnosis(ev, grad_summary, args.max_age)
        gallery(ev, seq, gt_tracks[ev["gt_id"]], vis[ev["gt_id"]], trackers[ev["sequence"]],
                args.output / name, subtitle=not args.no_subtitle)
        gallery_out.append({**ev, "figure": name, "diagnosis": diagnosis})
        print(f"[{ev['category']}] {name}\n  {diagnosis}\n")

    result = {
        "checkpoint": str(args.checkpoint), "detector": DETECTOR, "max_age": args.max_age,
        "tracker_protocol": "legacy" if args.legacy_tracker else "single_consume",
        "train_bptt_window": TRAIN_WINDOW,
        "analytical_horizon": {"curve": curve, "summary": grad_summary},
        "empirical_horizon": {"overall": overall, "by_blind_gap": table,
                              "visibility_threshold": VISIBILITY_THRESHOLD, "events": all_events},
        "failure_gallery": gallery_out,
    }
    with (args.output / "part4_memory_results.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("Resumo empírico:", json.dumps(overall, ensure_ascii=False))
    print("Resumo analítico:", json.dumps(grad_summary))
    print(f"Saídas em {args.output}")


if __name__ == "__main__":
    main()