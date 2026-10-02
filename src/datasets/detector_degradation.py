"""
Degradação controlada de detecções do MOT17 (Parte 5, opção "Qualidade
do detector").

Generaliza o detector_simulator.py da Parte 0 (que trabalha em 128x128,
com ruído em pixels absolutos) para detecções reais de pedestres:
  - descarte:  cada detecção é removida com probabilidade `drop_prob`
               (falso negativo)
  - ruído:     cada coordenada da caixa recebe ruído gaussiano com desvio
               PROPORCIONAL ao tamanho da caixa (`noise_rel` * largura/altura).
               Proporcional porque um pedestre de 30px e um de 300px não
               podem receber o mesmo erro em pixels.
  - falsos positivos: por quadro, Poisson(fp_ratio * n_detecções_originais).
               Escala com a densidade da cena. Tamanho copiado de caixas
               reais do quadro (com variação), posição uniforme na imagem,
               confiança sorteada da distribuição real de confianças da
               sequência (para o FP não ser distinguível só pelo score).

Formato de entrada/saída por quadro: array (N, 5) = [left, top, w, h, conf].
"""

from dataclasses import dataclass

import numpy as np

MIN_BOX_SIZE = 2.0  # pixels; evita caixas degeneradas depois do ruído


@dataclass(frozen=True)
class DegradationConfig:
    name: str
    drop_prob: float = 0.0
    noise_rel: float = 0.0
    fp_ratio: float = 0.0


# As 3 intensidades pedidas no enunciado (+ "limpo" como referência).
INTENSITIES = {
    "limpo": DegradationConfig("limpo"),
    "leve": DegradationConfig("leve", drop_prob=0.10, noise_rel=0.05, fp_ratio=0.10),
    "media": DegradationConfig("media", drop_prob=0.25, noise_rel=0.10, fp_ratio=0.30),
    "pesada": DegradationConfig("pesada", drop_prob=0.40, noise_rel=0.20, fp_ratio=0.60),
}

# Opcional: cada falha sozinha, na intensidade média (para isolar a causa).
ISOLATED = {
    "so_descarte": DegradationConfig("so_descarte", drop_prob=0.25),
    "so_ruido": DegradationConfig("so_ruido", noise_rel=0.10),
    "so_fp": DegradationConfig("so_fp", fp_ratio=0.30),
}


def split_by_frame(arr, n_frames):
    """
    arr: array MOT (M, >=7) com frame na coluna 0 (1-based) e
         left, top, w, h, conf nas colunas 2..6 (vale para det.txt e gt.txt).
    Devolve lista de n_frames arrays (N_t, 5) = [left, top, w, h, conf].
    """
    frames = arr[:, 0].astype(int)
    order = np.argsort(frames, kind="stable")
    arr, frames = arr[order], frames[order]
    starts = np.searchsorted(frames, np.arange(1, n_frames + 2))
    return [
        arr[starts[t]:starts[t + 1]][:, [2, 3, 4, 5, 6]].astype(np.float64)
        for t in range(n_frames)
    ]


def degrade_frame(dets, rng, cfg, img_w, img_h, conf_pool, size_pool):
    """Degrada as detecções de UM quadro. dets: (N, 5)."""
    n_orig = len(dets)
    out = dets.astype(np.float64).copy()

    # 1) descarte
    if cfg.drop_prob > 0 and len(out) > 0:
        keep = rng.random(len(out)) >= cfg.drop_prob
        out = out[keep]

    # 2) ruído proporcional ao tamanho da caixa
    if cfg.noise_rel > 0 and len(out) > 0:
        left, top, w, h = out[:, 0].copy(), out[:, 1].copy(), out[:, 2].copy(), out[:, 3].copy()
        n = len(out)
        x0 = left + rng.normal(0, cfg.noise_rel, n) * w
        x1 = left + w + rng.normal(0, cfg.noise_rel, n) * w
        y0 = top + rng.normal(0, cfg.noise_rel, n) * h
        y1 = top + h + rng.normal(0, cfg.noise_rel, n) * h
        out[:, 0] = x0
        out[:, 1] = y0
        out[:, 2] = np.maximum(x1 - x0, MIN_BOX_SIZE)
        out[:, 3] = np.maximum(y1 - y0, MIN_BOX_SIZE)

    # 3) falsos positivos (taxa baseada na contagem ORIGINAL, antes do descarte,
    #    para que a taxa de FP seja independente da taxa de descarte)
    if cfg.fp_ratio > 0:
        n_fp = rng.poisson(cfg.fp_ratio * n_orig)
        if n_fp > 0:
            ref_sizes = dets[:, 2:4] if n_orig > 0 else size_pool
            idx = rng.integers(0, len(ref_sizes), n_fp)
            scale = rng.uniform(0.8, 1.25, n_fp)
            w = np.minimum(ref_sizes[idx, 0] * scale, img_w - 1)
            h = np.minimum(ref_sizes[idx, 1] * scale, img_h - 1)
            left = rng.uniform(0, np.maximum(img_w - w, 1.0))
            top = rng.uniform(0, np.maximum(img_h - h, 1.0))
            conf = rng.choice(conf_pool, n_fp)
            fps = np.stack([left, top, w, h, conf], axis=1)
            out = np.concatenate([out, fps], axis=0) if len(out) else fps

    return out if len(out) else np.zeros((0, 5))


def degrade_sequence(det, n_frames, cfg, seed, img_w, img_h):
    """
    det: array de det.txt da sequência inteira (M, 10).
    Devolve lista (n_frames) de arrays (N_t, 5) degradados.
    O mesmo (seed, cfg) sempre produz o mesmo resultado.
    """
    rng = np.random.default_rng(seed)
    per_frame = split_by_frame(det, n_frames)
    conf_pool = det[:, 6].astype(np.float64)
    size_pool = det[:, 4:6].astype(np.float64)
    return [
        degrade_frame(d, rng, cfg, img_w, img_h, conf_pool, size_pool)
        for d in per_frame
    ]
