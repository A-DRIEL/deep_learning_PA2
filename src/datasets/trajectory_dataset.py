"""
Dataset de trajetórias para treinar a RNN de movimento (Trilha A).

Extrai, de cada sequência MOT17, todas as trajetórias reais (por ID),
e corta em janelas de T quadros CONSECUTIVOS (sem lacuna de frame) --
janelas que atravessam uma lacuna de anotação são descartadas, para
não ensinar a RNN a "pular" distância como se fosse um passo normal.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


def box_to_cxcywh_norm(box_left_top_w_h, img_w, img_h):
    left, top, w, h = box_left_top_w_h
    cx = (left + w / 2) / img_w
    cy = (top + h / 2) / img_h
    return np.array([cx, cy, w / img_w, h / img_h], dtype=np.float32)


def extract_trajectories(sequences_info):
    """
    sequences_info: lista de (seq_name, gt_array, img_w, img_h)
        gt_array: array MOT já filtrado (conf=1, class=1), colunas
                  [frame, id, left, top, w, h, conf, class, vis]

    Devolve: lista de trajetórias, cada uma um array (L, 4) em
    (cx, cy, w, h) normalizado, com L = comprimento da trajetória
    contínua (sem lacuna de frame).
    """
    all_trajectories = []

    for seq_name, gt, img_w, img_h in sequences_info:
        ids = np.unique(gt[:, 1]).astype(int)
        for obj_id in ids:
            rows = gt[gt[:, 1] == obj_id]
            rows = rows[np.argsort(rows[:, 0])]
            frames = rows[:, 0].astype(int)

            # separa em segmentos contínuos (sem pulo de frame)
            breaks = np.where(np.diff(frames) != 1)[0] + 1
            segments = np.split(rows, breaks)

            for seg in segments:
                if len(seg) < 2:
                    continue  # precisa de pelo menos 2 pontos p/ ter um "próximo"
                boxes = np.array([
                    box_to_cxcywh_norm(r[2:6], img_w, img_h) for r in seg
                ])
                all_trajectories.append(boxes)

    return all_trajectories


class TrajectoryWindowDataset(Dataset):
    """
    Corta cada trajetória em janelas de tamanho fixo window_size (para
    BPTT truncado). Trajetórias mais curtas que window_size+1 são
    descartadas (não dá pra formar nem uma janela com alvo).
    """

    def __init__(self, trajectories, window_size=8):
        self.window_size = window_size
        self.windows = []
        for traj in trajectories:
            L = len(traj)
            if L < window_size + 1:
                continue
            for start in range(0, L - window_size):
                self.windows.append(traj[start:start + window_size + 1])

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]  # (window_size+1, 4)
        inputs = torch.from_numpy(window[:-1])   # (window_size, 4)
        targets = torch.from_numpy(window[1:])   # (window_size, 4) -- previsão em cada passo
        return inputs, targets