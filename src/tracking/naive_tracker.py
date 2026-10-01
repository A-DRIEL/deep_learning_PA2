"""
Associação ingênua por IoU entre quadros consecutivos (Parte 1, item 2).

Regra de associação: matching guloso por IoU decrescente entre as
detecções do quadro t e as tracks vivas do quadro t-1 (mesma filosofia
do matching de instâncias no PA1 -- documentado explicitamente aqui,
como o enunciado exige).

Gestão de tracks:
  - detecção sem par acima do limiar -> nasce uma track nova
  - track sem detecção correspondente no quadro -> fica "órfã"
  - track órfã por mais de `max_age` quadros seguidos -> morre
"""

import numpy as np

from src.metrics.tracking_metrics import compute_iou


class NaiveTracker:
    def __init__(self, iou_threshold=0.3, max_age=10):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.next_id = 0
        self.tracks = {}       # id -> box atual
        self.age = {}          # id -> quadros seguidos sem observação
        self.history = {}      # id -> {frame: box}, para avaliação depois

    def step(self, frame_idx, detections):
        """
        detections: array (N, 4) de caixas (y0, x0, y1, x1) do quadro atual
        Devolve: dict id -> box, para as tracks vivas NESTE quadro
        """
        track_ids = list(self.tracks.keys())

        # --- matriz de IoU entre tracks vivas e detecções novas ---
        pairs = []
        for tid in track_ids:
            for j, det in enumerate(detections):
                iou = compute_iou(self.tracks[tid], det)
                if iou >= self.iou_threshold:
                    pairs.append((iou, tid, j))

        # --- matching guloso: maior IoU primeiro, cada track/detecção usada uma vez ---
        pairs.sort(key=lambda p: p[0], reverse=True)
        matched_tracks, matched_dets = set(), set()
        assignments = {}
        for iou, tid, j in pairs:
            if tid in matched_tracks or j in matched_dets:
                continue
            matched_tracks.add(tid)
            matched_dets.add(j)
            assignments[tid] = j

        current_frame_output = {}

        # --- atualiza tracks que casaram ---
        for tid, j in assignments.items():
            self.tracks[tid] = tuple(detections[j])
            self.age[tid] = 0
            self.history.setdefault(tid, {})[frame_idx] = tuple(detections[j])
            current_frame_output[tid] = tuple(detections[j])

        # --- envelhece e mata tracks que não casaram ---
        for tid in track_ids:
            if tid not in assignments:
                self.age[tid] += 1
                if self.age[tid] > self.max_age:
                    del self.tracks[tid]
                    del self.age[tid]
                # se não morreu ainda, continua "viva" mas sem observação neste quadro
                # (não entra em current_frame_output nem em history neste frame)

        # --- nasce uma track nova para cada detecção não usada ---
        for j, det in enumerate(detections):
            if j not in matched_dets:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = tuple(det)
                self.age[tid] = 0
                self.history.setdefault(tid, {})[frame_idx] = tuple(det)
                current_frame_output[tid] = tuple(det)

        return current_frame_output

    def run(self, detections_by_frame):
        """detections_by_frame: lista de arrays (N_t, 4), um por quadro"""
        for frame_idx, dets in enumerate(detections_by_frame):
            self.step(frame_idx, dets)
        return self.history