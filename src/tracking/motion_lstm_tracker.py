"""
Tracker da Trilha A: usa o MotionLSTM para extrapolar posição durante
oclusão, em vez de congelar a última observação (o que o NaiveTracker
da Parte 1 faz implicitamente).
"""

import numpy as np
import torch

from src.metrics.tracking_metrics import compute_iou


def pixel_to_norm_cxcywh(box_left_top_w_h, img_w, img_h):
    left, top, w, h = box_left_top_w_h
    cx = (left + w / 2) / img_w
    cy = (top + h / 2) / img_h
    return np.array([cx, cy, w / img_w, h / img_h], dtype=np.float32)


def norm_cxcywh_to_pixel_yxyx(box, img_w, img_h):
    cx, cy, w, h = box
    left = (cx - w / 2) * img_w
    top = (cy - h / 2) * img_h
    width = w * img_w
    height = h * img_h
    return (top, left, top + height, left + width)


class MotionLSTMTracker:
    def __init__(self, model, device, img_w, img_h, iou_threshold=0.3, max_age=30):
        self.model = model.to(device).eval()
        self.device = device
        self.img_w = img_w
        self.img_h = img_h
        self.iou_threshold = iou_threshold
        self.max_age = max_age

        self.next_id = 0
        self.current_box = {}          # id -> última posição (observada OU extrapolada), norm cxcywh
        self.hidden = {}               # id -> (h, c) do LSTM, DEPOIS do passo mais recente
        self.hidden_before_predict = {}  # id -> (h, c) ANTES da extrapolação do quadro atual
        self.age = {}
        self.history = {}              # id -> {frame: box pixel yxyx}, para avaliação
        # Caixa que a recorrência propôs antes de observar as detecções do quadro.
        # Mantida separada de history para permitir inspecionar o mapa intermediário.
        self.predicted_history = {}    # id -> {frame: box pixel yxyx}

    @torch.no_grad()
    def _predict_next(self, track_id):
        last_box = torch.from_numpy(self.current_box[track_id]).to(self.device)
        pred_position, hidden = self.model.predict_step(last_box, self.hidden[track_id])
        self.hidden[track_id] = hidden
        return pred_position.cpu().numpy()

    def step(self, frame_idx, detections_pixel):
        """detections_pixel: array (N, 4) em (left, top, w, h)"""
        track_ids = list(self.current_box.keys())

        # salva o hidden ANTES de extrapolar -- precisa dele se a observação
        # real chegar depois, para recalcular o estado corretamente
        self.hidden_before_predict = {tid: self.hidden[tid] for tid in track_ids}

        predicted = {}
        for tid in track_ids:
            predicted[tid] = self._predict_next(tid)
            box_pixel = norm_cxcywh_to_pixel_yxyx(predicted[tid], self.img_w, self.img_h)
            self.predicted_history.setdefault(tid, {})[frame_idx] = box_pixel

        dets_norm = [pixel_to_norm_cxcywh(d, self.img_w, self.img_h) for d in detections_pixel]

        def iou_norm(a, b):
            box_a = norm_cxcywh_to_pixel_yxyx(a, self.img_w, self.img_h)
            box_b = norm_cxcywh_to_pixel_yxyx(b, self.img_w, self.img_h)
            return compute_iou(box_a, box_b)

        pairs = []
        for tid in track_ids:
            for j, det in enumerate(dets_norm):
                iou = iou_norm(predicted[tid], det)
                if iou >= self.iou_threshold:
                    pairs.append((iou, tid, j))
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

        for tid in track_ids:
            if tid in assignments:
                # observado: recalcula o hidden a partir da OBSERVAÇÃO real,
                # não da extrapolação -- corrige o estado com dado de verdade
                observed = dets_norm[assignments[tid]]
                with torch.no_grad():
                    obs_t = torch.from_numpy(observed).to(self.device)
                    _, hidden = self.model.predict_step(obs_t, self.hidden_before_predict[tid])
                    self.hidden[tid] = hidden
                self.current_box[tid] = observed
                self.age[tid] = 0
                box_pixel = norm_cxcywh_to_pixel_yxyx(observed, self.img_w, self.img_h)
                self.history.setdefault(tid, {})[frame_idx] = box_pixel
                current_frame_output[tid] = box_pixel
            else:
                # ocluído: usa a extrapolação como posição atual, track continua viva
                self.current_box[tid] = predicted[tid]
                self.age[tid] += 1
                if self.age[tid] > self.max_age:
                    del self.current_box[tid]
                    del self.hidden[tid]
                    del self.age[tid]

        for j, det in enumerate(dets_norm):
            if j not in matched_dets:
                tid = self.next_id
                self.next_id += 1
                self.current_box[tid] = det
                self.hidden[tid] = None
                self.age[tid] = 0
                box_pixel = norm_cxcywh_to_pixel_yxyx(det, self.img_w, self.img_h)
                self.history.setdefault(tid, {})[frame_idx] = box_pixel
                current_frame_output[tid] = box_pixel

        return current_frame_output

    def run(self, detections_by_frame):
        for frame_idx, dets in enumerate(detections_by_frame):
            self.step(frame_idx, dets)
        return self.history
