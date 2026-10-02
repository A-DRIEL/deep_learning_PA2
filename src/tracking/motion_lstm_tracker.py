"""
Tracker da Trilha A: usa o MotionLSTM para extrapolar posição durante
oclusão, em vez de congelar a última observação (o que o NaiveTracker
da Parte 1 faz implicitamente).

Protocolo de estado (modo padrão, `legacy_state_update=False`)
--------------------------------------------------------------
No treino, cada caixa x_t entra no LSTM EXATAMENTE UMA VEZ e a saída é
a previsão de x_{t+1} = x_t + head(h_t). O tracker segue o mesmo
protocolo:

    1. ao nascer, a track consome a primeira detecção -> já existe uma
       previsão para o próximo quadro (`next_pred`);
    2. a cada quadro, a previsão pendente é comparada com as detecções;
    3. a entrada do PRÓXIMO passo é a observação casada (se houve) ou a
       própria previsão (se a track está ocluída) -- e só então é
       consumida pelo LSTM.

O modo `legacy_state_update=True` reproduz o comportamento anterior, em
que a última caixa era consumida duas vezes (uma ao prever, outra ao
"recalcular" o estado com a observação). Serve só para reproduzir os
números antigos das Partes 2/3.

Extras para a Parte 4: `death_frame` (quadro em que cada track foi
removida por max_age), `birth_frame`, e `run(first_frame=1)` para já
indexar quadros como no MOT (1-based), sem pós-processamento.
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
    # a extrapolação pode produzir largura/altura <= 0; sem o clamp o IoU
    # de uma caixa "invertida" fica sem sentido
    w = max(float(w), 1e-4)
    h = max(float(h), 1e-4)
    left = (cx - w / 2) * img_w
    top = (cy - h / 2) * img_h
    width = w * img_w
    height = h * img_h
    return (top, left, top + height, left + width)


class MotionLSTMTracker:
    def __init__(self, model, device, img_w, img_h, iou_threshold=0.3, max_age=30,
                 legacy_state_update=False):
        self.model = model.to(device).eval()
        self.device = device
        self.img_w = img_w
        self.img_h = img_h
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.legacy_state_update = legacy_state_update

        self.next_id = 0
        self.current_box = {}   # id -> última entrada do LSTM (observada OU extrapolada), norm cxcywh
        self.hidden = {}        # id -> (h, c)
        self.next_pred = {}     # id -> previsão (norm cxcywh) para o quadro seguinte
        self.hidden_before_predict = {}  # só usado no modo legacy
        self.age = {}           # quadros seguidos sem observação
        self.history = {}       # id -> {frame: box pixel yxyx} -- só quadros OBSERVADOS
        # Caixa que a recorrência propôs antes de ver as detecções do quadro
        # (o "mapa intermediário" da Parte 4).
        self.predicted_history = {}
        self.birth_frame = {}   # id -> quadro de nascimento
        self.death_frame = {}   # id -> quadro em que foi removida por max_age

    # ------------------------------------------------------------------
    # primitivas do LSTM
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _consume(self, tid, box):
        """Consome UMA entrada e guarda a previsão do próximo quadro."""
        x = torch.from_numpy(np.asarray(box, dtype=np.float32)).to(self.device)
        pred, hidden = self.model.predict_step(x, self.hidden.get(tid))
        self.hidden[tid] = hidden
        self.next_pred[tid] = pred.cpu().numpy()

    @torch.no_grad()
    def _predict_next(self, tid):
        """Modo legacy: consome current_box e prevê (como na versão antiga)."""
        last_box = torch.from_numpy(self.current_box[tid]).to(self.device)
        pred_position, hidden = self.model.predict_step(last_box, self.hidden[tid])
        self.hidden[tid] = hidden
        return pred_position.cpu().numpy()

    def _kill(self, tid, frame_idx):
        self.death_frame[tid] = frame_idx
        for d in (self.current_box, self.hidden, self.age, self.next_pred):
            d.pop(tid, None)

    # ------------------------------------------------------------------
    def step(self, frame_idx, detections_pixel):
        """detections_pixel: array (N, 4) em (left, top, w, h)"""
        legacy = self.legacy_state_update
        track_ids = list(self.current_box.keys())

        if legacy:
            self.hidden_before_predict = {tid: self.hidden[tid] for tid in track_ids}

        predicted = {}
        for tid in track_ids:
            predicted[tid] = self._predict_next(tid) if legacy else self.next_pred[tid]
            self.predicted_history.setdefault(tid, {})[frame_idx] = \
                norm_cxcywh_to_pixel_yxyx(predicted[tid], self.img_w, self.img_h)

        dets_norm = [pixel_to_norm_cxcywh(d, self.img_w, self.img_h) for d in detections_pixel]

        def iou_norm(a, b):
            return compute_iou(norm_cxcywh_to_pixel_yxyx(a, self.img_w, self.img_h),
                               norm_cxcywh_to_pixel_yxyx(b, self.img_w, self.img_h))

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
                observed = dets_norm[assignments[tid]]
                if legacy:
                    with torch.no_grad():
                        obs_t = torch.from_numpy(observed).to(self.device)
                        _, hidden = self.model.predict_step(obs_t, self.hidden_before_predict[tid])
                        self.hidden[tid] = hidden
                else:
                    self._consume(tid, observed)   # a observação vira a entrada do próximo passo
                self.current_box[tid] = observed
                self.age[tid] = 0
                box_pixel = norm_cxcywh_to_pixel_yxyx(observed, self.img_w, self.img_h)
                self.history.setdefault(tid, {})[frame_idx] = box_pixel
                current_frame_output[tid] = box_pixel
            else:
                self.age[tid] += 1
                if self.age[tid] > self.max_age:
                    self._kill(tid, frame_idx)
                    continue
                # ocluída: a própria previsão é a entrada do próximo passo
                self.current_box[tid] = predicted[tid]
                if not legacy:
                    self._consume(tid, predicted[tid])

        for j, det in enumerate(dets_norm):
            if j not in matched_dets:
                tid = self.next_id
                self.next_id += 1
                self.current_box[tid] = det
                self.hidden[tid] = None
                self.age[tid] = 0
                self.birth_frame[tid] = frame_idx
                if not legacy:
                    self._consume(tid, det)
                box_pixel = norm_cxcywh_to_pixel_yxyx(det, self.img_w, self.img_h)
                self.history.setdefault(tid, {})[frame_idx] = box_pixel
                current_frame_output[tid] = box_pixel

        return current_frame_output

    def run(self, detections_by_frame, first_frame=0):
        """first_frame=0 mantém a convenção antiga (0-based); use 1 para MOT17."""
        for i, dets in enumerate(detections_by_frame):
            self.step(first_frame + i, dets)
        return self.history
