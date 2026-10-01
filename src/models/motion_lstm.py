"""
Modelo de movimento (Trilha A): LSTM que prevê o DESLOCAMENTO (delta)
em relação à última posição observada, não a posição absoluta.

"""

import torch
import torch.nn as nn


class MotionLSTM(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, hidden=None):
        """
        x: (batch, seq_len, 4) -- histórico de posições observadas
        Devolve:
            pred_position: (batch, seq_len, 4) -- posição absoluta prevista
                            (x + delta previsto, residual)
            pred_delta:    (batch, seq_len, 4) -- o delta em si (útil pra loss)
        """
        out, hidden = self.lstm(x, hidden)
        pred_delta = self.head(out)
        pred_position = x + pred_delta  # <- o atalho residual
        return pred_position, pred_delta, hidden

    def predict_step(self, last_box, hidden):
        x = last_box.unsqueeze(0).unsqueeze(0)
        out, hidden = self.lstm(x, hidden)
        delta = self.head(out)
        pred_position = x + delta
        return pred_position.squeeze(0).squeeze(0), hidden