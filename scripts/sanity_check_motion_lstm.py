# scripts/sanity_check_motion_lstm.py
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from src.datasets.mot17 import MOT17Sequence, SEQUENCE_IDS
from src.datasets.trajectory_dataset import extract_trajectories, TrajectoryWindowDataset
from src.models.motion_lstm import MotionLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root = Path("data/raw/MOT17/train")
sequences_info = []
for seq_id in SEQUENCE_IDS:
    seq = MOT17Sequence(root / f"MOT17-{seq_id}-SDP", load_gt=True)
    if seq.gt is not None:
        sequences_info.append((seq.info.name, seq.gt, seq.info.im_width, seq.info.im_height))

trajectories = extract_trajectories(sequences_info)
dataset = TrajectoryWindowDataset(trajectories, window_size=8)
loader = DataLoader(dataset, batch_size=256, shuffle=False)

model = MotionLSTM(input_dim=4, hidden_dim=64).to(device)
model.load_state_dict(torch.load("outputs/motion_lstm.pt", map_location=device))
model.eval()

total_lstm_err, total_trivial_err, n = 0.0, 0.0, 0
with torch.no_grad():
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        pred_position, pred_delta, _ = model(inputs)

        # baseline trivial: "próxima posição = última observada" (velocidade zero)
        trivial_pred = inputs

        lstm_err = (pred_position - targets).abs().mean().item()
        trivial_err = (trivial_pred - targets).abs().mean().item()

        total_lstm_err += lstm_err * inputs.size(0)
        total_trivial_err += trivial_err * inputs.size(0)
        n += inputs.size(0)

print(f"Erro médio absoluto -- LSTM: {total_lstm_err/n:.8f}")
print(f"Erro médio absoluto -- baseline trivial (copia última posição): {total_trivial_err/n:.8f}")
print(f"\nLSTM é {'melhor' if total_lstm_err < total_trivial_err else 'PIOR OU IGUAL'} que o trivial")