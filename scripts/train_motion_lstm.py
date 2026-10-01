# scripts/train_motion_lstm.py
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.datasets.mot17 import MOT17Sequence, SEQUENCE_IDS
from src.datasets.trajectory_dataset import extract_trajectories, TrajectoryWindowDataset
from src.models.motion_lstm import MotionLSTM

Path("outputs").mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

root = Path("data/raw/MOT17/train")
DETECTOR = "SDP"

sequences_info = []
for seq_id in SEQUENCE_IDS:
    seq = MOT17Sequence(root / f"MOT17-{seq_id}-{DETECTOR}", load_gt=True)
    if seq.gt is not None:
        sequences_info.append((seq.info.name, seq.gt, seq.info.im_width, seq.info.im_height))

trajectories = extract_trajectories(sequences_info)
dataset = TrajectoryWindowDataset(trajectories, window_size=8)

n_val = int(0.1 * len(dataset))
train_set, val_set = random_split(dataset, [len(dataset) - n_val, n_val])
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
val_loader = DataLoader(val_set, batch_size=128)

model = MotionLSTM(input_dim=4, hidden_dim=64, num_layers=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.SmoothL1Loss()

n_epochs = 50
for epoch in range(n_epochs):
    model.train()
    train_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        pred_position, pred_delta, _ = model(inputs)
        loss = criterion(pred_position, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
    train_loss /= len(train_set)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            pred_position, pred_delta, _ = model(inputs)  # <- corrigido aqui
            val_loss += criterion(pred_position, targets).item() * inputs.size(0)
    val_loss /= len(val_set)

    print(f"Época {epoch+1}/{n_epochs}: train_loss={train_loss:.5f}, val_loss={val_loss:.5f}")

torch.save(model.state_dict(), "outputs/motion_lstm.pt")
print("\nModelo salvo em outputs/motion_lstm.pt")