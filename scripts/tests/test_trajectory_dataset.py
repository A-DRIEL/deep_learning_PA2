# scripts/test_trajectory_dataset.py
from pathlib import Path

from src.datasets.mot17 import MOT17Sequence, SEQUENCE_IDS
from src.datasets.trajectory_dataset import extract_trajectories, TrajectoryWindowDataset

root = Path("data/raw/MOT17/train")
DETECTOR = "SDP"

sequences_info = []
for seq_id in SEQUENCE_IDS:
    seq = MOT17Sequence(root / f"MOT17-{seq_id}-{DETECTOR}", load_gt=True)
    if seq.gt is not None:
        sequences_info.append((seq.info.name, seq.gt, seq.info.im_width, seq.info.im_height))

trajectories = extract_trajectories(sequences_info)
print(f"Total de segmentos de trajetória contínua: {len(trajectories)}")
print(f"Comprimento médio: {sum(len(t) for t in trajectories) / len(trajectories):.1f} quadros")
print(f"Comprimento mínimo/máximo: {min(len(t) for t in trajectories)} / {max(len(t) for t in trajectories)}")

dataset = TrajectoryWindowDataset(trajectories, window_size=8)
print(f"\nJanelas de treino (window_size=8): {len(dataset)}")

inputs, targets = dataset[0]
print(f"Shape de uma amostra: inputs={inputs.shape}, targets={targets.shape}")
print(f"Exemplo de input[0]: {inputs[0]}")