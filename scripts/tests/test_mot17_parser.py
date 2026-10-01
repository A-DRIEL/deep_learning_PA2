# scripts/test_mot17_parser.py
from pathlib import Path
from src.datasets.mot17 import MOT17Sequence, list_available_sequences

root = Path("data/raw/MOT17/train")
print("Sequências FRCNN disponíveis:", list_available_sequences(root, "FRCNN"))

seq = MOT17Sequence(root / "MOT17-02-FRCNN")
print(f"\n{seq.info.name}: {seq.info.seq_length} quadros, {seq.info.im_width}x{seq.info.im_height}, {seq.info.frame_rate}fps")
print(f"Detecções no quadro 1: {len(seq.detections_at(1))}")
print(f"GT no quadro 1: {len(seq.gt_at(1))} objetos, IDs: {seq.gt_at(1)[:, 0].astype(int)}")