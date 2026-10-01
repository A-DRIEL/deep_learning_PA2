"""
Parser para sequências do MOT17.

Formato do gt.txt / det.txt (uma linha por detecção/anotação):
    frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility

- gt.txt: id é o identificador real da trajetória; conf=1 considera a caixa,
  conf=0 marca "ignorar" (não conta pra avaliação); class=1 é pedestre
  (outras classes são distratores: pessoa em veículo, reflexo, etc.);
  visibility é a fração visível do objeto (0 a 1) -- ouro pra Parte 4.
- det.txt: id sempre -1 (não tem identidade ainda, é detecção crua);
  conf aqui é a confiança do detector, não um flag binário.

As 7 sequências únicas do MOT17-train, cada uma com 3 conjuntos de
detecções (DPM, FRCNN, SDP) sobre o MESMO vídeo/gt:
  02, 04, 05, 09, 10, 11, 13
"""

import configparser
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SEQUENCE_IDS = ["02", "04", "05", "09", "10", "11", "13"]
DETECTOR_NAMES = ["DPM", "FRCNN", "SDP"]


@dataclass
class SequenceInfo:
    name: str
    frame_rate: int
    seq_length: int
    im_width: int
    im_height: int
    im_dir: str
    im_ext: str


def read_seqinfo(seq_path: Path) -> SequenceInfo:
    config = configparser.ConfigParser()
    config.read(seq_path / "seqinfo.ini")
    s = config["Sequence"]
    return SequenceInfo(
        name=s["name"],
        frame_rate=int(s["frameRate"]),
        seq_length=int(s["seqLength"]),
        im_width=int(s["imWidth"]),
        im_height=int(s["imHeight"]),
        im_dir=s["imDir"],
        im_ext=s["imExt"],
    )


def read_annotations(txt_path: Path) -> np.ndarray:
    """
    Lê gt.txt ou det.txt e devolve um array (N, 9):
        [frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility]

    Para det.txt, 'class' e 'visibility' costumam vir como -1 (não se aplicam);
    mantemos as 9 colunas por consistência, o chamador ignora o que não usa.
    """
    if not txt_path.exists():
        raise FileNotFoundError(f"Arquivo de anotação não encontrado: {txt_path}")

    data = np.loadtxt(txt_path, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)  # sequência com uma única linha, caso raro
    return data


def filter_gt_for_evaluation(gt: np.ndarray, min_visibility: float = 0.0) -> np.ndarray:
    """
    Aplica os dois filtros padrão do MOT17 antes de usar o gt pra treino/avaliação:
      - conf == 1 (descarta linhas marcadas como 'ignorar')
      - class == 1 (só pedestre; descarta distratores)
    min_visibility permite também descartar objetos muito ocluídos, se vocês
    quiserem estudar isso separadamente (ex.: Parte 4, usando o campo visibility).
    """
    conf, cls, vis = gt[:, 6], gt[:, 7], gt[:, 8]
    mask = (conf == 1) & (cls == 1) & (vis >= min_visibility)
    return gt[mask]


class MOT17Sequence:
    """Uma sequência específica (ex.: MOT17-02-FRCNN), com acesso por quadro."""

    def __init__(self, seq_path: Path, load_gt: bool = True):
        self.seq_path = Path(seq_path)
        self.info = read_seqinfo(self.seq_path)

        self.det = read_annotations(self.seq_path / "det" / "det.txt")

        self.gt = None
        if load_gt:
            gt_path = self.seq_path / "gt" / "gt.txt"
            if gt_path.exists():
                self.gt = filter_gt_for_evaluation(read_annotations(gt_path))
            # sequências de teste não têm gt.txt público -- self.gt fica None

    def frame_path(self, frame_idx: int) -> Path:
        """frame_idx é 1-based, como no MOT (primeiro quadro = 1, não 0)."""
        filename = f"{frame_idx:06d}{self.info.im_ext}"
        return self.seq_path / self.info.im_dir / filename

    def detections_at(self, frame_idx: int) -> np.ndarray:
        """Devolve as detecções (sem id de trajetória) daquele quadro."""
        rows = self.det[self.det[:, 0] == frame_idx]
        return rows[:, 2:6]  # bb_left, bb_top, bb_width, bb_height

    def gt_at(self, frame_idx: int) -> np.ndarray:
        """Devolve [id, bb_left, bb_top, bb_width, bb_height] do gabarito naquele quadro."""
        if self.gt is None:
            raise ValueError(f"{self.info.name} não tem gt.txt (é sequência de teste?)")
        rows = self.gt[self.gt[:, 0] == frame_idx]
        return rows[:, [1, 2, 3, 4, 5]]

    def __len__(self):
        return self.info.seq_length


def list_available_sequences(root: Path, detector: str = "FRCNN") -> list[str]:
    """Lista as sequências disponíveis pra um detector específico (ex.: 'FRCNN')."""
    root = Path(root)
    return sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and p.name.endswith(f"-{detector}")
    )