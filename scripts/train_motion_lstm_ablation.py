"""
Parte 3, Eixo 2: teacher forcing -> scheduled sampling -> free-running,
3 seeds cada, + comparação com/sem gradient clipping.

Treino passo-a-passo dentro de cada janela (não mais um forward único):
em cada passo t, decide se a entrada do próximo passo vem do dado real
(window[t+1]) ou da própria previsão do modelo -- dependendo do regime.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.datasets.mot17 import MOT17Sequence, SEQUENCE_IDS
from src.datasets.trajectory_dataset import extract_trajectories, TrajectoryWindowDataset
from src.models.motion_lstm import MotionLSTM

Path("outputs").mkdir(exist_ok=True)
Path("outputs/ablation_checkpoints").mkdir(exist_ok=True)

WINDOW_SIZE = 8
N_EPOCHS = 30
BATCH_SIZE = 128


def sampling_probability(regime, epoch, n_epochs):
    """Probabilidade de usar a PRÓPRIA previsão (em vez da observação real)
    como entrada do próximo passo, dado o regime e o progresso do treino."""
    if regime == "teacher_forcing":
        return 0.0
    if regime == "free_running":
        return 1.0
    if regime == "scheduled_sampling":
        return min(1.0, epoch / (n_epochs * 0.7))  # sobe linear, satura em 70% do treino
    raise ValueError(regime)


def train_one_config(regime, seed, train_set, val_set, device, use_grad_clip=True,
                      n_epochs=N_EPOCHS):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = MotionLSTM(input_dim=4, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.SmoothL1Loss()

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    history = {"train_loss": [], "val_loss": [], "grad_norm": []}

    for epoch in range(n_epochs):
        model.train()
        p_own = sampling_probability(regime, epoch, n_epochs)
        epoch_loss, epoch_grad_norm, n_batches = 0.0, 0.0, 0

        for windows, _ in train_loader:
            # windows: (batch, window_size+1, 4) -- vem do mesmo TrajectoryWindowDataset,
            # mas aqui usamos a janela INTEIRA (inputs+targets concatenados) para controlar
            # passo a passo; ignoramos o split automático do dataset (_) e refazemos manualmente
            full_window = torch.cat([windows, _], dim=1)[:, :WINDOW_SIZE + 1, :] \
                if False else None  # placeholder, substituído abaixo

        # --- loop manual, reconstruindo a janela completa a partir do dataset original ---
        for batch in train_loader:
            inputs, targets = batch  # inputs=(B,T,4) todas reais; targets=(B,T,4) deslocado +1
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)

            optimizer.zero_grad()
            hidden = None
            current_input = inputs[:, 0:1, :]  # primeiro passo SEMPRE observação real
            total_loss = 0.0

            for t in range(WINDOW_SIZE):
                pred_position, pred_delta, hidden = model(current_input, hidden)
                target_t = targets[:, t:t+1, :]
                step_loss = criterion(pred_position, target_t)
                total_loss = total_loss + step_loss

                use_own = (torch.rand(1).item() < p_own) and (t < WINDOW_SIZE - 1)
                if use_own:
                    current_input = pred_position.detach()  # própria previsão (sem propagar grad por aqui)
                else:
                    if t < WINDOW_SIZE - 1:
                        current_input = inputs[:, t+1:t+2, :]  # próxima observação real

            total_loss = total_loss / WINDOW_SIZE
            total_loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0 if use_grad_clip else float("inf")
            )
            optimizer.step()

            epoch_loss += total_loss.item() * batch_size
            epoch_grad_norm += grad_norm.item() * batch_size
            n_batches += batch_size

        train_loss = epoch_loss / n_batches
        mean_grad_norm = epoch_grad_norm / n_batches

        model.eval()
        val_loss, n_val = 0.0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                pred_position, _, _ = model(inputs)
                val_loss += criterion(pred_position, targets).item() * inputs.size(0)
                n_val += inputs.size(0)
        val_loss /= n_val

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["grad_norm"].append(mean_grad_norm)

        if np.isnan(train_loss) or np.isnan(mean_grad_norm):
            print(f"  [{regime}, seed={seed}, clip={use_grad_clip}] NaN na época {epoch} -- parando")
            break

    return model, history


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    root = Path("data/raw/MOT17/train")
    sequences_info = []
    for seq_id in SEQUENCE_IDS:
        seq = MOT17Sequence(root / f"MOT17-{seq_id}-SDP", load_gt=True)
        if seq.gt is not None:
            sequences_info.append((seq.info.name, seq.gt, seq.info.im_width, seq.info.im_height))

    trajectories = extract_trajectories(sequences_info)
    dataset = TrajectoryWindowDataset(trajectories, window_size=WINDOW_SIZE)
    n_val = int(0.1 * len(dataset))
    train_set, val_set = random_split(dataset, [len(dataset) - n_val, n_val])

    all_results = {}

    # --- experimento principal: 3 regimes x 3 seeds, clipping LIGADO ---
    for regime in ["teacher_forcing", "scheduled_sampling", "free_running"]:
        all_results[regime] = []
        for seed in [0, 1, 2]:
            print(f"Treinando regime={regime}, seed={seed}, clip=True...")
            model, history = train_one_config(regime, seed, train_set, val_set, device, use_grad_clip=True)
            final_val = history["val_loss"][-1] if history["val_loss"] else float("nan")
            print(f"  val_loss final: {final_val:.6f}")
            torch.save(model.state_dict(), f"outputs/ablation_checkpoints/{regime}_seed{seed}.pt")
            all_results[regime].append({"seed": seed, "history": history})

    # --- comparação extra: teacher_forcing SEM clipping ---
    all_results["teacher_forcing_no_clip"] = []
    for seed in [0, 1, 2]:
        print(f"Treinando regime=teacher_forcing, seed={seed}, clip=False...")
        model, history = train_one_config("teacher_forcing", seed, train_set, val_set, device, use_grad_clip=False)
        final_val = history["val_loss"][-1] if history["val_loss"] else float("nan")
        print(f"  val_loss final: {final_val:.6f}")
        all_results["teacher_forcing_no_clip"].append({"seed": seed, "history": history})

    with open("outputs/ablation_eixo2_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nResultados salvos em outputs/ablation_eixo2_results.json")


if __name__ == "__main__":
    main()