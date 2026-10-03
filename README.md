# PA2 — Identidade ao longo do tempo: detecção, recorrência e rastreamento

Aprendizado Profundo · Prof. Dario Oliveira · Dupla: Adriel Dias e Gabriel Schuenker

Um LSTM por track como **modelo de movimento** (Trilha A) sobre as detecções públicas do MOT17. Ele prevê a caixa do quadro seguinte, a associação usa IoU entre a caixa prevista e a observada, e o estado continua rodando sob oclusão. Comparamos com um baseline ingênuo de IoU, ablamos o regime de treino (Eixo 2), medimos o horizonte de memória (Parte 4) e estressamos a qualidade do detector (Parte 5).

**Nenhum rastreador ou métrica de biblioteca é usado.** Associação, gestão de tracks, perdas, IDF1, ID switches e fragmentações são de nossa autoria.

## Entregáveis

| Item | Onde |
|---|---|
| Ambiente, dados, treino e avaliação | este README |
| IDF1, ID switches, fragmentações | [`metrics.py`](metrics.py) (`python metrics.py` roda os testes feitos à mão) |
| Uso de IA | [`AI_LOG.md`](AI_LOG.md) |
| Vídeo com identidades coloridas + contagem de objetos únicos | [`inferencia.ipynb`](inferencia.ipynb) (roda sem retreinar) |
| Checkpoint do modelo temporal | `outputs/motion_lstm.pt` (ver [Checkpoint](#checkpoint)) |

```
.
├── metrics.py  inferencia.ipynb  AI_LOG.md  README.md  pyproject.toml
├── src/
│   ├── datasets/   mot17.py, synthetic_video.py, detector_simulator.py,
│   │               detector_degradation.py, trajectory_dataset.py
│   ├── models/     motion_lstm.py
│   ├── tracking/   naive_tracker.py, motion_lstm_tracker.py, runners.py
│   ├── metrics/    tracking_metrics.py, detection_metrics.py
│   ├── analysis/   find_critical_moments.py, memory_horizon.py
│   └── viz/        plot_style.py
├── scripts/        um script por experimento (+ scripts/tests/)
├── outputs/        figuras, JSONs e checkpoints gerados
└── data/raw/       MOT17 (não versionado)
```

## 1. Ambiente

Python ≥ 3.11. Usamos [uv](https://docs.astral.sh/uv/); os extras `cpu` e `gpu` são mutuamente exclusivos.

```bash
uv sync --extra cpu          # só CPU
# uv sync --extra gpu        # GPU (CUDA 13.0, índice pytorch-cu130)

source .venv/bin/activate
uv pip install jupyterlab    # só para abrir o inferencia.ipynb
```

Sem uv: `python -m venv .venv && source .venv/bin/activate && pip install -e . jupyterlab`.

**Todos os comandos abaixo rodam da raiz do repositório**, com `python -m` (os scripts importam `src.*`).

## 2. Dados (MOT17)

```bash
mkdir -p data/raw && cd data/raw
curl -L -O https://motchallenge.net/data/MOT17.zip     # ~5,5 GB: imagens + det + gt
unzip MOT17.zip -d MOT17
cd ../..

ls data/raw/MOT17/train      # deve listar MOT17-02-DPM ... MOT17-13-SDP
```

Cada pasta tem `img1/`, `det/det.txt`, `gt/gt.txt` e `seqinfo.ini`. Se o `unzip` criar um nível extra (`MOT17/MOT17/train`), mova a pasta `train/` para `data/raw/MOT17/`.

Para treinar e calcular métricas basta o pacote **só de anotações** (~10 MB, `https://motchallenge.net/data/MOT17Labels.zip`, mesma estrutura). As imagens só são necessárias para as figuras com quadros (`visualize_part2_cases`, `compare_lockin_across_regimes`, Parte 4) e para o vídeo do notebook.

## 3. Treinar e avaliar (um comando cada)

```bash
# TREINA o LSTM de movimento (Trilha A) → outputs/motion_lstm.pt
python -m scripts.train_motion_lstm

# AVALIA: baseline ingênuo vs. Trilha A nas 7 sequências (IDF1, ID switches, fragmentações)
python -m scripts.run_part2_trilha_a
```

- **Treino:** janelas de 8 quadros, smooth-L1, Adam (lr 1e-3), batch 128, 50 épocas. **MOT17-09 é reservada**: nunca entra no treino e serve de validação e de held-out da Parte 5.
- **Avaliação:** imprime uma linha por sequência (baseline | Trilha A) e os agregados. Lembre que MOT17-09 é a única sequência que o checkpoint nunca viu.
- Para pular o treino, use o checkpoint versionado em `outputs/motion_lstm.pt`.

## 4. Inferência em qualquer sequência (sem retreinar)

Abra `inferencia.ipynb`, edite `SEQ_PATH` na primeira célula e rode tudo:

```python
SEQ_PATH = "data/raw/MOT17/train/MOT17-09-SDP"   # ou uma sequência de teste, uma pasta de imagens, ou um .mp4
```

O notebook gera em `outputs/inference/`:

- `<seq>_identities.mp4`: vídeo com cada ID numa cor fixa (a cor depende só do ID);
- `<seq>_tracks.txt`: tracks no formato MOTChallenge;
- `<seq>_summary.json` e `<seq>_counts.png`: contagem de objetos únicos e, se houver `gt.txt`, IDF1, switches, fragmentações e erro de contagem.

Pastas no formato MOT17 usam o `det.txt`. Pastas de imagens ou vídeos usam o Faster R-CNN do torchvision, ligado por `USE_TORCHVISION_DETECTOR`, com NMS próprio. Essa opção baixa pesos pré-treinados na primeira execução.

## 5. Reproduzindo cada figura e tabela

Dependências entre etapas: Parte 2 → `outputs/motion_lstm.pt`; Parte 3 → `outputs/ablation_checkpoints/`; Partes 4 e 5 usam checkpoints das etapas anteriores (indicado abaixo).

| Parte | Comando | Produz |
|---|---|---|
| **0** · métricas (casos a, b, c) | `python -m scripts.tests.test_tracking_metrics` ou `python metrics.py` | saída no terminal |
| 0 · oclusão real | `python -m scripts.verify_occlusion` | `outputs/occlusion_verification.png` |
| 0 · simulador de detector | `python -m scripts.tests.test_detector_simulator` | saída no terminal |
| 0 · piso fácil (IDF1 ≈ 1) | `python -m scripts.tests.test_easy_floor` | saída no terminal |
| 0 · onde o baseline quebra | `python -m scripts.stress_synthetic_baseline` | `outputs/synthetic_stress_baseline.png` |
| **1** · escolha do detector | `python -m scripts.compare_detectors` | recall e precisão de DPM, FRCNN e SDP |
| 1 · baseline no MOT17 | `python -m scripts.run_part1_baseline` | tabela no terminal |
| 1 · gráfico do descolamento | `python -m scripts.part1_difficulty_plot` | `outputs/part1_difficulty_plot.png` |
| **2** · Trilha A | `python -m scripts.run_part2_trilha_a` | tabela baseline vs. Trilha A |
| 2 · sanidade (LSTM vs. "copiar última posição") | `python -m scripts.sanity_check_motion_lstm` | saída no terminal |
| 2 · casos de sucesso e falha | `python -m scripts.visualize_part2_cases` | `outputs/part2_case_{success,failure}.png` |
| **3** · ablação, Eixo 2 (3 regimes × 3 seeds + sem clipping) | `python -m scripts.train_motion_lstm_ablation` | `outputs/ablation_checkpoints/*.pt`, `outputs/ablation_eixo2_results.json` |
| 3 · curvas e clipping | `python -m scripts.analyze_ablation_eixo2` | `outputs/ablation_eixo2_curves.png`, `outputs/ablation_clipping_comparison.png` |
| 3 · IDF1 e switches no tracker | `python -m scripts.run_part3_eixo2_tracking` | `outputs/part3_eixo2_tracking_results.json`, `outputs/part3_eixo2_tracking_comparison.png` |
| 3 · versão por sequência da figura | `python -m scripts.replot_part3_eixo2` | sobrescreve a figura acima (lê o JSON) |
| 3 · lock-in no mesmo caso (MOT17-04, gt 3) | `python -m scripts.compare_lockin_across_regimes` | `outputs/part3_lockin_comparison.png` |
| **4** · galeria de falhas e horizonte de memória | `python -m scripts.run_part4_memory_analysis` | `outputs/part4/` (`gradient_horizon.png`, `memory_survival.png`, `failure_*.png`, JSON) |
| 4 · correção (antes/depois) | `python -m scripts.run_part4_correction` | `outputs/part4/correction_*.{png,json}` |
| **5** · estresse do detector | `python -m scripts.tests.test_part5_components` | testes do degradador e do mAP |
| 5 · rodar | `python -m scripts.run_part5_detector_stress --checkpoint outputs/motion_lstm.pt --seeds 0 1 2 --heldout 09 --isolate` | `outputs/part5_detector_stress.json` (retoma se interrompido) |
| 5 · analisar | `python -m scripts.analyze_part5` | `outputs/part5_main.png`, `outputs/part5_per_sequence.png`, `outputs/part5_table.md` |

## 6. Decisões de projeto

| Decisão | Escolha |
|---|---|
| Fonte de detecções | **SDP**, escolhida comparando recall e precisão dos três detectores públicos nas 7 sequências (`scripts/compare_detectors.py`) |
| Split | Por sequência. **MOT17-09 (densidade 10,1 caixas/quadro) fica fora do treino.** A justificativa completa está na apresentação |
| Associação (baseline) | Matching guloso por IoU decrescente entre detecções do quadro *t* e tracks vivas. Limiar fixo 0,3. ID novo quando nada casa. Track morre após `max_age = 30` quadros sem observação (1 s a 30 fps) (`src/tracking/naive_tracker.py`) |
| Trilha A | Estado da track = caixa (cx, cy, w, h) normalizada. LSTM de 1 camada e 64 unidades, um estado oculto por track, com cabeça linear que prevê o **delta** sobre a última posição (atalho residual). Perda smooth-L1. Associação por IoU entre caixa prevista e observada, com os mesmos 0,3 e `max_age = 30` do baseline. Sob oclusão, a própria previsão é a entrada do passo seguinte. Cada caixa entra no LSTM **uma única vez**, como no treino (`src/models/motion_lstm.py`, `src/tracking/motion_lstm_tracker.py`) |
| Eixo de dificuldade (Parte 1) | Densidade (caixas/quadro). Correlaciona com a razão de contagem em 5 das 7 sequências; MOT17-02 e MOT17-04 são exceções documentadas, não escondidas |
| Escolhas do enunciado | Trilha **A** · Eixo **2** da ablação (teacher forcing → scheduled sampling → free-running, com e sem gradient clipping) · Parte 5: **qualidade do detector** |
| Métricas | `metrics.py`: IDF1 via Húngaro (atribuição global), switches só entre quadros consecutivos, fragmentações separadas dos switches. As definições exatas estão no topo do arquivo |

`metrics.py` produz os mesmos resultados que `src/metrics/tracking_metrics.py` (usado pelos scripts), mas indexado por quadro, então é bem mais rápido.

## Checkpoint

`outputs/motion_lstm.pt` é o modelo da Trilha A (LSTM 4→64 + cabeça linear, ≈ 18 mil parâmetros, < 100 KB). Ele está **versionado no repositório** (exceção no `.gitignore`), então o notebook roda logo após o `git clone`. Os checkpoints da ablação (`outputs/ablation_checkpoints/{regime}_seed{0,1,2}.pt`) são regenerados por `scripts.train_motion_lstm_ablation`.