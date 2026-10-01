"""
Simulador de detector imperfeito para a Parte 0.

Pega caixas verdadeiras e as estraga de propósito, de três formas
independentes e combináveis:
  - descarta uma fração delas (falso negativo)
  - adiciona ruído gaussiano nas coordenadas (imprecisão de localização)
  - injeta caixas falsas que não correspondem a nada real (falso positivo)

Isso permite testar a Parte 1 inteira no sintético, sem precisar do
MOT17 nem de um detector de verdade -- é literalmente o experimento
que a Parte 5 (teste de estresse) também usa, então essa função é
reaproveitada mais adiante no PA.
"""

import numpy as np


def simulate_detections(true_boxes, rng, drop_prob=0.1, noise_std=2.0,
                         false_positive_rate=0.1, image_size=128):
    """
    true_boxes: lista de (y0, x0, y1, x1) -- caixas reais nesse quadro
                (None é permitido e ignorado -- objeto fora da tela)
    drop_prob: probabilidade de descartar cada caixa individualmente
    noise_std: desvio padrão (em pixels) do ruído somado a cada coordenada
    false_positive_rate: probabilidade de injetar UMA caixa falsa nesse quadro
                          (não escala com o número de objetos reais -- é por
                          quadro, o jeito mais simples de controlar)
    image_size: tamanho da imagem, usado para gerar falsos positivos
                dentro dos limites válidos

    Devolve: array (N, 4) de caixas (y0, x0, y1, x1), N variável por quadro
    """
    kept = []
    for box in true_boxes:
        if box is None:
            continue
        if rng.random() < drop_prob:
            continue  # falso negativo: descarta essa caixa
        y0, x0, y1, x1 = box
        noise = rng.normal(0, noise_std, size=4)
        kept.append((y0 + noise[0], x0 + noise[1], y1 + noise[2], x1 + noise[3]))

    if rng.random() < false_positive_rate:
        h = rng.uniform(8, 20)
        w = rng.uniform(8, 20)
        y0 = rng.uniform(0, image_size - h)
        x0 = rng.uniform(0, image_size - w)
        kept.append((y0, x0, y0 + h, x0 + w))

    if len(kept) == 0:
        return np.zeros((0, 4))
    return np.array(kept)