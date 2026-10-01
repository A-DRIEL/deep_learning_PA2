"""
Estilo visual compartilhado para todas as figuras do PA2 -- evita
repetir configuração de matplotlib em cada script, e garante
consistência visual entre todas as figuras da apresentação.
"""

import matplotlib.pyplot as plt

PALETTE = {
    "teal": "#128C94",
    "coral": "#E4572E",
    "green": "#2E8B57",
    "muted": "#5B6B70",
    "bg_light": "#F2F7F7",
}


def apply_style():
    """Chama uma vez, no início do script, antes de criar qualquer figura."""
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25
    plt.rcParams["grid.linewidth"] = 0.6
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["legend.frameon"] = False