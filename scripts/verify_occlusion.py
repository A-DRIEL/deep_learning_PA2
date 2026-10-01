# scripts/verify_occlusion.py
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src.datasets.synthetic_video import generate_video

Path("outputs").mkdir(exist_ok=True)

rng = np.random.default_rng(42)
frames, boxes, visibility, on_screen, objects = generate_video(
    rng, n_frames=45, n_objects=8, occlusion_duration=8
)

occluded_id = 0
vis = np.array(visibility[occluded_id])
onscreen = np.array(on_screen[occluded_id])

# só conta como "oclusão de verdade" quadros em que o objeto ESTÁ na tela
# mas está coberto por outro -- não confundir com "fora da tela"
occluded_frames = np.sum((vis < 0.1) & onscreen)
offscreen_frames = np.sum(~onscreen)

print(f"Quadros fora da tela: {offscreen_frames}")
print(f"Quadros ocluído (visibilidade<0.1, na tela): {occluded_frames}")
print(f"Visibilidade mínima (só quadros na tela): {vis[onscreen].min():.3f}")

fig, axes = plt.subplots(2, 1, figsize=(10, 6), height_ratios=[1, 2])
axes[0].plot(vis, color="#E4572E", linewidth=2)
axes[0].axhline(0.1, color="gray", linestyle="--", linewidth=1)
axes[0].set_xlabel("Quadro")
axes[0].set_ylabel("Fração visível")
axes[0].set_title(f"Objeto {occluded_id}: visibilidade ao longo do vídeo")

mid = len(frames) // 2
frame_idxs = list(range(mid - 8, mid + 9, 2))
n_shown = len(frame_idxs)
for i, t in enumerate(frame_idxs):
    ax = fig.add_axes([0.05 + i * 0.9 / n_shown, 0.02, 0.85 / n_shown, 0.35])
    ax.imshow(frames[t], cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"t={t}\nvis={vis[t]:.2f}", fontsize=8)
    ax.axis("off")

plt.savefig("outputs/occlusion_verification.png", dpi=150, bbox_inches="tight")