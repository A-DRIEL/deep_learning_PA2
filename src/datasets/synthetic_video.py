"""
Gerador de vídeo sintético com elipses em movimento e oclusão real
(ordem de profundidade fixa -- quem tem depth maior sempre fica na
frente, do primeiro ao último quadro).
"""

import numpy as np
from skimage.draw import ellipse as draw_ellipse


def generate_video(rng, size=128, n_frames=45, n_objects=8,
                    typical_speed=2.0, occlusion_duration=8):
    H = W = size
    objects = []
    for i in range(n_objects):
        r_major = rng.uniform(6, 14)
        r_minor = rng.uniform(4, 10)
        intensity = rng.uniform(0.55, 0.95)
        cy = rng.uniform(20, H - 20)
        cx = rng.uniform(20, W - 20)
        angle = rng.uniform(0, 2 * np.pi)
        speed = rng.uniform(0.5, 1.5) * typical_speed
        objects.append({
            "id": i, "r_major": r_major, "r_minor": r_minor,
            "intensity": intensity, "cy0": cy, "cx0": cx,
            "vy": speed * np.sin(angle), "vx": speed * np.cos(angle),
            "depth": i,
        })

    occluded, occluder = objects[0], objects[1]

    margin = 15
    total_travel = (W - 2 * margin)
    v = total_travel / (n_frames - 1)
    occluded_r_minor = 6.0
    occluded_r_major = 8.0
    occluded.update(
        cy0=H / 2, vy=0.0, vx=v, cx0=margin,
        r_major=occluded_r_major, r_minor=occluded_r_minor,
    )

    mid_frame = n_frames // 2
    occluded_x_at_mid = occluded["cx0"] + v * mid_frame

    if occlusion_duration <= 0:
        # sem oclusão de verdade: posiciona o "oclusor" fora do caminho,
        # bem longe, pra não interagir em nenhum pixel com o objeto 0
        occluder.update(cy0=10.0, cx0=10.0, vy=0.0, vx=0.0, r_major=3.0, r_minor=3.0)
    else:
        # oclusor precisa ser MAIOR que o ocluído nos dois eixos, com a folga
        # calibrada para que "estar dentro dessa folga horizontal" dure
        # exatamente occlusion_duration quadros
        extra_margin = (v * occlusion_duration) / 2.0
        occluder_r_minor = occluded_r_minor + extra_margin
        occluder_r_major = occluded_r_major + 4.0  # folga vertical fixa, só para garantir cobertura total
        occluder.update(
            cy0=H / 2, cx0=occluded_x_at_mid, vy=0.0, vx=0.0,
            r_major=occluder_r_major, r_minor=occluder_r_minor,
        )

    background_level = rng.uniform(0.05, 0.2)
    frames = np.zeros((n_frames, H, W), dtype=np.float32)
    boxes = {o["id"]: [] for o in objects}
    visibility = {o["id"]: [] for o in objects}
    on_screen = {o["id"]: [] for o in objects}
    ordered = sorted(objects, key=lambda o: o["depth"])

    for t in range(n_frames):
        pixel_sets = {}
        for obj in objects:
            cy = obj["cy0"] + obj["vy"] * t
            cx = obj["cx0"] + obj["vx"] * t
            rr, cc = draw_ellipse(cy, cx, obj["r_major"], obj["r_minor"], shape=(H, W))
            pixel_sets[obj["id"]] = (rr, cc)

        owner = np.full((H, W), -1, dtype=np.int32)
        for obj in ordered:
            rr, cc = pixel_sets[obj["id"]]
            owner[rr, cc] = obj["id"]

        frame = np.full((H, W), background_level, dtype=np.float32)
        for obj in ordered:
            rr, cc = pixel_sets[obj["id"]]
            frame[rr, cc] = obj["intensity"]

        for obj in objects:
            rr, cc = pixel_sets[obj["id"]]
            full_area = len(rr)
            on_screen[obj["id"]].append(full_area > 0)
            if full_area == 0:
                boxes[obj["id"]].append(None)
                visibility[obj["id"]].append(0.0)
                continue
            y0, y1, x0, x1 = rr.min(), rr.max(), cc.min(), cc.max()
            boxes[obj["id"]].append((y0, x0, y1, x1))
            visible = int((owner[rr, cc] == obj["id"]).sum())
            visibility[obj["id"]].append(visible / full_area)

        noise = rng.normal(0, rng.uniform(0.02, 0.06), size=(H, W))
        frames[t] = np.clip(frame + noise, 0, 1)

    return frames, boxes, visibility, on_screen, objects