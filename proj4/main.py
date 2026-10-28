import os
import time
import numpy as np
import matplotlib.pyplot as plt
from homography import (load_image, load_points, build_system, computeH, apply_H, warped_bbox,
                        warpImageNearestNeighbor, warpImageBilinear, chain_homographies, union_bbox, blend)


# Config

PAIRS = [("1", "2"), ("2", "3"), ("3", "4"), ("4", "5"), ("6", "7"), ("7", "8")]   # points/<a>_<b>.json, H maps a -> b

MOSAICS = [   # (name, images in order, reference image)
    ("bins",      ["1", "2", "3"],           "2"),
    ("entrance",  ["3", "4", "5"],           "4"),
    ("courtyard", ["1", "2", "3", "4", "5"], "3"),
    ("apartment", ["6", "7", "8"],           "7"),
]

RECTS = [     # (name, image, width/height of the real rectangle, or None to estimate from the clicks)
    ("berkeley_sign", "1", None),
    ("makerspace_window", "5", None),
]

RECT_SCALE = 0.5          # rectification runs at higher resolution so the NN/bilinear difference is visible
RECT_MARGIN = 0.6         # how much of the surroundings to keep around a rectified rectangle
MAX_CANVAS = (2.5, 1.0)   # clamp mosaic canvas to this many reference-image (widths, heights) on each side
OUT = "../images/proj_4"   # straight into the website's image folder


def save_fig(fig, name):
    fig.savefig(f"{OUT}/{name}.jpg", bbox_inches="tight", dpi=110, pil_kwargs={"quality": 90})
    plt.close(fig)

def save_im(im, name):
    plt.imsave(f"{OUT}/{name}.jpg", np.clip(im, 0, 1), pil_kwargs={"quality": 92})



# A.2

def show_correspondences(a, b, imA, imB, ptsA, ptsB):
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    for ax, im, pts, n in zip(axes, [imA, imB], [ptsA, ptsB], [a, b]):
        ax.imshow(im)
        for i, (x, y) in enumerate(pts):
            ax.plot(x, y, "o", ms=6, mec="k", mfc=colors[i % 20])
            ax.annotate(str(i), (x, y), xytext=(4, 4), textcoords="offset points", color=colors[i % 20], weight="bold")
        ax.set_title(f"image {n}")
        ax.axis("off")
    plt.tight_layout()
    save_fig(fig, f"corr_{a}_{b}")

def recover_homographies(images):
    pair_Hs = {}
    np.set_printoptions(precision=4, suppress=True, linewidth=150)
    with open("homographies.txt", "w") as f:
        for i, (a, b) in enumerate(PAIRS):
            path = f"points/{a}_{b}.json"
            if not os.path.exists(path):
                print(f"skipping pair {a}-{b}: no {path}")
                continue
            p = load_points(path)
            show_correspondences(a, b, images[a], images[b], p["im1"], p["im2"])
            H = computeH(p["im1"], p["im2"])
            pair_Hs[(a, b)] = H
            err = np.sqrt(np.mean(np.sum((apply_H(H, p["im1"]) - p["im2"]) ** 2, axis=1)))
            f.write(f"===== {a} -> {b}  ({len(p['im1'])} correspondences, RMS reprojection error {err:.2f}px)\n")
            if i == 0:
                A, bvec = build_system(p["im1"], p["im2"])
                f.write(f"A =\n{A}\n\nb =\n{bvec}\n\n")
            f.write(f"H =\n{H}\n\n")
            print(f"H {a}->{b} ({len(p['im1'])} pts, RMS error {err:.2f}px)\n{H}")
    return pair_Hs



# A.3

def rectify(name, im, quad, aspect):
    w = (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])) / 2
    h = (np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])) / 2
    if aspect is not None:
        w = h * aspect
    rect = np.array([[0, 0], [w, 0], [w, h], [0, h]])
    H = computeH(quad, rect)

    full = warped_bbox(im.shape, H)
    m = RECT_MARGIN
    bbox = (max(full[0], int(-m * w)), max(full[1], int(-m * h)),
            min(full[2], int((1 + m) * w)), min(full[3], int((1 + m) * h)))

    t = time.perf_counter()
    nn, _, _ = warpImageNearestNeighbor(im, H, bbox)
    t_nn = time.perf_counter() - t
    t = time.perf_counter()
    bil, _, _ = warpImageBilinear(im, H, bbox)
    t_bil = time.perf_counter() - t

    s = int(min(w, h) / 2)
    cx, cy = int(w / 2) - bbox[0], int(h / 2) - bbox[1]
    crop = np.s_[cy - s // 2: cy + s // 2, cx - s // 2: cx + s // 2]

    fig, axes = plt.subplots(1, 5, figsize=(24, 7), gridspec_kw={"width_ratios": [1, 1, 1, 0.8, 0.8]})
    axes[0].imshow(im)
    axes[0].plot(*np.vstack([quad, quad[:1]]).T, "r-", lw=2)
    axes[0].set_title("original + clicked corners")
    axes[1].imshow(nn)
    axes[1].set_title(f"nearest neighbor ({t_nn:.2f}s)")
    axes[2].imshow(bil)
    axes[2].set_title(f"bilinear ({t_bil:.2f}s)")
    axes[3].imshow(nn[crop], interpolation="nearest")
    axes[3].set_title("NN zoom")
    axes[4].imshow(bil[crop], interpolation="nearest")
    axes[4].set_title("bilinear zoom")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    save_fig(fig, f"rect_{name}")
    save_im(nn, f"rect_{name}_nn")
    save_im(bil, f"rect_{name}_bilinear")
    print(f"rectified {name}: NN {t_nn:.2f}s, bilinear {t_bil:.2f}s")



# A.4

def make_mosaic(name, names, ref, images, pair_Hs):
    Hs = chain_homographies(pair_Hs, names, ref)
    rh, rw = images[ref].shape[:2]
    bbox = union_bbox([warped_bbox(images[n].shape, Hs[n]) for n in names])
    kx, ky = MAX_CANVAS
    bbox = (max(bbox[0], int(-kx * rw)), max(bbox[1], int(-ky * rh)),
            min(bbox[2], int((1 + kx) * rw)), min(bbox[3], int((1 + ky) * rh)))
    canvas_shape = (bbox[3] - bbox[1] + 1, bbox[2] - bbox[0] + 1)

    warped, alphas = [], []
    for n in names:
        b = warped_bbox(images[n].shape, Hs[n])
        b = (max(b[0], bbox[0]), max(b[1], bbox[1]), min(b[2], bbox[2]), min(b[3], bbox[3]))
        out, alpha, _ = warpImageBilinear(images[n], Hs[n], b)
        place = np.s_[b[1] - bbox[1]: b[3] - bbox[1] + 1, b[0] - bbox[0]: b[2] - bbox[0] + 1]
        warped.append(np.zeros(canvas_shape + (3,)))
        alphas.append(np.zeros(canvas_shape))
        warped[-1][place], alphas[-1][place] = out, alpha
    naive, feather, two_band = blend(warped, alphas)

    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 5))
    for ax, n in zip(axes, names):
        ax.imshow(images[n])
        ax.set_title(f"image {n}" + (" (reference)" if n == ref else ""))
        ax.axis("off")
    plt.tight_layout()
    save_fig(fig, f"mosaic_{name}_sources")

    fig, axes = plt.subplots(3, 1, figsize=(14, 18))
    for ax, im, t in zip(axes, [naive, feather, two_band], ["overwrite", "distance-transform feathering", "two-band blend"]):
        ax.imshow(im)
        ax.set_title(t)
        ax.axis("off")
    plt.tight_layout()
    save_fig(fig, f"mosaic_{name}_compare")
    save_im(two_band, f"mosaic_{name}")
    print(f"mosaic {name}: canvas {two_band.shape[1]}x{two_band.shape[0]}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    images = {n: load_image(f"images/{n}.jpg") for n in sorted({n for p in PAIRS for n in p})}
    for n, im in images.items():
        save_im(im, f"source_{n}")

    pair_Hs = recover_homographies(images)

    for name, n, aspect in RECTS:
        path = f"points/rect_{name}.json"
        if os.path.exists(path):
            rectify(name, load_image(f"images/{n}.jpg", RECT_SCALE), load_points(path, RECT_SCALE)["pts"], aspect)
        else:
            print(f"skipping rectification {name}: no {path}")

    for name, names, ref in MOSAICS:
        needed = list(zip(names, names[1:]))
        if all(p in pair_Hs or p[::-1] in pair_Hs for p in needed):
            make_mosaic(name, names, ref, images, pair_Hs)
        else:
            print(f"skipping mosaic {name}: missing correspondences for one of {needed}")
