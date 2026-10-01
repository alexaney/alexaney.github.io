import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from image_sharpening import show_high
from gaussian_and_laplacian_stacks import load, gaussian_stack, laplacian_stack

# =========================================================
# Masks  (1 = image A, 0 = image B), always shape (H, W, 3)
# =========================================================
def step_mask(shape, direction="vertical"):
    H, W = shape[:2]
    mask = np.zeros((H, W, 3))
    if direction == "vertical":
        mask[:, : W // 2] = 1          # left half = A
    else:
        mask[: H // 2] = 1             # top half = A
    return mask

def ellipse_mask(shape, center, radii):
    """center = (row, col), radii = (row_radius, col_radius), in pixels."""
    H, W = shape[:2]
    rows, cols = np.mgrid[0:H, 0:W]
    inside = ((rows - center[0]) / radii[0])**2 + ((cols - center[1]) / radii[1])**2 <= 1
    return np.dstack([inside.astype(np.float64)] * 3)

def load_mask(path, shape):
    """Black/white image drawn in a paint program → 0/1 mask matching `shape`."""
    m = np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255.0
    m = cv2.resize(m, (shape[1], shape[0]))
    m = (m > 0.5).astype(np.float64)
    return np.dstack([m] * 3)

def match_size(img, shape):
    """Resize img to the same height/width as `shape`."""
    return cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)

# =========================================================
# Blending (the page-230 algorithm)
# =========================================================
def blend(imA, imB, mask, levels=5, sigma0=2):
    LA = laplacian_stack(imA, levels, sigma0)
    LB = laplacian_stack(imB, levels, sigma0)
    GM = gaussian_stack(mask, levels, sigma0)
    left    = GM * LA               # masked A, every level
    right   = (1 - GM) * LB         # masked B, every level
    blended = left + right          # blend, every level
    result  = np.clip(blended.sum(axis=0), 0, 1)
    return result, left, right, blended

# =========================================================
# Display
# =========================================================
def show_result(name, imA, imB, mask, result):
    naive = mask * imA + (1 - mask) * imB          # plain cut-and-paste, for comparison
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    for ax, (title, im) in zip(axes, [("Image A", imA), ("Image B", imB), ("Mask", mask),
                                      ("Naive paste", naive), ("Multires blend", result)]):
        ax.imshow(np.clip(im, 0, 1))
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(f"{name}_result.png")
    plt.imsave(f"{name}.png", result)

def show_process(name, left, right, blended):
    """Figure 10 style: rows = Laplacian levels, cols = masked A / masked B / blend."""
    levels = len(blended)
    cols = [("Masked A", left), ("Masked B", right), ("Blend", blended)]
    fig, axes = plt.subplots(levels + 1, 3, figsize=(12, 4 * (levels + 1)))
    for lvl in range(levels):
        for c, (title, stack) in enumerate(cols):
            axes[lvl, c].imshow(show_high(stack[lvl]))
            axes[lvl, c].set_title(f"{title}, level {lvl}")
    for c, (title, stack) in enumerate(cols):
        axes[levels, c].imshow(np.clip(stack.sum(axis=0), 0, 1))
        axes[levels, c].set_title(f"{title}, all levels summed")
    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(f"{name}_process.png")

# =========================================================
# Script
# =========================================================
if __name__ == "__main__":
    # (name, image A, image B, mask) — mask is "vertical", "horizontal", or a mask file
    blends = [
        ("orapple",   "apple.jpeg",  "orange.jpeg",  "vertical"),
        ("tennis_pickleball", "tennis_ball_aligned.jpg", "pickleball_aligned.jpg", "tennis_mask.png"),
        ("mountain_beach", "mountains.jpg", "beach.jpg", "horizontal"),   # top = mountains, bottom = beach
        # ("my_seam",   "sky.jpg",     "ocean.jpg",    "horizontal"),
        # ("eye_hand",  "eye.jpg",     "hand.jpg",     "eye_mask.png"),
        # ("moon_face", "face.jpg",    "moon.jpg",     "face_mask.png"),
    ]
    favorite = "mountain_beach"          # ← change to your best result
    levels = 5
    sigma0 = 2

    for name, pathA, pathB, mask_spec in blends:
        imA = load(pathA)
        imB = match_size(load(pathB), imA.shape)
        if mask_spec in ("vertical", "horizontal"):
            mask = step_mask(imA.shape, mask_spec)
        else:
            mask = load_mask(mask_spec, imA.shape)

        result, left, right, blended = blend(imA, imB, mask, levels, sigma0)
        show_result(name, imA, imB, mask, result)
        if name == favorite:
            show_process(name, left, right, blended)
        plt.show()
