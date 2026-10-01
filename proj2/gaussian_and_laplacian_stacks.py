import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from image_sharpening import conv_color, show_high

def load(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0

def blur(img, sigma):
    ksize = 2 * int(np.ceil(3 * sigma)) + 1 # to make sure int
    g1d = cv2.getGaussianKernel(ksize, sigma)
    return conv_color(conv_color(img, g1d), g1d.T) # faster version

def gaussian_stack(img, levels, sigma0=2):
    stack = [img]
    for i in range(levels - 1):
        stack.append(blur(stack[-1], sigma0 * 2 ** i))
    return np.stack(stack)

def laplacian_stack(img, levels, sigma0=2):
    G = gaussian_stack(img, levels, sigma0)
    L = [G[i] - G[i+1] for i in range(levels - 1)] + [G[-1]]
    return np.stack(L)

def show_stack(stack, title, is_laplacian=False):
    n = len(stack)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    for i, ax in enumerate(axes):
        im = stack[i]
        if is_laplacian and i < n - 1:       # difference levels have negatives → rescale
            im = show_high(im)
        ax.imshow(np.clip(im, 0, 1))
        ax.set_title(f"{title} {i}")
        ax.axis("off")
    plt.tight_layout()
    return fig

# Script
# =========================================================
if __name__ == "__main__":
    apple = load("apple.jpeg")
    orange = load("orange.jpeg")
    assert apple.shape == orange.shape, "apple and orange must be the same size"
    H, W, _ = apple.shape

    # 6 levels → L0..L4 are all difference (band-pass) levels and L5 is the leftover
    # blurry Gaussian, so rows 0, 2, 4 of Fig. 3.42 are all real Laplacian levels
    levels = 6
    sigma0 = 2

    # ---------- stacks ----------
    GA, LA = gaussian_stack(apple, levels, sigma0), laplacian_stack(apple, levels, sigma0)
    GB, LB = gaussian_stack(orange, levels, sigma0), laplacian_stack(orange, levels, sigma0)

    # sanity check: Laplacian levels add back up to the original
    print("apple reconstructs: ", np.allclose(LA.sum(axis=0), apple))
    print("orange reconstructs:", np.allclose(LB.sum(axis=0), orange))

    show_stack(GA, "Apple Gaussian").savefig("apple_gaussian_stack.png", bbox_inches="tight")
    show_stack(LA, "Apple Laplacian", is_laplacian=True).savefig("apple_laplacian_stack.png", bbox_inches="tight")
    show_stack(GB, "Orange Gaussian").savefig("orange_gaussian_stack.png", bbox_inches="tight")
    show_stack(LB, "Orange Laplacian", is_laplacian=True).savefig("orange_laplacian_stack.png", bbox_inches="tight")
    plt.show()

    # ---------- mask: left half = apple ----------
    mask = np.zeros((H, W, 3))
    mask[:, : W // 2] = 1
    GM = gaussian_stack(mask, levels, sigma0)            # sharp at level 0, soft at the top

    # ---------- blend each level ----------
    left    = GM * LA                                    # masked apple, every level
    right   = (1 - GM) * LB                              # masked orange, every level
    blended = left + right                               # orapple, every level

    # ---------- recreate Szeliski Fig. 3.42 ----------
    rows = [0, 2, 4]
    cols = [("Apple", left), ("Orange", right), ("Orapple", blended)]

    fig, axes = plt.subplots(4, 3, figsize=(12, 16))
    for r, lvl in enumerate(rows):
        for c, (name, stack) in enumerate(cols):
            axes[r, c].imshow(show_high(stack[lvl]))
            axes[r, c].set_title(f"{name}, level {lvl}")
    for c, (name, stack) in enumerate(cols):
        axes[3, c].imshow(np.clip(stack.sum(axis=0), 0, 1))
        axes[3, c].set_title(f"{name}, all levels summed")
    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout()
    fig.savefig("fig_3_42.png")
    plt.show()

    plt.imsave("orapple.png", np.clip(blended.sum(axis=0), 0, 1))

