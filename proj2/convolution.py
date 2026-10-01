import numpy as np
from PIL import Image   
import matplotlib.pyplot as plt
import time
from scipy.signal import convolve2d

def zero_pad(img, ph, pw):
    """Surround img with ph rows of zeros top/bottom and pw cols left/right."""
    H, W = np.shape(img)
    padded = np.zeros((H + 2 * ph, W + 2 * pw), dtype=np.float64)
    padded[ph:ph + H, pw:pw + W] = img
    return padded

def conv2d_4loops(img, kernel, mode="same"):
    img = img.astype(np.float64)
    k = kernel[::-1, ::-1].astype(np.float64)
    h, w = np.shape(img)
    kh, kw = np.shape(k)

    padded = zero_pad(img, kh-1, kw-1)
    outH, outW = h + kh - 1, w + kw - 1
    out = np.zeros((outH, outW))

    for i in range(outH):
        for j in range(outW):
            total = 0.0
            for u in range(kh):
                for v in range(kw):
                    total += padded[i + u, j + v] * k[u, v]
            out[i, j] = total
    if mode == "same":
        r, c = (kh - 1) // 2, (kw - 1) // 2
        out = out[r:r + h, c:c + w]
    return out

def conv2d_2loops(img, kernel, mode="same"):
    img = img.astype(np.float64)
    k = kernel[::-1, ::-1].astype(np.float64)
    h, w = np.shape(img)
    kh, kw = np.shape(k)

    padded = zero_pad(img, kh-1, kw-1)
    outH, outW = h + kh - 1, w + kw - 1
    out = np.zeros((outH, outW))

    for i in range(outH):
        for j in range(outW):
            patch = padded[i:i + kh, j:j + kw]
            out[i, j] = np.sum(patch * k)
    if mode == "same":
        r, c = (kh - 1) // 2, (kw - 1) // 2
        out = out[r:r + h, c:c + w]
    return out

im = Image.open("me.jpg").convert("L")        # "L" = grayscale
im = im.resize((im.width // 4, im.height // 4))  # SHRINK IT (see below)
img = np.asarray(im, dtype=np.float64) / 255.0   # float in [0, 1]

box = np.ones((9, 9)) / 81.0     # 9x9 box: weights sum to 1 → brightness preserved
Dx  = np.array([[1, 0, -1]])        # shape (1, 2): horizontal difference
Dy  = np.array([[1], [0], [-1]])      # shape (2, 1): vertical difference

blurred = conv2d_2loops(img, box, "same")
gx      = conv2d_2loops(img, Dx,  "same")
gy      = conv2d_2loops(img, Dy,  "same")

# Dx/Dy outputs have negative values; use a symmetric range so 0 = mid-gray
mx = np.abs(gx).max()
my = np.abs(gy).max()

fig, axes = plt.subplots(1, 4, figsize=(16, 5))
axes[0].imshow(img,     cmap="gray", vmin=0, vmax=1);   axes[0].set_title("Original")
axes[1].imshow(blurred, cmap="gray", vmin=0, vmax=1);   axes[1].set_title("Box 9x9")
axes[2].imshow(gx,      cmap="gray", vmin=-mx, vmax=mx); axes[2].set_title("Dx")
axes[3].imshow(gy,      cmap="gray", vmin=-my, vmax=my); axes[3].set_title("Dy")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()

filters = {"box": box, "Dx": Dx, "Dy": Dy}

for name, kernel in filters.items():
    # --- run all three versions, timing each ---
    t0 = time.perf_counter()
    out4 = conv2d_4loops(img, kernel, "same")
    t4 = time.perf_counter() - t0

    t0 = time.perf_counter()
    out2 = conv2d_2loops(img, kernel, "same")
    t2 = time.perf_counter() - t0

    t0 = time.perf_counter()
    ref = convolve2d(img, kernel, mode="same", boundary="fill", fillvalue=0)
    tsp = time.perf_counter() - t0

    # --- compare ---
    print(f"--- {name} ---")
    print("  shapes:", out4.shape, out2.shape, ref.shape)
    print("  4-loop max diff:", np.max(np.abs(out4 - ref)), " match:", np.allclose(out4, ref))
    print("  2-loop max diff:", np.max(np.abs(out2 - ref)), " match:", np.allclose(out2, ref))
    print(f"  time: 4-loop {t4:.3f}s | 2-loop {t2:.3f}s | scipy {tsp:.5f}s")