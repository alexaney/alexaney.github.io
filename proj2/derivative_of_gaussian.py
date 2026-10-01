import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from scipy.signal import convolve2d

img = np.asarray(Image.open("cameraman.png").convert("L"), dtype=np.float64) / 255.0

# Method 1

Dx = np.array([[1, 0, -1]])
Dy = np.array([[1], [0], [-1]])

sigma = 2
ksize = sigma * 6 + 1
gld = cv2.getGaussianKernel(ksize, sigma)
G = gld @ gld.T

blurred = convolve2d(img, G, mode="same", boundary="symm")
gx = convolve2d(blurred, Dx, mode="same", boundary="symm")
gy = convolve2d(blurred, Dy, mode="same", boundary="symm")

grad_mag = np.sqrt(gx**2 + gy**2)

thresholds = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12]
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for ax, t in zip(axes.ravel(), thresholds):
    ax.imshow(grad_mag > t, cmap="gray")
    ax.set_title(f"threshold: {t}")
    ax.axis("off")
plt.tight_layout()
plt.show()

threshold = 0.06                             # ← set after looking at the grid
edges = grad_mag > threshold

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
axes[0].imshow(blurred, cmap="gray");  axes[0].set_title("Blurred")
axes[1].imshow(grad_mag, cmap="gray"); axes[1].set_title("Gradient magnitude (blur → D)")
axes[2].imshow(edges, cmap="gray");    axes[2].set_title(f"Edges (t = {threshold})")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()

# Method 2
DoG_x = convolve2d(G, Dx, mode="full")
DoG_y = convolve2d(G, Dy, mode="full")

m1, m2, m3 = np.abs(DoG_x).max(), np.abs(DoG_y).max(), np.abs(G).max()
fig, axes = plt.subplots(1, 3, figsize=(8, 4))
axes[0].imshow(DoG_x, cmap="gray", vmin=-m1, vmax=m1, interpolation="nearest"); axes[0].set_title("DoG_x")
axes[1].imshow(DoG_y, cmap="gray", vmin=-m2, vmax=m2, interpolation="nearest"); axes[1].set_title("DoG_y")
axes[2].imshow(G, cmap="gray", vmin=-m3, vmax=m3, interpolation="nearest"); axes[2].set_title("G")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()

gx2 = convolve2d(img, DoG_x, mode="same", boundary="symm")
gy2 = convolve2d(img, DoG_y, mode="same", boundary="symm")
grad_mag2 = np.sqrt(gx2**2 + gy2**2)
edges2 = grad_mag2 > threshold

m = ksize
print("gx match:  ", np.allclose(gx[m:-m, m:-m], gx2[m:-m, m:-m]))
print("gy match:  ", np.allclose(gy[m:-m, m:-m], gy2[m:-m, m:-m]))
print("max diff:  ", np.max(np.abs(grad_mag[m:-m, m:-m] - grad_mag2[m:-m, m:-m])))

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(edges, cmap="gray");  axes[0].set_title("Method 1: blur → D")
axes[1].imshow(edges2, cmap="gray"); axes[1].set_title("Method 2: DoG")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()
