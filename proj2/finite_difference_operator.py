import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import convolve2d

img = np.asarray(Image.open("cameraman.png").convert("L"), dtype=np.float64) / 255.0

Dx = np.array([[1, 0, -1]])
Dy = np.array([[1], [0], [-1]])

gx = convolve2d(img, Dx, mode="same", boundary="symm")
gy = convolve2d(img, Dy, mode="same", boundary="symm")

grad_mag = np.sqrt(gx**2 + gy**2)

thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for ax, t in zip(axes.ravel(), thresholds):
    ax.imshow(grad_mag > t, cmap="gray")
    ax.set_title(f"threshold = {t}")
    ax.axis("off")
plt.tight_layout()
plt.show()

threshold = 0.15
edges = grad_mag > threshold

mx, my = np.abs(gx).max(), np.abs(gy).max()
fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
axes[0].imshow(gx, cmap="gray", vmin=-mx, vmax=mx); axes[0].set_title("∂I/∂x")
axes[1].imshow(gy, cmap="gray", vmin=-my, vmax=my); axes[1].set_title("∂I/∂y")
axes[2].imshow(grad_mag, cmap="gray");              axes[2].set_title("Gradient magnitude")
axes[3].imshow(edges, cmap="gray");                 axes[3].set_title(f"Edges (t = {threshold})")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()