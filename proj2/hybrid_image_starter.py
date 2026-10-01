import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from align_image_code import align_images
from image_sharpening import conv_color, gaussian_2d, show_high

HERE = os.path.dirname(os.path.abspath(__file__))

def low_pass(img, sigma):
    return conv_color(img, gaussian_2d(sigma))

def high_pass(img, sigma):
    return img - low_pass(img, sigma)

def hybrid_image(im1, im2, sigma1, sigma2, gray_high=False):
    """im1 → high frequencies (seen up close), im2 → low frequencies (seen far away).
    gray_high=True keeps only the brightness of the high frequencies, so all the color
    comes from the low-frequency image."""
    high = high_pass(im1, sigma1)
    if gray_high:
        high = np.repeat(high.mean(axis=2, keepdims=True), 3, axis=2)
    low = low_pass(im2, sigma2)
    hybrid = np.clip(low + high, 0, 1)
    return hybrid, low, high

def log_fft(img):
    gray = img.mean(axis=2) if img.ndim == 3 else img
    return np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))) + 1e-8)

def distance_strip(img, scales=(1, 0.5, 0.25, 0.125)):
    """Same image at shrinking sizes, side by side — simulates stepping back."""
    H = img.shape[0]
    pieces = []
    for s in scales:
        small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        pad = np.ones((H - small.shape[0],) + small.shape[1:])
        pieces.append(np.vstack([pad, small]))
    return np.hstack(pieces)
# First load images
name = "dog_shark"

# high sf
im1 = plt.imread('./dog.jpg') / 255.

# low sf
im2 = plt.imread('./shark.jpg') / 255.

# Next align images. Instead of the eyes, line up the eyes-midpoint → mouth axis so the
# dog's smile lands on the shark's grin (the eyes are too far apart on the shark: aligning
# them would scale the dog up 2x and put its nose on the shark's mouth).
# (x, y): dog eyes-midpoint, dog mouth, shark eyes-midpoint, shark grin
pts = ((268, 92), (275, 185), (306, 167), (305, 262))
im1_aligned, im2_aligned = align_images(im1, im2, pts, crop_padding=True)

# (previous attempt, kept for reference)
# name = "me_lebron"
# im1 = plt.imread('./me_boat.jpg') / 255.
# im2 = plt.imread('./lebron.jpg') / 255.
# im1_aligned, im2_aligned = align_images(im1, im2)

## You will provide the code below. Sigma1 and sigma2 are arbitrary
## cutoff values for the high and low frequencies

sigma1 = 4
sigma2 = 8
hybrid, low, high = hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2, gray_high=True)
plt.imsave(f"{name}_hybrid.png", hybrid)

plt.figure(figsize=(14, 5))
plt.imshow(distance_strip(hybrid))
plt.title(f"Hybrid (σ_high={sigma1}, σ_low={sigma2}) — big = close, small = far")
plt.axis("off")
plt.savefig(f"{name}_distance.png", bbox_inches="tight")
plt.show()

# Frequency analysis
panels = [
    ("Input 1 (high)", im1_aligned),
    ("Input 2 (low)",  im2_aligned),
    ("High-passed 1",  show_high(high)),
    ("Low-passed 2",   np.clip(low, 0, 1)),
    ("Hybrid",         hybrid),
]
ffts = [log_fft(im1_aligned), log_fft(im2_aligned), log_fft(high), log_fft(low), log_fft(hybrid)]

fig, axes = plt.subplots(2, 5, figsize=(22, 9))
for i, (title, im) in enumerate(panels):
    axes[0, i].imshow(im)
    axes[0, i].set_title(title)
    axes[1, i].imshow(ffts[i], cmap="gray")
    axes[1, i].set_title("FFT: " + title)
for ax in axes.ravel():
    ax.axis("off")
plt.tight_layout()
plt.savefig(f"{name}_process.png", bbox_inches="tight")
plt.show()