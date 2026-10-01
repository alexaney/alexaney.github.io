import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageOps
from scipy.signal import convolve2d

def conv_color(img, kernel):
    """Convolves each color channel separately and then stacks them together"""
    return np.dstack([convolve2d(img[:, :, c], kernel, mode="same", boundary="symm") for c in range(img.shape[2])])

def gaussian_2d(sigma):
    ksize = 6 * sigma + 1
    g1d = cv2.getGaussianKernel(ksize, sigma)
    return g1d @ g1d.T

def unsharp_mask_filter(sigma, alpha):
    G = gaussian_2d(sigma)
    ksize = G.shape[0]
    delta = np.zeros((ksize, ksize))
    delta[ksize // 2, ksize // 2] = 1
    return (1 + alpha) * delta - alpha * G

def show_high(high):
    return (high - high.min()) / (high.max() - high.min())

if __name__ == "__main__":
    # exif_transpose applies the phone's rotation tag, so photos come out right side up
    img = np.asarray(ImageOps.exif_transpose(Image.open("campanile.jpeg")).convert("RGB"), dtype=np.float64) / 255.0

    sigma = 1
    alpha = 2

    G = gaussian_2d(sigma)

    # blur --> high freq --> add back
    blurred = conv_color(img, G)
    high = img - blurred
    sharpened = np.clip(img + alpha * high, 0, 1)

    U = unsharp_mask_filter(sigma, alpha)
    single_sharpened = np.clip(conv_color(img, U), 0, 1)

    print("single conv matches step-by-step:", np.allclose(sharpened, single_sharpened))
    print("max diff:", np.max(np.abs(sharpened - single_sharpened)))

    # ---------- show the process ----------
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(img);             axes[0].set_title("Original")
    axes[1].imshow(blurred);         axes[1].set_title(f"Blurred (σ={sigma})")
    axes[2].imshow(show_high(high)); axes[2].set_title("High frequencies")
    axes[3].imshow(sharpened);       axes[3].set_title(f"Sharpened (α={alpha})")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig("sharpen_process.png", bbox_inches="tight")
    plt.show()

    # ---------- try several strengths ----------
    alphas = [0.5, 1, 2, 5]
    fig, axes = plt.subplots(1, len(alphas) + 1, figsize=(20, 5))
    axes[0].imshow(img); axes[0].set_title("Original")
    for ax, a in zip(axes[1:], alphas):
        ax.imshow(np.clip(conv_color(img, unsharp_mask_filter(sigma, a)), 0, 1))
        ax.set_title(f"α = {a}")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    # blurring sharp image and then resharpening it
    sharp_img = img
    blurred_on_purpose = conv_color(sharp_img, G)
    resharpened = np.clip(conv_color(blurred_on_purpose, unsharp_mask_filter(sigma, 2)), 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(sharp_img);          axes[0].set_title("Original (sharp)")
    axes[1].imshow(blurred_on_purpose); axes[1].set_title("Blurred")
    axes[2].imshow(resharpened);        axes[2].set_title("Re-sharpened")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()
