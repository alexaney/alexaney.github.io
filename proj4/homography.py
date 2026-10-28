import json
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import distance_transform_edt, gaussian_filter

SCALE = 0.25 



# I/O 

def load_image(path, scale=SCALE):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if scale != 1:
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
    return np.asarray(im, dtype=np.float64) / 255

def load_points(path, scale=SCALE):
    with open(path) as f:
        return {k: np.array(v, dtype=np.float64) * scale for k, v in json.load(f).items()}



# A.2  Recover homographies

def build_system(im1_pts, im2_pts):
    n = len(im1_pts)
    A = np.zeros((2 * n, 8))
    b = np.zeros(2 * n)
    for i, ((x, y), (xp, yp)) in enumerate(zip(im1_pts, im2_pts)):
        A[2 * i]     = [x, y, 1, 0, 0, 0, -x * xp, -y * xp]
        A[2 * i + 1] = [0, 0, 0, x, y, 1, -x * yp, -y * yp]
        b[2 * i], b[2 * i + 1] = xp, yp
    return A, b

def computeH(im1_pts, im2_pts):
    A, b = build_system(im1_pts, im2_pts)
    h = np.linalg.lstsq(A, b, rcond=None)[0]
    return np.append(h, 1).reshape(3, 3)

def apply_H(H, pts):
    p = np.column_stack([pts, np.ones(len(pts))]) @ H.T
    return p[:, :2] / p[:, 2:]



# A.3  Warping 

def warped_bbox(shape, H):
    h, w = shape[:2]
    corners = apply_H(H, np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], float))
    xmin, ymin = np.floor(corners.min(0)).astype(int)
    xmax, ymax = np.ceil(corners.max(0)).astype(int)
    return xmin, ymin, xmax, ymax

def _inverse_map(shape, H, bbox):
    xmin, ymin, xmax, ymax = bbox
    xs, ys = np.meshgrid(np.arange(xmin, xmax + 1), np.arange(ymin, ymax + 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        src = apply_H(np.linalg.inv(H), np.column_stack([xs.ravel(), ys.ravel()]))
    src = np.nan_to_num(src, nan=-1, posinf=-1, neginf=-1)
    sx, sy = src[:, 0].reshape(xs.shape), src[:, 1].reshape(xs.shape)
    h, w = shape[:2]
    alpha = (sx >= 0) & (sx <= w - 1) & (sy >= 0) & (sy <= h - 1)
    return sx, sy, alpha

def warpImageNearestNeighbor(im, H, bbox=None):
    bbox = warped_bbox(im.shape, H) if bbox is None else bbox
    sx, sy, alpha = _inverse_map(im.shape, H, bbox)
    xi = np.clip(np.round(sx).astype(int), 0, im.shape[1] - 1)
    yi = np.clip(np.round(sy).astype(int), 0, im.shape[0] - 1)
    return im[yi, xi] * alpha[..., None], alpha.astype(float), bbox

def warpImageBilinear(im, H, bbox=None):
    bbox = warped_bbox(im.shape, H) if bbox is None else bbox
    sx, sy, alpha = _inverse_map(im.shape, H, bbox)
    h, w = im.shape[:2]
    x0 = np.clip(np.floor(sx).astype(int), 0, w - 2)
    y0 = np.clip(np.floor(sy).astype(int), 0, h - 2)
    dx = np.clip(sx - x0, 0, 1)[..., None]
    dy = np.clip(sy - y0, 0, 1)[..., None]
    out = (im[y0, x0]         * (1 - dx) * (1 - dy) +
           im[y0, x0 + 1]     * dx       * (1 - dy) +
           im[y0 + 1, x0]     * (1 - dx) * dy +
           im[y0 + 1, x0 + 1] * dx       * dy)
    return out * alpha[..., None], alpha.astype(float), bbox



# A.4  Mosaics

def chain_homographies(pair_Hs, names, ref):
    def step(a, b):
        return pair_Hs[(a, b)] if (a, b) in pair_Hs else np.linalg.inv(pair_Hs[(b, a)])
    r = names.index(ref)
    Hs = {ref: np.eye(3)}
    for i in range(r - 1, -1, -1):
        Hs[names[i]] = Hs[names[i + 1]] @ step(names[i], names[i + 1])
    for i in range(r + 1, len(names)):
        Hs[names[i]] = Hs[names[i - 1]] @ step(names[i], names[i - 1])
    return Hs

def union_bbox(boxes):
    b = np.array(boxes)
    return b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()

def blend(warped, alphas, sigma=4):
    weights = [distance_transform_edt(a) for a in alphas]
    wsum = np.maximum(sum(weights), 1e-8)[..., None]
    covered = (sum(alphas) > 0)[..., None]

    naive = np.zeros_like(warped[0])
    for im, a in zip(warped, alphas):
        naive = np.where(a[..., None] > 0, im, naive)

    feather = sum(w[..., None] * im for w, im in zip(weights, warped)) / wsum

    lows, highs = [], []
    for im, a in zip(warped, alphas):
        low = gaussian_filter(im, (sigma, sigma, 0)) / np.maximum(gaussian_filter(a, sigma), 1e-8)[..., None]
        lows.append(low * a[..., None])
        highs.append((im - low) * a[..., None])
    low = sum(w[..., None] * l for w, l in zip(weights, lows)) / wsum
    winner = np.argmax(np.stack(weights), axis=0)
    high = np.take_along_axis(np.stack(highs), winner[None, ..., None], axis=0)[0]
    two_band = np.clip(low + high, 0, 1) * covered

    return naive, np.clip(feather, 0, 1) * covered, two_band
