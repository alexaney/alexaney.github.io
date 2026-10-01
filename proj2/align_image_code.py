"""Lets the user click two corresponding points on each of two images,
then aligns the images so those two points overlap (matching center,
scale, and rotation), and crops both to a common size.
"""
import math

import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as sktr


def get_points(im1: np.ndarray, im2: np.ndarray) -> tuple:
    print('Please select 2 points in each image for alignment.')
    plt.imshow(im1)
    p1, p2 = plt.ginput(2)
    plt.close()
    plt.imshow(im2)
    p3, p4 = plt.ginput(2)
    plt.close()
    return (p1, p2, p3, p4)


def recenter(im: np.ndarray, r: float, c: float) -> np.ndarray:
    R, C = im.shape[:2]
    rpad = int(np.abs(2*r+1 - R))
    cpad = int(np.abs(2*c+1 - C))
    pad_width = [(0 if r > (R-1)/2 else rpad, 0 if r < (R-1)/2 else rpad),
                 (0 if c > (C-1)/2 else cpad, 0 if c < (C-1)/2 else cpad)]
    if im.ndim == 3:
        pad_width.append((0, 0))
    return np.pad(im, pad_width, 'constant')


def find_centers(p1: tuple, p2: tuple) -> tuple:
    cx = np.round(np.mean([p1[0], p2[0]]))
    cy = np.round(np.mean([p1[1], p2[1]]))
    return cx, cy


def align_image_centers(im1: np.ndarray, im2: np.ndarray, pts: tuple) -> tuple:
    p1, p2, p3, p4 = pts

    cx1, cy1 = find_centers(p1, p2)
    cx2, cy2 = find_centers(p3, p4)

    im1 = recenter(im1, cy1, cx1)
    im2 = recenter(im2, cy2, cx2)
    return im1, im2


def rescale_images(im1: np.ndarray, im2: np.ndarray, pts: tuple) -> tuple:
    p1, p2, p3, p4 = pts
    len1 = np.sqrt((p2[1] - p1[1])**2 + (p2[0] - p1[0])**2)
    len2 = np.sqrt((p4[1] - p3[1])**2 + (p4[0] - p3[0])**2)
    dscale = len2/len1
    channel_axis = -1 if im1.ndim == 3 else None
    if dscale < 1:
        im1 = sktr.rescale(im1, dscale, channel_axis=channel_axis)
    else:
        im2 = sktr.rescale(im2, 1./dscale, channel_axis=channel_axis)
    return im1, im2


def rotate_im1(im1: np.ndarray, pts: tuple) -> tuple:
    p1, p2, p3, p4 = pts
    theta1 = math.atan2(-(p2[1] - p1[1]), (p2[0] - p1[0]))
    theta2 = math.atan2(-(p4[1] - p3[1]), (p4[0] - p3[0]))
    dtheta = theta2 - theta1
    im1 = sktr.rotate(im1, dtheta*180/np.pi)
    return im1, dtheta


def match_img_size(im1: np.ndarray, im2: np.ndarray) -> tuple:
    # Make images the same size
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]
    if h1 < h2:
        im2 = im2[int(np.floor((h2-h1)/2.)) : -int(np.ceil((h2-h1)/2.)), :]
    elif h1 > h2:
        im1 = im1[int(np.floor((h1-h2)/2.)) : -int(np.ceil((h1-h2)/2.)), :]
    if w1 < w2:
        im2 = im2[:, int(np.floor((w2-w1)/2.)) : -int(np.ceil((w2-w1)/2.))]
    elif w1 > w2:
        im1 = im1[:, int(np.floor((w1-w2)/2.)) : -int(np.ceil((w1-w2)/2.))]
    assert im1.shape == im2.shape
    return im1, im2


def crop_to_valid(valid: np.ndarray) -> tuple:
    """Shrink a box until every pixel in it is valid, trimming whichever edge has
    the most invalid pixels each step. Returns (top, bottom, left, right)."""
    t, b, l, r = 0, valid.shape[0], 0, valid.shape[1]
    while True:
        v = valid[t:b, l:r]
        bad = {"t": (~v[0]).sum(), "b": (~v[-1]).sum(),
               "l": (~v[:, 0]).sum(), "r": (~v[:, -1]).sum()}
        edge = max(bad, key=bad.get)
        if bad[edge] == 0:
            return t, b, l, r
        if edge == "t":
            t += 1
        elif edge == "b":
            b -= 1
        elif edge == "l":
            l += 1
        else:
            r -= 1


def align_images(im1: np.ndarray, im2: np.ndarray, pts: tuple = None,
                 crop_padding: bool = False) -> tuple:
    """pts = (p1, p2, p3, p4) as (x, y): two points on im1, then the matching two on im2.
    If pts is None, click them interactively. If crop_padding is True, cut away the
    black borders that recentering/rotating adds, keeping only pixels both images cover."""
    if pts is None:
        pts = get_points(im1, im2)
    if crop_padding:
        # extra all-ones channel that goes through the same transforms; wherever it
        # ends up < 1, that pixel came from zero padding instead of the real image
        im1 = np.dstack([im1, np.ones(im1.shape[:2])])
        im2 = np.dstack([im2, np.ones(im2.shape[:2])])
    im1, im2 = align_image_centers(im1, im2, pts)
    im1, im2 = rescale_images(im1, im2, pts)
    im1, angle = rotate_im1(im1, pts)
    im1, im2 = match_img_size(im1, im2)
    if crop_padding:
        t, b, l, r = crop_to_valid((im1[..., -1] > 0.99) & (im2[..., -1] > 0.99))
        im1, im2 = im1[t:b, l:r, :-1], im2[t:b, l:r, :-1]
    return im1, im2


if __name__ == "__main__":
    # 1. load the image
    # 2. align the two images by calling align_images
    # Now you are ready to write your own code for creating hybrid images!
    pass
