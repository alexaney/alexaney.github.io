# CS180 (CS280A): Project 1 starter Python code

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

import numpy as np
import skimage as sk
import skimage.io as skio
import skimage.transform as sktr
from skimage.filters import sobel

def find_crop(r, g, b, max_margin=0.15, factor=1.5):
    h, w = r.shape[:2]
    disagreement = np.abs(r - g) + np.abs(g - b) + np.abs(r - b)

    row_score = disagreement.mean(axis=1)
    col_score = disagreement.mean(axis=0)

    interior_level = np.median(disagreement[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)])
    threshold = interior_level * factor

    def trim(score, margin):
        region = score[:margin]
        bad = np.where(region > threshold)[0]
        return 0 if len(bad) == 0 else bad[-1] + 1

    margin_h = int(h * max_margin)
    margin_w = int(w * max_margin)

    top = trim(row_score, margin_h)
    bottom = h - trim(row_score[::-1], margin_h)
    left = trim(col_score, margin_w)
    right = w - trim(col_score[::-1], margin_w)

    return top, bottom, left, right



def align(move, reference, window = 15, base_shift = (0, 0)):
    best_score = 0
    best_shift = (0, 0)

    base_y, base_x = base_shift

    for dy in range(base_y - window, base_y + window + 1):
        for dx in range(base_x - window, base_x + window + 1):
            shift = np.roll(move, (dy, dx), axis = (0, 1))
            score = ncc(crop_border(shift), crop_border(reference))

            if best_score == 0 or score > best_score:
                best_score = score
                best_shift = (dy, dx)
    return best_shift

def pyramid_align(move, ref, min_size=400):
    h = move.shape[0]

    # edge-based scoring (bells & whistles) -- swap move/ref for edges_move/edges_ref
    # below to re-enable
    edges_move = sobel(move)
    edges_ref = sobel(ref)

    if h <= min_size:
        return align(edges_move, edges_ref, window=15, base_shift=(0, 0))

    small_move = sktr.rescale(move, 0.5, anti_aliasing=True, channel_axis=None)
    small_ref = sktr.rescale(ref, 0.5, anti_aliasing=True, channel_axis=None)

    small_shift = pyramid_align(small_move, small_ref, min_size=400)
    scaled_shift = (small_shift[0] * 2, small_shift[1] * 2)

    return align(move, ref, window=4, base_shift=scaled_shift)



def crop_border(im, pct = 0.1):
    h, w = im.shape[:2]
    dy, dx = int(h * pct), int(w * pct)
    return im[dy : h - dy, dx : w - dx]

def l2(im1, im2):
    return np.sqrt(np.sum((im1 - im2) ** 2))

def ncc(im1, im2):
    im1_no_mean = im1 - np.mean(im1)
    im2_no_mean = im2 - np.mean(im2)

    im1_norm = im1_no_mean / np.linalg.norm(im1_no_mean)
    im2_norm = im2_no_mean / np.linalg.norm(im2_no_mean)

    return np.sum(im1_norm * im2_norm)

def main():
    # name of the input file
    imname = 'emir.tif'

    # read in the image
    im = skio.imread(imname)

    # convert to double (might want to do this later on to save memory)    
    im = sk.img_as_float(im)
        
    # compute the height of each part (just 1/3 of total)
    height = int(np.floor(im.shape[0] / 3.0))

    # separate color channels
    b = im[:height]
    g = im[height: 2*height]
    r = im[2*height: 3*height]

    # align the images
    # functions that might be useful for aligning the images include:
    # np.roll, np.sum, sk.transform.rescale (for multiscale)

    shift_g = pyramid_align(g, b)
    shift_r = pyramid_align(r, b)
    print(f"{imname}: G shift={shift_g}, R shift={shift_r}")

    ag = np.roll(g, shift_g, axis=(0, 1))
    ar = np.roll(r, shift_r, axis=(0, 1))

    top, bottom, left, right = find_crop(ar, ag, b)
    ar, ag, b = ar[top:bottom, left:right], ag[top:bottom, left:right], b[top:bottom, left:right]

    # create a color image
    im_out = np.dstack([ar, ag, b])

    # save the image
    fname = 'out.jpg'
    skio.imsave(fname, sk.img_as_ubyte(np.clip(im_out, 0, 1)))

    # display the image
    skio.imshow(im_out)
    skio.show()

if __name__ == "__main__":
    main()

