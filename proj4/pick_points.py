"""
Click correspondences and save them to points/.

  python pick_points.py pair 1 2         click a point in image 1, then the same point in image 2, repeat
  python pick_points.py rect 1 sign      click 4 corners of a rectangle: TL, TR, BR, BL

  u = undo last click, s = save, or just close the window to save.
  Use the toolbar zoom/pan for precision; clicks are ignored while zoom/pan is active.
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from homography import load_image

DISPLAY_SCALE = 0.5
COLORS = plt.cm.tab20(np.linspace(0, 1, 20))
plt.rcParams["keymap.save"] = []


def pick(mode, names, out_path):
    ims = [load_image(f"images/{n}.jpg", DISPLAY_SCALE) for n in names]
    keys = ["im1", "im2"] if mode == "pair" else ["pts"]
    pts = {k: [] for k in keys}
    if os.path.exists(out_path):
        with open(out_path) as f:
            pts = {k: [list(np.array(p) * DISPLAY_SCALE) for p in v] for k, v in json.load(f).items()}

    fig, axes = plt.subplots(1, len(ims), figsize=(8 * len(ims), 9), squeeze=False)
    axes = list(axes[0])
    for ax, im, n in zip(axes, ims, names):
        ax.imshow(im)
        ax.set_xticks([]), ax.set_yticks([])
    artists, history = [], [k for i in range(len(pts[keys[0]])) for k in keys]

    def turn():
        return int(np.argmin([len(pts[k]) for k in keys])) if mode == "pair" else 0

    def redraw():
        for a in artists:
            a.remove()
        artists.clear()
        for ax, k in zip(axes, keys):
            for i, (x, y) in enumerate(pts[k]):
                c = COLORS[i % 20]
                artists.append(ax.plot(x, y, "o", ms=6, mec="k", mfc=c)[0])
                artists.append(ax.annotate(str(i), (x, y), xytext=(5, 5), textcoords="offset points",
                                           color=c, fontsize=11, weight="bold"))
        if mode == "pair":
            fig.suptitle(f"{len(pts['im2'])} pairs - click in image {names[turn()]}   (u = undo, s = save)")
        else:
            labels = ["top-left", "top-right", "bottom-right", "bottom-left"]
            n = len(pts["pts"])
            fig.suptitle(f"click the {labels[n]} corner of the real-world rectangle (poster, window...), not of the photo"
                         if n < 4 else "done - close to save")
        fig.canvas.draw_idle()

    def on_click(e):
        tb = fig.canvas.toolbar
        if (tb is not None and tb.mode != "") or e.inaxes not in axes or e.button != 1:
            return
        k = keys[turn()]
        if e.inaxes is not axes[keys.index(k)] or (mode == "rect" and len(pts[k]) >= 4):
            return
        pts[k].append([e.xdata, e.ydata])
        history.append(k)
        redraw()

    def save():
        if len({len(v) for v in pts.values()}) != 1 or (mode == "rect" and len(pts["pts"]) != 4):
            print("not saved: unmatched points (or rect needs exactly 4)")
            return
        with open(out_path, "w") as f:
            json.dump({k: (np.array(v) / DISPLAY_SCALE).tolist() for k, v in pts.items()}, f, indent=1)
        print(f"saved {len(pts[keys[0]])} points to {out_path}")

    def on_key(e):
        if e.key == "u" and history:
            pts[history.pop()].pop()
            redraw()
        elif e.key == "s":
            save()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("close_event", lambda e: save())
    redraw()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    os.makedirs("points", exist_ok=True)
    mode = sys.argv[1]
    if mode == "pair":
        a, b = sys.argv[2], sys.argv[3]
        pick("pair", [a, b], f"points/{a}_{b}.json")
    elif mode == "rect":
        im, name = sys.argv[2], sys.argv[3]
        pick("rect", [im], f"points/rect_{name}.json")
    else:
        print(__doc__)
