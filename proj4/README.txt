CS180 Project 4A: Image Warping and Mosaicing

Requirements: Python 3 with numpy, scipy, matplotlib, pillow.

To reproduce all results, run from this folder:

    python main.py

This uses the photos in images/ and the clicked correspondences in points/, and writes every
figure on the website to ../images/proj_4/ plus the recovered homographies (and the full A, b
system for pair 1->2) to homographies.txt. It takes about 20 seconds.

Files
  homography.py   all of the implementations
                    A.2: build_system, computeH
                    A.3: warpImageNearestNeighbor, warpImageBilinear (inverse warping, written from scratch)
                    A.4: chain_homographies, blend (overwrite, distance-transform feathering, two-band blend)
  main.py         runs everything: correspondence figures, homographies, rectification, mosaics
  pick_points.py  the tool I used to click correspondences (not needed to reproduce results)
                    python pick_points.py pair 1 2     point pairs between photos 1 and 2
                    python pick_points.py rect 1 name  4 corners of a rectangle for rectification
  images/         source photos (set 1: 1-5, set 2: 6-8)
  points/         clicked correspondences, in full-resolution pixel coordinates

No high-level homography or warping functions (cv2.findHomography, cv2.warpPerspective,
skimage.transform, etc.) are used. Bells & whistles were not attempted.
