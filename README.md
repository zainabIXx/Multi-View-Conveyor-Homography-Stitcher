
# xis_donut_belt_homography_2026-05

Computes a perspective homography between two asynchronous conveyor belt cameras using ChArUco corner correspondences and RANSAC, then warps and stitches both video feeds into a single wide-angle view.

---

## folder structure

```
donut_belt_homography/
├── calib_data/                  # auto-generated, do not edit
├── left/                        # ← place left camera calibration images here (.png)
├── right/                       # ← place right camera calibration images here (.png)
├── main_data/
│   ├── left.mp4                 # ← place left camera video here
│   ├── right.mp4                # ← place right camera video here
│   ├── belt_wide.mp4            # output — full resolution stitched video (3400x1645)
│   └── belt_wide_scaled.mp4     # output — scaled down for easy viewing
└── src/
    ├── detect_corners.py
    ├── homography_calibrate.py
    └── wide_stitch.py
```

---

## pipeline

### step 1 — detect_corners.py

Loads each synchronized left/right calibration image pair and detects ChArUco corners in both. A ChArUco board combines a checkerboard with ArUco markers, allowing each inner corner to be uniquely identified by ID. Only corners visible in both cameras simultaneously are kept and matched by ID. Results are saved to `calib_data/`.

**Board:** 10×7 squares, DICT_4X4_50, square 35mm, marker 26mm

```
python src/detect_corners.py
```

Expected: 20 usable pairs out of 28, 6–54 shared corners per pair.

---

### step 2 — homography_calibrate.py

Takes matched 2D corner coordinates from both cameras and computes a 3×3 perspective homography matrix `H` using RANSAC via `cv2.findHomography`. `H` maps any pixel coordinate in the right camera to the corresponding location in the left camera.

Homography was chosen over stereo rectification because the cameras have a large separation (~165cm) and significant rotation (6.6°) between them, which caused stereo rectification to produce degenerate results. Since the conveyor belt is a flat plane, a homography perfectly describes the geometric relationship between the two views.

**Result:** 0.74px mean reprojection error, 541/592 inliers

```
python src/homography_calibrate.py
```

---

### step 3 — wide_stitch.py

Applies `H` to every frame of `right.mp4` using `cv2.warpPerspective`, projecting it into the left camera's coordinate system. The right camera's field of view extends to x=3316 in left camera space, so both frames are composited onto a 3400×1645 canvas. The 851px overlap zone (x=1197–2048) is blended with a linear gradient to avoid a hard seam.

```
python src/wide_stitch.py
```

**Output:** `main_data/belt_wide.mp4`, `main_data/belt_wide_scaled.mp4`

---

## using H on new footage

```python
import cv2, numpy as np

H = np.load(r"calib_data/H_right_to_left.npy")

frame_right = cv2.imread("any_right_frame.png")
frame_warped = cv2.warpPerspective(frame_right, H, (3400, 1645))
# frame_warped is now in left camera's coordinate system
```

---

## requirements

```
pip install opencv-contrib-python numpy
```

Python 3.10+. `opencv-contrib-python` required (not `opencv-python`) for the ArUco module.
