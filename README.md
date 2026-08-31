###Multi-View Conveyor Homography Stitcher

**Dual-camera ChArUco homography pipeline for stitching two asynchronous conveyor-belt cameras into a single wide-angle view.**

---

## Overview

A production line is watched by two cameras — `left` and `right` — mounted side by side over the same conveyor belt, each covering part of the belt's width. The cameras are **not hardware-synchronized** and sit at different distances and angles from the belt, so their video feeds can't simply be placed edge to edge.

Multi-View Conveyor Homography Stitcher solves this by:

1. Detecting **ChArUco board corners** (a checkerboard + ArUco marker hybrid, where every inner corner has a unique, trackable ID) in synchronized calibration image pairs from both cameras.
2. Matching corners that are visible in *both* cameras by ID, and computing a **3×3 perspective homography matrix `H`** with `cv2.findHomography` + RANSAC, mapping any pixel in the right camera to its corresponding location in the left camera's coordinate system.
3. Applying `H` to every frame of the right camera's video, compositing it onto the left camera's frame with a blended overlap region, and producing one continuous wide-angle video of the belt.

A homography — rather than full stereo rectification — is used deliberately: the conveyor belt is a flat plane, and the cameras have a large baseline (~165cm) and significant relative rotation (~6.6°), which caused stereo rectification to produce degenerate results. Because everything of interest sits on (or very near) the belt plane, a single planar homography accurately describes the geometric relationship between the two views.

---

## How it works

```
 left/*.png  right/*.png
      │           │
      ▼           ▼
┌─────────────────────────┐
│   detect_corners.py      │   detect ChArUco corners per camera,
│                           │   match by ID, keep only shared corners
└─────────────┬─────────────┘
              ▼
     calib_data/  (matched corner coords + diagnosis_report.txt)
              │
              ▼
┌─────────────────────────┐
│ homography_calibrate.py  │   cv2.findHomography(..., RANSAC)
└─────────────┬─────────────┘
              ▼
     calib_data/H_right_to_left.npy
              │
              ▼
┌─────────────────────────┐
│    wide_stitch.py        │   warp right.mp4 with H, blend overlap
│                           │   with left.mp4, composite onto canvas
└─────────────┬─────────────┘
              ▼
  main_data/belt_wide.mp4
  main_data/belt_wide_scaled.mp4
```

---

## Folder structure

```
Multi-View Conveyor Homography Stitcher/
├── calib_data/                  # auto-generated, do not edit
│   ├── H_right_to_left.npy      # the computed 3×3 homography matrix
│   ├── matched_corners.*        # per-pair matched ChArUco corner coordinates
│   └── diagnosis_report.txt     # per-frame calibration quality report (see below)
├── left/                        # ← place left camera calibration images here (.png)
├── right/                       # ← place right camera calibration images here (.png)
├── left_calibration_matrix.json   # intrinsic calibration for the left camera
├── right_calibration_matrix.json  # intrinsic calibration for the right camera
├── main_data/
│   ├── left.mp4                 # ← place left camera video here
│   ├── right.mp4                # ← place right camera video here
│   ├── belt_wide.mp4            # output — full resolution stitched video (3400×1645)
│   └── belt_wide_scaled.mp4     # output — scaled down for easy viewing
└── src/
    ├── detect_corners.py
    ├── homography_calibrate.py
    └── wide_stitch.py
```

---

## Requirements

```bash
pip install opencv-contrib-python numpy
```

- Python 3.10+
- `opencv-contrib-python` is required (not plain `opencv-python`) — the ArUco module lives in the `contrib` package and will silently be missing otherwise.

---

## ChArUco board specification

| Property        | Value        |
|------------------|--------------|
| Layout           | 10 × 7 squares |
| Dictionary       | `DICT_4X4_50` |
| Square size      | 35 mm |
| Marker size      | 26 mm |

Both cameras must observe the **same physical board** at the same moment for a frame pair to be usable — the board should be held flat, fully in view of both cameras simultaneously, and moved to a range of positions/tilts across the capture session to give the homography solver good coverage.

---

## Usage

### Step 1 — `detect_corners.py`

Loads each synchronized left/right calibration image pair, detects ChArUco corners in both, and keeps only the corners that are visible in **both** cameras at the same time, matched by their unique ID. Also runs per-frame quality diagnostics and writes everything to `calib_data/`.

```bash
python src/detect_corners.py
```

**Expected result:** ~20 usable pairs out of 28 captured, 6–54 shared corners per pair.

<details>
<summary>Example diagnostic output (abridged — see <code>calib_data/diagnosis_report.txt</code> for the full report)</summary>

```
✓  20260413_154229_0000.png  |  54 corners (0 bad)  |  max_dev 0.34mm  |  RMS 0.142mm  |  reproj 0.21px  |  tilt 2.3°  |  dist 1679mm  |  [OK]
⚠  20260413_154308_0000.png  |  54 corners (2 bad)  |  max_dev 3.00mm  |  RMS 0.486mm  |  reproj 0.33px  |  tilt 2.3°  |  dist 1661mm  |  [WARN]
✗  20260413_154200_0000.png  |  54 corners (2 bad)  |  max_dev 4.39mm  |  RMS 0.868mm  |  reproj 0.69px  |  tilt 1.7°  |  dist 1680mm  |  [BAD]  [OUTLIER_CORNERS]
–  20260413_154101_0034.png  |  NO BOARD

STEREO SYNC CHECK
  Usable pairs   : 20
  Sync misses    : 8
  Bad both sides : 0
```
</details>

#### Reading the diagnostic report

Each calibration image gets a per-frame quality line, followed by a per-camera summary and a stereo sync check:

- **`corners (N bad)`** — total ChArUco corners detected, and how many were flagged as geometric outliers.
- **`max_dev` / `RMS`** — maximum and root-mean-square deviation (in mm) between detected corner positions and the ideal planar board geometry, after 3D correction. This is the primary signal of how well the board was seen.
- **`reproj`** — mean reprojection error in pixels against the camera's intrinsic calibration. Values above ~1.0px suggest an intrinsic calibration quality issue rather than a bad capture.
- **`tilt`** — angle of the board plane relative to the camera's optical axis. Steep tilts (>15–20°) tend to correlate with worse corner localization.
- **`dist`** — estimated distance from camera to board, in mm.
- **Status codes:**
  - `[OK]` — RMS < 2.0mm
  - `[WARN]` — RMS 2.0–4.0mm
  - `[BAD]` — RMS ≥ 4.0mm (dropped from the homography solve)
  - `NO BOARD` — board not detected in that frame at all
- **`[OUTLIER_CORNERS]`** — a specific cause tag attached to bad frames, flagging that a small number of individual corners (not the whole board) are driving the error up.

The **stereo sync check** at the end is the number that matters most for this pipeline: because the two cameras aren't hardware-synced, a pair only counts as **`USABLE`** if the board was successfully detected in *both* cameras for that timestamp. Everything else is a **`SYNC MISS`** (board seen in only one camera, or neither) and is excluded from the homography solve. A low usable-pair count relative to total frames usually means the capture session needs more simultaneous board visibility, not more frames.

---

### Step 2 — `homography_calibrate.py`

Takes the matched 2D corner coordinates from both cameras (produced in step 1) and computes a 3×3 perspective homography matrix `H` via `cv2.findHomography` with RANSAC. `H` maps any pixel coordinate in the right camera to the corresponding location in the left camera.

```bash
python src/homography_calibrate.py
```

**Typical result:** 0.74px mean reprojection error, 541/592 inliers.

The resulting matrix is saved to `calib_data/H_right_to_left.npy`.

---

### Step 3 — `wide_stitch.py`

Applies `H` to every frame of `right.mp4` using `cv2.warpPerspective`, projecting it into the left camera's coordinate system. The right camera's field of view extends to `x=3316` in left-camera space, so both frames are composited onto a `3400×1645` canvas. The 851px overlap zone (`x=1197–2048`) is blended with a linear gradient to avoid a hard seam between the two source images.

```bash
python src/wide_stitch.py
```

**Output:**
- `main_data/belt_wide.mp4` — full resolution stitched video
- `main_data/belt_wide_scaled.mp4` — scaled down for easy viewing

---

## Using `H` on new footage

Once computed, `H` can be reused on any new right-camera frame without rerunning calibration, as long as the cameras haven't physically moved:

```python
import cv2, numpy as np

H = np.load(r"calib_data/H_right_to_left.npy")

frame_right = cv2.imread("any_right_frame.png")
frame_warped = cv2.warpPerspective(frame_right, H, (3400, 1645))
# frame_warped is now in the left camera's coordinate system
```

---

## Troubleshooting

- **Low usable-pair count** — check the stereo sync check section of `diagnosis_report.txt` first. A high `SYNC MISS` count means the board wasn't visible in both cameras at the same moment often enough during capture; recapture with more overlap between the two fields of view.
- **High reprojection error (>1.0px) from `homography_calibrate.py`** — usually traces back to a handful of `[WARN]`/`[BAD]` frames in the step 1 report still being included, or to bad intrinsic calibration (`left_calibration_matrix.json` / `right_calibration_matrix.json`).
- **Visible seam in `belt_wide.mp4`** — confirm the overlap region (`x=1197–2048`) actually contains real content from both source videos at that resolution; a mismatched canvas size or camera that's since moved will misalign the blend.
- **`ImportError` on `cv2.aruco`** — you likely have `opencv-python` installed instead of `opencv-contrib-python`. Uninstall the former and install the latter.

---

## Limitations

- The homography is only geometrically correct **on the belt plane**. Objects with meaningful height above the belt (e.g. tall donuts, packaging) will show parallax error proportional to their height, since a single homography cannot represent 3D depth.
- `H` is only valid as long as neither camera moves. Any physical adjustment to camera position or angle requires recapturing calibration images and rerunning steps 1–2.
- Calibration assumes both cameras share overlapping coverage of the board across a range of positions/tilts; sparse or narrow-range captures will produce a less-constrained (less accurate) solve even with a high inlier ratio.

---

## License

_Add your license here._
