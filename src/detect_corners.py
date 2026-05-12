# detect_corners.py
import cv2, glob, numpy as np
from pathlib import Path

LEFT_DIR   = r"D:\xis\xis_donut_belt_homography_2026-05\left"
RIGHT_DIR  = r"D:\xis\xis_donut_belt_homography_2026-05\right"
OUT_DIR    = r"D:\xis\xis_donut_belt_homography_2026-05\calib_data"

ARUCO_DICT    = cv2.aruco.DICT_4X4_50
SQUARES_X     = 10
SQUARES_Y     = 7
SQUARE_LENGTH = 0.035
MARKER_LENGTH = 0.026

Path(OUT_DIR).mkdir(exist_ok=True)

dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
board      = cv2.aruco.CharucoBoard(
    (SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, dictionary
)
params = cv2.aruco.DetectorParameters()

def detect_one(img_path):
    img  = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m_corners, m_ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
    if m_ids is None or len(m_ids) < 4:
        return None, None, None, gray.shape[::-1]
    ok, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
        m_corners, m_ids, gray, board
    )
    if not ok or ch_corners is None or len(ch_corners) < 6:
        return None, None, None, gray.shape[::-1]
    obj_pts, img_pts = board.matchImagePoints(ch_corners, ch_ids)
    return obj_pts, img_pts, ch_ids, gray.shape[::-1]

left_imgs  = sorted(glob.glob(LEFT_DIR  + "/*.png"))
right_imgs = sorted(glob.glob(RIGHT_DIR + "/*.png"))
assert len(left_imgs) == len(right_imgs), "Image count mismatch!"

all_obj, all_left, all_right = [], [], []
image_size = None
skipped = 0

for lp, rp in zip(left_imgs, right_imgs):
    name = Path(lp).name

    obj_l, img_l, ids_l, sz = detect_one(lp)
    obj_r, img_r, ids_r, _  = detect_one(rp)

    if obj_l is None:
        print(f"  SKIP (left failed):  {name}")
        skipped += 1; continue
    if obj_r is None:
        print(f"  SKIP (right failed): {name}")
        skipped += 1; continue

    ids_l_flat = ids_l.flatten()
    ids_r_flat = ids_r.flatten()
    shared_ids = np.intersect1d(ids_l_flat, ids_r_flat)

    if len(shared_ids) < 6:
        print(f"  SKIP (only {len(shared_ids)} shared corners): {name}")
        skipped += 1; continue

    mask_l = np.isin(ids_l_flat, shared_ids)
    mask_r = np.isin(ids_r_flat, shared_ids)

    all_obj.append(obj_l[mask_l])
    all_left.append(img_l[mask_l])
    all_right.append(img_r[mask_r])
    image_size = sz
    print(f"  OK  {name}  ({len(shared_ids)} shared corners)")

print(f"\nUsable pairs: {len(all_obj)}  |  Skipped: {skipped}")

if len(all_obj) == 0:
    print("\nERROR: No usable pairs.")
else:
    np.save(f"{OUT_DIR}/obj_pts.npy",   np.array(all_obj,   dtype=object), allow_pickle=True)
    np.save(f"{OUT_DIR}/left_pts.npy",  np.array(all_left,  dtype=object), allow_pickle=True)
    np.save(f"{OUT_DIR}/right_pts.npy", np.array(all_right, dtype=object), allow_pickle=True)
    np.save(f"{OUT_DIR}/image_size.npy", image_size)
    print("Saved detection data. Proceed to homography_calibrate.py")
