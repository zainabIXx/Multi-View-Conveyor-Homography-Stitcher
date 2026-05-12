# homography_calibrate.py
# Computes a perspective homography from right camera to left camera
# using matched ChArUco corner coordinates and RANSAC.
import cv2, numpy as np, json, glob
from pathlib import Path

DATA_DIR  = r"D:\xis\xis_donut_belt_homography_2026-05\calib_data"
LEFT_DIR  = r"D:\xis\xis_donut_belt_homography_2026-05\left"
RIGHT_DIR = r"D:\xis\xis_donut_belt_homography_2026-05\right"

ARUCO_DICT    = cv2.aruco.DICT_4X4_50
SQUARES_X, SQUARES_Y = 10, 7
SQUARE_LENGTH = 0.035
MARKER_LENGTH = 0.026

dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
board      = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
params     = cv2.aruco.DetectorParameters()

def detect(img_path):
    img  = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mc, mi, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
    if mi is None or len(mi) < 4:
        return None, None
    ok, cc, ci = cv2.aruco.interpolateCornersCharuco(mc, mi, gray, board)
    if not ok or cc is None or len(cc) < 6:
        return None, None
    return cc, ci.flatten()

left_imgs  = sorted(glob.glob(LEFT_DIR  + "/*.png"))
right_imgs = sorted(glob.glob(RIGHT_DIR + "/*.png"))

all_left_pts  = []
all_right_pts = []

for lp, rp in zip(left_imgs, right_imgs):
    cc_l, ci_l = detect(lp)
    cc_r, ci_r = detect(rp)
    if cc_l is None or cc_r is None:
        continue
    shared = np.intersect1d(ci_l, ci_r)
    if len(shared) < 6:
        continue
    ml = np.isin(ci_l, shared)
    mr = np.isin(ci_r, shared)
    order_l = np.argsort(ci_l[ml])
    order_r = np.argsort(ci_r[mr])
    pts_l = cc_l[ml][order_l].reshape(-1, 2)
    pts_r = cc_r[mr][order_r].reshape(-1, 2)
    all_left_pts.append(pts_l)
    all_right_pts.append(pts_r)
    print(f"  {Path(lp).name}: {len(shared)} point pairs")

all_l = np.vstack(all_left_pts)
all_r = np.vstack(all_right_pts)
print(f"\nTotal point pairs for homography: {len(all_l)}")

H, mask = cv2.findHomography(all_r, all_l, cv2.RANSAC, ransacReprojThreshold=3.0)
inliers = mask.sum()
print(f"Inliers: {inliers} / {len(all_l)}")

all_r_h = np.hstack([all_r, np.ones((len(all_r), 1))])
proj = (H @ all_r_h.T).T
proj = proj[:, :2] / proj[:, 2:3]
errs = np.linalg.norm(proj - all_l, axis=1)[mask.flatten() == 1]
print(f"Reprojection error: mean={errs.mean():.3f} max={errs.max():.3f} px")

np.save(f"{DATA_DIR}/H_right_to_left.npy", H)
with open(f"{DATA_DIR}/homography.json", "w") as f:
    json.dump({"H_right_to_left": H.tolist(),
               "reprojection_error_mean": float(errs.mean()),
               "reprojection_error_max":  float(errs.max()),
               "inliers": int(inliers),
               "total_points": int(len(all_l))}, f, indent=2)

print(f"\nSaved H_right_to_left.npy and homography.json")
print("Proceed to wide_stitch.py")
