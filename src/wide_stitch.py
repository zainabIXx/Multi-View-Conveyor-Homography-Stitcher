import cv2, numpy as np

BASE      = r"D:\xis\xis_donut_belt_homography_2026-05"
DATA_DIR  = f"{BASE}\\calib_data"
MAIN_DATA = f"{BASE}\\main_data"

H = np.load(f"{DATA_DIR}\\H_right_to_left.npy")

OUT_W  = 3400
OUT_H  = 1645

cap_l = cv2.VideoCapture(f"{MAIN_DATA}\\left.mp4")
cap_r = cv2.VideoCapture(f"{MAIN_DATA}\\right.mp4")
fps   = cap_l.get(cv2.CAP_PROP_FPS)
total = int(cap_l.get(cv2.CAP_PROP_FRAME_COUNT))

# --- auto-detect both top and bottom crop from first frame ---
ok_l, fl = cap_l.read()
ok_r, fr = cap_r.read()
canvas_l0 = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
canvas_l0[:1536, :2048] = fl
canvas_r0 = cv2.warpPerspective(fr, H, (OUT_W, OUT_H))
combined0 = np.maximum(canvas_l0, canvas_r0)

# scan top-down for first non-black row
CROP_Y1 = 0
for row in range(OUT_H):
    if combined0[row].mean() > 10:
        CROP_Y1 = row
        break

# scan bottom-up for last non-black row
CROP_Y2 = OUT_H
for row in range(OUT_H - 1, -1, -1):
    if combined0[row].mean() > 10:
        CROP_Y2 = row
        break

print(f"  Auto CROP_Y1={CROP_Y1}, CROP_Y2={CROP_Y2}")

CROP_H  = CROP_Y2 - CROP_Y1
DISP_H  = int(CROP_H * 2048 / OUT_W)
DISP_W  = int(DISP_H * 16 / 9)

print(f"  Output scaled size: {DISP_W}x{DISP_H}")

fourcc    = cv2.VideoWriter_fourcc(*"mp4v")
out_wide  = cv2.VideoWriter(f"{MAIN_DATA}\\belt_wide.mp4",        fourcc, fps, (OUT_W,  OUT_H))
out_small = cv2.VideoWriter(f"{MAIN_DATA}\\belt_wide_scaled.mp4", fourcc, fps, (DISP_W, DISP_H))

def match_brightness(source, target):
    """Adjust source brightness to match target using LAB L-channel mean/std."""
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
    src_mean, src_std = source_lab[:, :, 0].mean(), source_lab[:, :, 0].std()
    tgt_mean, tgt_std = target_lab[:, :, 0].mean(), target_lab[:, :, 0].std()
    source_lab[:, :, 0] = (source_lab[:, :, 0] - src_mean) * (tgt_std / src_std) + tgt_mean
    source_lab[:, :, 0] = np.clip(source_lab[:, :, 0], 0, 255)
    return cv2.cvtColor(source_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

OVERLAP_START = 1197
OVERLAP_END   = 2048
BLEND_W       = OVERLAP_END - OVERLAP_START
blend_mask = np.ones((OUT_H, OUT_W), dtype=np.float32)
for i in range(BLEND_W):
    x = OVERLAP_START + i
    blend_mask[:, x] = 1.0 - (i / BLEND_W)
blend_mask[:, OVERLAP_END:] = 0.0
mask3 = np.stack([blend_mask] * 3, axis=2)

def process_frame(fl, fr):
    fr = match_brightness(fr, fl)
    canvas_l = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
    canvas_l[:1536, :2048] = fl
    canvas_r = cv2.warpPerspective(fr, H, (OUT_W, OUT_H))
    frame_out = (canvas_l.astype(np.float32) * mask3 +
                 canvas_r.astype(np.float32) * (1 - mask3)).astype(np.uint8)
    out_wide.write(frame_out)
    cropped = frame_out[CROP_Y1:CROP_Y2, :]
    out_small.write(cv2.resize(cropped, (DISP_W, DISP_H)))

# process first frame (already read)
frame_n = 1
process_frame(fl, fr)

while True:
    ok_l, fl = cap_l.read()
    ok_r, fr = cap_r.read()
    if not ok_l or not ok_r:
        break
    process_frame(fl, fr)
    frame_n += 1
    if frame_n % 100 == 0:
        print(f"  {frame_n}/{total} ({frame_n/total*100:.0f}%)")

cap_l.release(); cap_r.release()
out_wide.release(); out_small.release()
print(f"\nDone. {frame_n} frames.")
print(f"  {MAIN_DATA}\\belt_wide.mp4")
print(f"  {MAIN_DATA}\\belt_wide_scaled.mp4")