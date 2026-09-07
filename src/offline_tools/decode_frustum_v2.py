# decode_frustum_v2.py
#
# Same idea as decode_frustum.py, but it digs deeper:
#   1) Prints what the point cloud actually looks like after centering
#      (so we can check the geometry step is sane).
#   2) Tries the model FOUR different ways of handling the "intensity"
#      channel (since 0% foreground points at that being the suspect),
#      and prints the result of each, side by side.
#
# RUN (GPU free, no ROS/bag/filter needed - same as before):
#     python3 decode_frustum_v2.py

import os
import sys
import pickle
import numpy as np
import torch

np.random.seed(0)
torch.manual_seed(0)

ROOT = os.environ.get("MODEL_ROOT", "./model")
sys.path.insert(0, ROOT)
from frustum_pointnet import FrustumPointNet

FRUSTUM       = os.path.join(ROOT, "sample_frustum.npz")
WEIGHTS       = os.path.join(ROOT, "model_37_2_epoch_400.pth")
MEAN_XYZ_PKL  = os.path.join(ROOT, "kitti_centered_frustum_mean_xyz.pkl")
MEAN_SIZE_PKL = os.path.join(ROOT, "kitti_train_mean_car_size.pkl")

NUM_POINTS = 1024
NH = 4


def load_pickle(path):
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            return pickle.load(f, encoding="latin1")

def wrap_to_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def bin_center(b):
    return wrap_to_pi(b * (2 * np.pi / NH))


# ------------------------------- load inputs --------------------------------
data = np.load(FRUSTUM)
cam_points = data["cam_points"].astype(np.float64)
box = data["box"]
fx, fy, cx, cy = data["K"]

mean_xyz  = np.array(load_pickle(MEAN_XYZ_PKL),  dtype=np.float64).reshape(3)
mean_size = np.array(load_pickle(MEAN_SIZE_PKL), dtype=np.float64).reshape(3)

xyz = cam_points[:, :3]
raw_intensity = cam_points[:, 3].copy()

print("Loaded frustum: %d points" % cam_points.shape[0])
print("  2D box [x1,x2,y1,y2] = [%.0f, %.0f, %.0f, %.0f]" % (box[0], box[1], box[2], box[3]))
print("  mean_xyz (dataset offset) =", np.round(mean_xyz, 3))
print("  mean_size [h,w,l]         =", np.round(mean_size, 3))

print("\n--- raw point cloud (camera frame, before centering) ---")
print("  x: min=%.2f max=%.2f mean=%.2f" % (xyz[:,0].min(), xyz[:,0].max(), xyz[:,0].mean()))
print("  y: min=%.2f max=%.2f mean=%.2f" % (xyz[:,1].min(), xyz[:,1].max(), xyz[:,1].mean()))
print("  z: min=%.2f max=%.2f mean=%.2f" % (xyz[:,2].min(), xyz[:,2].max(), xyz[:,2].mean()))
print("  intensity: min=%.2f max=%.2f mean=%.2f" % (raw_intensity.min(), raw_intensity.max(), raw_intensity.mean()))


# --------------------- normalize geometry (intensity handled later) ----------
u_center = 0.5 * (box[0] + box[1])
frustum_angle = np.arctan2(u_center - cx, fx)

c, s = np.cos(frustum_angle), np.sin(frustum_angle)
frustum_R = np.array([[ c, 0, -s],
                      [ 0, 1,  0],
                      [ s, 0,  c]], dtype=np.float64)

centered = (frustum_R @ xyz.T).T - mean_xyz   # (M,3)

print("\n--- after frustum rotation + centering (this is what the model sees as XYZ) ---")
print("  x: min=%.2f max=%.2f mean=%.2f" % (centered[:,0].min(), centered[:,0].max(), centered[:,0].mean()))
print("  y: min=%.2f max=%.2f mean=%.2f" % (centered[:,1].min(), centered[:,1].max(), centered[:,1].mean()))
print("  z: min=%.2f max=%.2f mean=%.2f" % (centered[:,2].min(), centered[:,2].max(), centered[:,2].mean()))
print("  (for comparison, a car's points after this step should form a tight cluster")
print("   spanning only a few metres in each direction)")


# pick the 1024 points ONCE, reuse for every variant so the comparison is fair
M = centered.shape[0]
if M >= NUM_POINTS:
    idx = np.random.choice(M, NUM_POINTS, replace=False)
else:
    idx = np.random.choice(M, NUM_POINTS, replace=True)
centered_sampled = centered[idx]            # (1024,3)
raw_intensity_sampled = raw_intensity[idx]  # (1024,)


# ------------------------------- load model ----------------------------------
if not torch.cuda.is_available():
    print("\nERROR: no GPU visible.")
    sys.exit(1)

torch.cuda.empty_cache()
model = FrustumPointNet("decode_run_v2", ROOT)
try:
    state = torch.load(WEIGHTS, map_location="cuda")
except Exception:
    state = torch.load(WEIGHTS, map_location="cuda", weights_only=False)
model.load_state_dict(state)
model = model.cuda().eval()


def run_variant(name, intensity_values):
    feats = np.column_stack([centered_sampled, intensity_values]).astype(np.float32)  # (1024,4)
    inp = torch.from_numpy(feats.T).float().unsqueeze(0).cuda()  # (1,4,1024)

    with torch.no_grad():
        seg, tnet, bbox, seg_mean, dontcare = model(inp)

    seg      = seg[0].cpu().numpy()
    tnet     = tnet[0].cpu().numpy()
    bbox     = bbox[0].cpu().numpy()
    seg_mean = seg_mean[0].cpu().numpy()
    dontcare = int(dontcare[0].cpu().numpy())

    fg = int(np.sum(seg[:, 1] > seg[:, 0]))

    print("\n--- variant: %s ---" % name)
    print("  intensity channel fed to model: min=%.3f max=%.3f mean=%.3f"
          % (intensity_values.min(), intensity_values.max(), intensity_values.mean()))
    print("  foreground points: %d / 1024  (%.1f%%)" % (fg, 100.0 * fg / NUM_POINTS))

    if fg == 0:
        print("  -> model still sees no car here. Box would be the generic default, skipping.")
        return fg

    center_rot = bbox[0:3] + mean_xyz + seg_mean + tnet
    center_cam = np.linalg.inv(frustum_R) @ center_rot
    h = bbox[3] + mean_size[0]
    w = bbox[4] + mean_size[1]
    l = bbox[5] + mean_size[2]
    bin_id    = int(np.argmax(bbox[6:6 + NH]))
    bin_resid = bbox[6 + NH:6 + 2 * NH][bin_id]
    yaw = wrap_to_pi(bin_center(bin_id) + bin_resid + frustum_angle)

    print("  center  x=%.2f  y=%.2f  z=%.2f  m" % (center_cam[0], center_cam[1], center_cam[2]))
    print("  size    h=%.2f  w=%.2f  l=%.2f  m" % (h, w, l))
    print("  yaw     %.2f rad (%.1f deg)" % (yaw, np.degrees(yaw)))
    return fg


print("\n================= TRYING 4 WAYS TO HANDLE INTENSITY =================")

results = {}
results["A: /255 (what we tried before)"] = run_variant(
    "A: /255 (what we tried before)", raw_intensity_sampled / 255.0)

results["B: set to zero"] = run_variant(
    "B: set to zero", np.zeros_like(raw_intensity_sampled))

results["C: /2000 (assumes ~0-2000 sensor range)"] = run_variant(
    "C: /2000 (assumes ~0-2000 sensor range)", raw_intensity_sampled / 2000.0)

results["D: raw, no scaling"] = run_variant(
    "D: raw, no scaling", raw_intensity_sampled)


print("\n================= SUMMARY =================")
any_fg = False
for name, fg in results.items():
    flag = "FOUND CAR POINTS" if fg > 0 else "still 0%"
    print("  %-38s -> %4d / 1024   %s" % (name, fg, flag))
    if fg > 0:
        any_fg = True

print()
if any_fg:
    print(">>> At least one intensity option works! Use that one going forward.")
    print(">>> If 'B: set to zero' is the one that works, that's fine -")
    print(">>> it just means we drop the intensity info for now (geometry alone is enough).")
else:
    print(">>> All four still give 0%. Intensity probably isn't the (only) cause.")
    print(">>> Next step: check the 'after centering' numbers printed above -")
    print(">>> paste this whole output and we'll look at the geometry/calibration instead.")
