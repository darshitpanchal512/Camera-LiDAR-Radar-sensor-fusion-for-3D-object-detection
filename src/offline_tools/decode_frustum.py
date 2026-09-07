# decode_frustum.py
# Loads the ONE car frustum you saved, prepares it exactly the way the
# pretrained Frustum-PointNet (Car) expects, runs the model, and prints the 3D car box.
#
# RUN (inside the container, with the GPU FREE - bag / ROS / YOLO all STOPPED):
#     nvidia-smi                       # check the 2GB GPU is mostly free first
#     python3 decode_frustum.py
#
# Needs these files in your model directory (set MODEL_ROOT env var; defaults to ./model):
#     sample_frustum.npz                 (made by frustum_saver_node.py)
#     frustum_pointnet.py
#     model_37_2_epoch_400.pth
#     kitti_centered_frustum_mean_xyz.pkl
#     kitti_train_mean_car_size.pkl

import os
import sys
import pickle
import numpy as np
import torch

# Make the result repeatable. The model randomly samples points, so without a
# fixed seed you'd get slightly different numbers each run.
np.random.seed(0)
torch.manual_seed(0)

ROOT = os.environ.get("MODEL_ROOT", "./model")
sys.path.insert(0, ROOT)                 # so "import frustum_pointnet" finds the file
from frustum_pointnet import FrustumPointNet

FRUSTUM       = os.path.join(ROOT, "sample_frustum.npz")
WEIGHTS       = os.path.join(ROOT, "model_37_2_epoch_400.pth")
MEAN_XYZ_PKL  = os.path.join(ROOT, "kitti_centered_frustum_mean_xyz.pkl")
MEAN_SIZE_PKL = os.path.join(ROOT, "kitti_train_mean_car_size.pkl")

NUM_POINTS = 1024
NH = 4                                   # the model uses 4 heading (direction) bins


# ----------------------------- small helpers --------------------------------
def load_pickle(path):
    # These mean files were saved long ago, possibly in Python 2, so try a fallback.
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            return pickle.load(f, encoding="latin1")

def wrap_to_pi(a):
    # squeeze any angle into the range -pi .. +pi
    return (a + np.pi) % (2 * np.pi) - np.pi

def bin_center(b):
    # the 4 bins evenly split the full circle; this is the centre angle of bin b
    return wrap_to_pi(b * (2 * np.pi / NH))


# ------------------------------- load inputs --------------------------------
if not os.path.exists(FRUSTUM):
    print("ERROR: %s not found." % FRUSTUM)
    print("Run frustum_saver_node.py first to create it.")
    sys.exit(1)

data = np.load(FRUSTUM)
cam_points = data["cam_points"].astype(np.float64)   # (M,4): camX, camY, camZ, intensity
box = data["box"]                                    # [x1, x2, y1, y2]
fx, fy, cx, cy = data["K"]

mean_xyz  = np.array(load_pickle(MEAN_XYZ_PKL),  dtype=np.float64).reshape(3)   # frustum offset
mean_size = np.array(load_pickle(MEAN_SIZE_PKL), dtype=np.float64).reshape(3)   # [h, w, l]

print("Loaded frustum: %d points" % cam_points.shape[0])
print("  2D box [x1,x2,y1,y2] = [%.0f, %.0f, %.0f, %.0f]" % (box[0], box[1], box[2], box[3]))
print("  mean_xyz (frustum offset) =", np.round(mean_xyz, 3))
print("  mean_size [h,w,l]         =", np.round(mean_size, 3))

xyz = cam_points[:, :3]
intensity = cam_points[:, 3].copy()

# KITTI intensity is 0..1. If yours looks like 0..255, bring it down to match.
if intensity.size and intensity.max() > 1.5:
    print("  intensity looks like 0..255 (max=%.1f) -> scaling /255" % intensity.max())
    intensity = intensity / 255.0
elif intensity.size:
    print("  intensity range = [%.3f, %.3f] (already 0..1, good)"
          % (intensity.min(), intensity.max()))


# --------------------- normalize, exactly like the training set --------------
# 1) frustum angle = horizontal angle of the ray through the 2D box centre
u_center = 0.5 * (box[0] + box[1])
frustum_angle = np.arctan2(u_center - cx, fx)

# 2) rotate the points so the frustum centre points straight ahead (+z)
c, s = np.cos(frustum_angle), np.sin(frustum_angle)
frustum_R = np.array([[ c, 0, -s],
                      [ 0, 1,  0],
                      [ s, 0,  c]], dtype=np.float64)
centered = (frustum_R @ xyz.T).T          # (M,3)

# 3) subtract the dataset's mean frustum offset
centered = centered - mean_xyz            # (M,3)

# 4) build the 4-channel input [x, y, z, intensity] and sample exactly 1024 points
feats = np.column_stack([centered, intensity])     # (M,4)
M = feats.shape[0]
if M >= NUM_POINTS:
    idx = np.random.choice(M, NUM_POINTS, replace=False)
else:
    idx = np.random.choice(M, NUM_POINTS, replace=True)   # too few -> repeat some
feats = feats[idx]                                  # (1024,4)

inp = torch.from_numpy(feats.T).float().unsqueeze(0)      # shape (1, 4, 1024)


# ------------------------------- run the model -------------------------------
if not torch.cuda.is_available():
    print("\nERROR: no GPU visible. This model's forward() needs CUDA.")
    print("Make sure the bag / ROS / YOLO are all stopped, then try again.")
    sys.exit(1)

torch.cuda.empty_cache()

model = FrustumPointNet("decode_run", ROOT)            # makes a harmless <ROOT>/training_logs folder
try:
    state = torch.load(WEIGHTS, map_location="cuda")
except Exception:
    state = torch.load(WEIGHTS, map_location="cuda", weights_only=False)
model.load_state_dict(state)
model = model.cuda().eval()

inp = inp.cuda()
with torch.no_grad():
    seg, tnet, bbox, seg_mean, dontcare = model(inp)

seg      = seg[0].cpu().numpy()           # (1024, 2)
tnet     = tnet[0].cpu().numpy()          # (3,)
bbox     = bbox[0].cpu().numpy()          # (14,)
seg_mean = seg_mean[0].cpu().numpy()      # (3,)
dontcare = int(dontcare[0].cpu().numpy()) # 1 = found car points, 0 = found none

# how many of the 1024 points the model labelled as "the car"
fg = int(np.sum(seg[:, 1] > seg[:, 0]))
fg_pct = 100.0 * fg / NUM_POINTS

print("\n--- model output ---")
print("  foreground points: %d / %d  (%.1f%%)" % (fg, NUM_POINTS, fg_pct))
if dontcare == 0:
    print("  WARNING: the model found NO car points (segmentation empty).")
    print("           The box below is meaningless in that case.")


# --------------------------------- decode ------------------------------------
# centre in the rotated/centred frame = seg centroid + T-Net shift + predicted residual
center_rot = bbox[0:3] + mean_xyz + seg_mean + tnet
# rotate back into the camera frame
center_cam = np.linalg.inv(frustum_R) @ center_rot

# size = mean car size + predicted residual   (order is height, width, length)
h = bbox[3] + mean_size[0]
w = bbox[4] + mean_size[1]
l = bbox[5] + mean_size[2]

# heading: pick the best direction bin, add its fine residual, add the frustum angle back
bin_id    = int(np.argmax(bbox[6:6 + NH]))
bin_resid = bbox[6 + NH:6 + 2 * NH][bin_id]
yaw = wrap_to_pi(bin_center(bin_id) + bin_resid + frustum_angle)

print("\n--- 3D car box (camera frame: x-right, y-down, z-forward) ---")
print("  center  x=%.2f  y=%.2f  z=%.2f   (metres)" % (center_cam[0], center_cam[1], center_cam[2]))
print("  size    height=%.2f  width=%.2f  length=%.2f   (metres)" % (h, w, l))
print("  yaw     %.3f rad  (%.1f deg)" % (yaw, np.degrees(yaw)))


# ------------------------------ sanity checks --------------------------------
print("\n--- sanity check (is this a believable car?) ---")
ok = True

def check(name, val, lo, hi):
    global ok
    good = lo <= val <= hi
    ok = ok and good
    print("  [%s] %-7s = %6.2f   (a real car is about %.1f - %.1f)"
          % ("OK " if good else "OFF", name, val, lo, hi))

check("height", h, 1.3, 2.2)
check("width",  w, 1.4, 2.3)
check("length", l, 3.0, 5.8)

z_ok = 0.0 <= center_cam[2] <= 70.0
ok = ok and z_ok
print("  [%s] z(depth)= %6.2f   (the car should be in front: 0 - 70 m)"
      % ("OK " if z_ok else "OFF", center_cam[2]))

print()
if ok and dontcare == 1:
    print(">>> RESULT: this looks like a real car box. The full pipeline works end to end. <<<")
    print("    Save this output for your report - it's your first 3D detection on your own data.")
else:
    print(">>> RESULT: the numbers look off. That's still useful - it tells us where to look. <<<")
    print("    Likely suspects, in order: intensity scale, the frustum offset file, or the")
    print("    camera-frame convention. Paste this whole output back and we'll narrow it down,")
    print("    or cross-check on a single known KITTI frame.")
