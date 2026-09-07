# decode_frustum_v3.py
#
# The previous run showed the frustum is 117 m deep - it contains the CAR plus
# the BUILDING behind it. The model can't handle that. This script tries
# several "depth cutoff" strategies on the same saved snapshot, no new
# bag/ROS run needed.
#
# RUN:  python3 decode_frustum_v3.py

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

xyz_all = cam_points[:, :3]
intensity_all = cam_points[:, 3].copy()

print("Loaded frustum: %d points total" % cam_points.shape[0])
print("  2D box [x1,x2,y1,y2] = [%.0f, %.0f, %.0f, %.0f]" % (box[0], box[1], box[2], box[3]))
print("  z (depth) spread: min=%.1f  max=%.1f  mean=%.1f"
      % (xyz_all[:,2].min(), xyz_all[:,2].max(), xyz_all[:,2].mean()))


# ------------------- depth histogram (so you can SEE the car) ----------------
print("\n--- depth histogram (how many points at each distance) ---")
z = xyz_all[:, 2]
# bin the depths into 5m slabs
edges = np.arange(0, max(80, z.max() + 5), 5.0)
counts, _ = np.histogram(z, bins=edges)
# print as a tiny text bar chart
max_count = counts.max() if counts.max() > 0 else 1
for i, c in enumerate(counts):
    if c == 0:
        continue
    bar = "#" * int(40 * c / max_count)
    print("  %3.0f-%3.0f m: %4d  %s" % (edges[i], edges[i+1], c, bar))
print("  (the BIG cluster nearest the camera is almost certainly the car;")
print("   anything past it is the background building.)")


# auto-pick the densest 5m slab and use it as a "car likely lives here" estimate
densest_bin = int(np.argmax(counts))
car_z_low  = edges[densest_bin] - 2.0          # a bit of slack in front
car_z_high = edges[densest_bin] + 7.0          # a car is ~5m long
print("\n  guess: densest slab is around %.0f - %.0f m -> car probably here"
      % (edges[densest_bin], edges[densest_bin+1]))


# ------------------------------- model setup --------------------------------
if not torch.cuda.is_available():
    print("\nERROR: no GPU."); sys.exit(1)

torch.cuda.empty_cache()
model = FrustumPointNet("decode_run_v3", ROOT)
try:
    state = torch.load(WEIGHTS, map_location="cuda")
except Exception:
    state = torch.load(WEIGHTS, map_location="cuda", weights_only=False)
model.load_state_dict(state)
model = model.cuda().eval()


def run_with_depth_filter(name, z_low, z_high):
    """Keep only points with z in [z_low, z_high], then run the full pipeline."""
    keep = (xyz_all[:, 2] >= z_low) & (xyz_all[:, 2] <= z_high)
    xyz = xyz_all[keep]
    intensity = intensity_all[keep] / 255.0   # same scaling we used before; doesn't matter much per v2

    print("\n=== variant: %s ===" % name)
    print("  depth range kept: z in [%.1f, %.1f] m" % (z_low, z_high))
    print("  points after filter: %d (was %d)" % (xyz.shape[0], xyz_all.shape[0]))

    if xyz.shape[0] < 20:
        print("  -> too few points after filter (need at least ~20); skipping.")
        return 0

    # frustum-rotate + subtract dataset offset, just like before
    u_center = 0.5 * (box[0] + box[1])
    frustum_angle = np.arctan2(u_center - cx, fx)
    c_, s_ = np.cos(frustum_angle), np.sin(frustum_angle)
    frustum_R = np.array([[ c_, 0, -s_],
                          [ 0,  1,  0 ],
                          [ s_, 0,  c_]], dtype=np.float64)
    centered = (frustum_R @ xyz.T).T - mean_xyz

    print("  after centering: x[%.1f..%.1f] y[%.1f..%.1f] z[%.1f..%.1f]"
          % (centered[:,0].min(), centered[:,0].max(),
             centered[:,1].min(), centered[:,1].max(),
             centered[:,2].min(), centered[:,2].max()))

    M = centered.shape[0]
    if M >= NUM_POINTS:
        idx = np.random.choice(M, NUM_POINTS, replace=False)
    else:
        idx = np.random.choice(M, NUM_POINTS, replace=True)
    feats = np.column_stack([centered[idx], intensity[idx]]).astype(np.float32)
    inp = torch.from_numpy(feats.T).float().unsqueeze(0).cuda()

    with torch.no_grad():
        seg, tnet, bbox, seg_mean, dontcare = model(inp)

    seg      = seg[0].cpu().numpy()
    tnet     = tnet[0].cpu().numpy()
    bbox     = bbox[0].cpu().numpy()
    seg_mean = seg_mean[0].cpu().numpy()
    dontcare = int(dontcare[0].cpu().numpy())

    fg = int(np.sum(seg[:, 1] > seg[:, 0]))
    print("  foreground points: %d / 1024  (%.1f%%)" % (fg, 100.0 * fg / NUM_POINTS))

    if fg == 0:
        print("  -> no car points found at this depth range.")
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
    print("  size    h=%.2f  w=%.2f  l=%.2f  m  (a real car: ~h1.5  w1.8  l4.3)"
          % (h, w, l))
    print("  yaw     %.2f rad (%.1f deg)" % (yaw, np.degrees(yaw)))

    # quick verdict
    sane = (1.3 <= h <= 2.2) and (1.4 <= w <= 2.3) and (3.0 <= l <= 5.8) \
           and (0 <= center_cam[2] <= 80)
    print("  verdict: %s" % ("LOOKS LIKE A REAL CAR" if sane else "numbers still off"))
    return fg


print("\n#################  TRYING DIFFERENT DEPTH CUTOFFS  #################")

# 1) KITTI's default: 0..80m (just removes the very-far stuff)
run_with_depth_filter("1. KITTI default (0 - 80 m)", 0.0, 80.0)

# 2) Closer cutoff: 0..50m (good for typical urban driving)
run_with_depth_filter("2. closer cutoff (0 - 50 m)", 0.0, 50.0)

# 3) Auto-detected car slab: just the dense slab plus a few metres
run_with_depth_filter(
    "3. auto: densest 5m slab + slack  (%.0f - %.0f m)" % (car_z_low, car_z_high),
    car_z_low, car_z_high)

# 4) Same auto slab but TIGHTER (no slack)
auto_low2  = edges[densest_bin] - 0.5
auto_high2 = edges[densest_bin] + 5.5
run_with_depth_filter(
    "4. auto: densest slab tight  (%.0f - %.0f m)" % (auto_low2, auto_high2),
    auto_low2, auto_high2)

print("\n=== summary ===")
print("If ANY variant gave a sane-looking box, that's our first real 3D")
print("detection on your own data. We'll then bake the working depth")
print("filter into the live pipeline. If all still fail, the histogram")
print("above will tell us where the car points actually are, and we")
print("can target that range manually.")
