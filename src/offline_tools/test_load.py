# test_load.py
# GOAL: confirm the pretrained Frustum-PointNet (Car) model loads its weights
# and runs one forward pass in THIS environment, before we build anything else.
#
# SETUP: put all THREE of these in one folder (defaults to ./model):
#    - this file (test_load.py)                — or run from any dir
#    - frustum_pointnet.py                     — the model definition
#    - model_37_2_epoch_400.pth                — the pretrained weights
#
# RUN (with the bag / ROS / YOLO all STOPPED, so the GPU is free):
#    export MODEL_ROOT=./model
#    python3 test_load.py

import os
import sys
import torch

MODEL_ROOT = os.environ.get("MODEL_ROOT", "./model")
sys.path.insert(0, MODEL_ROOT)
from frustum_pointnet import FrustumPointNet

WEIGHTS = os.path.join(MODEL_ROOT, "model_37_2_epoch_400.pth")

print("PyTorch version :", torch.__version__)
print("CUDA available  :", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("WARNING: this model's forward() needs a GPU. Make sure CUDA is visible.")

# 1. Build the network (this also creates a couple of empty log folders - harmless)
net = FrustumPointNet(model_id="load_test", project_dir=MODEL_ROOT, num_points=1024)

# 2. Load the pretrained weights into it
try:
    state = torch.load(WEIGHTS, map_location="cuda")
except Exception:
    print("Retrying load with weights_only=False ...")
    state = torch.load(WEIGHTS, map_location="cuda", weights_only=False)

net.load_state_dict(state)
print("Weights loaded OK  -> the model code matches the checkpoint.")

net = net.cuda()
net.eval()

# 3. Run ONE forward pass on a dummy frustum: 1 object, 4 channels (X,Y,Z,intensity), 1024 points
dummy = torch.randn(1, 4, 1024).cuda()
with torch.no_grad():
    out_seg, out_tnet, out_bbox, out_seg_mean, out_dontcare = net(dummy)

print("")
print("Forward pass ran. Output shapes:")
print("  segmentation :", tuple(out_seg.shape),      " expected (1, 1024, 2)")
print("  T-Net        :", tuple(out_tnet.shape),     " expected (1, 3)")
print("  BboxNet      :", tuple(out_bbox.shape),     " expected (1, 14)")
print("  seg mean xyz :", tuple(out_seg_mean.shape), " expected (1, 3)")
print("  dont-care    :", tuple(out_dontcare.shape), " expected (1,)")
print("")
print("SUCCESS: the pretrained Car model loads and runs in this environment.")
