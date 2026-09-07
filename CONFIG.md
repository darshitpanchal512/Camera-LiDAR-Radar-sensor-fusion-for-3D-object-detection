# Configuration

All file paths in this repository are driven by environment variables with sensible defaults. Set them once and the code Just Works — no editing of source files required.

## Environment variables

| Variable | Default | Used by | What it points to |
|---|---|---|---|
| `MODEL_ROOT` | `./model` | offline tools, `test_load` | Folder containing `frustum_pointnet.py`, weights (`.pth`), and the two KITTI mean pickles |
| `YOLO_ENGINE_PATH` | `./yolo26n.engine` | frustum extractor, saver | TensorRT YOLO engine file |
| `CAMERA_CALIB_YAML` | `./calibration/camera_front_center_autoware_camera_calibration.yaml` | LiDAR projection | Camera intrinsics YAML |
| `SAMPLE_FRUSTUM_PATH` | `./sample_frustum.npz` | frustum saver | Where the saver writes one captured frustum |
| `LIVE_FRUSTUM_DUMP_PATH` | `./live_frustum_dump.npz` | car inference | Debug dump of the first live frustum |

## Quick start

Create a folder for the model files, drop them in, then export the variables:

```bash
mkdir -p ./model ./calibration

# Download the three third-party files (see below) into ./model/

# Point everything at those locations
export MODEL_ROOT="$PWD/model"
export YOLO_ENGINE_PATH="$PWD/model/yolo26n.engine"
export CAMERA_CALIB_YAML="$PWD/calibration/your_calibration.yaml"
```

That's it — every script will find what it needs.

## Runtime dependencies

- **ROS 2 Humble** with `sensor_msgs`, `std_msgs`, `visualization_msgs`, `cv_bridge`, `message_filters`
- **Python 3.10** with `torch` (CUDA build), `numpy`, `opencv-python`, `ultralytics`
- **A CUDA-capable GPU** — 2 GB is enough for inference (I ran on a GTX 1650)
- **TensorRT** if you use the YOLO engine as-is; otherwise adapt to any Ultralytics-compatible detector

## Third-party files you need to download separately

From [fregu856/3DOD_thesis](https://github.com/fregu856/3DOD_thesis) — place all three inside your `MODEL_ROOT` folder:

- `frustum_pointnet.py` — the model class definition
- `model_37_2_epoch_400.pth` — the pretrained KITTI (car) weights
- `kitti_centered_frustum_mean_xyz.pkl` and `kitti_train_mean_car_size.pkl` — KITTI dataset statistics used by the model

You'll also need your own **YOLO segmentation engine** (any Ultralytics-supported detector works; I used `yolo26n.engine` compiled with TensorRT).

## What you cannot get from this repo

- **The bag files.** ANTON platform recordings contain identifiable people and vehicles and are not distributable. You can adapt the code to any ROS 2 bag with a rectified camera image, a LiDAR point cloud, and matching camera intrinsics/extrinsics.
- **The team's finalized nuScenes-trained model.** This repository is the initial (KITTI-weights) version — the finalized fusion model is the team's collective work and is not published here.
