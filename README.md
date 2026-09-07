# Camera-LiDAR-Radar-sensor-fusion-for-3D-object-detection
"Camera-LiDAR-Radar sensor fusion pipeline for 3D object detection. ROS 2 + PyTorch. THI Ingolstadt / CARISSMA."


**Project — THI Ingolstadt / CARISSMA C-ISAFE  ·  Supervisor: Dr. Thiago de Borba**


A ROS 2 pipeline that fuses camera, LiDAR and radar to produce 3D bounding boxes for vehicles and pedestrians on real recorded drives from the ANTON platform. This repository is the portfolio version of the work I contributed to that project — the AI inference backbone, offline analysis tooling, and the LiDAR-preprocessing side of the sensor stack.

The full pipeline was a four-person team effort. My contribution and the team's shared architecture are both credited below.

---

## What the full pipeline does

The final architecture is a **sequential middle-level fusion pipeline**. YOLO26-seg produces instance masks on the camera stream. Those masks carve pixel-accurate frustums from the LiDAR and radar data, which are unified into a single **seven-channel representation** carrying position, reflectivity, Doppler velocity, and a modality flag. A custom **Frustum-PointNet with a T-Net spatial-alignment stage** then regresses full 3D bounding boxes — centre, dimensions, and yaw. Everything runs live off recorded drives, published as ROS 2 markers, visualised in RViz.

<img width="2000" height="1125" alt="image" src="https://github.com/user-attachments/assets/f0549f42-e6ce-4000-8058-104ebdd33a41" />

```
Camera image ─▶ YOLO26-seg (masks) ─┐
                                     ├─▶ Frustum extractor (7-channel) ─▶ Frustum-PointNet ─▶ 3D boxes ─▶ RViz
LiDAR   ─▶ preprocessing ───────────┤
Radar   ─▶ preprocessing ───────────┘
```

---

## contribution

I worked on three connected slices of this project — the details of each are in `docs/`:

- **LiDAR preprocessing** — turning the raw 360° laser scan into a clean, evenly-spaced, ground-free point cloud ready for fusion. ROI cropping, voxel downsampling, ground removal, intensity preservation. (See [`src/lidar_preprocessing/`](src/lidar_preprocessing/).)

- **Frustum extractor and Car inference node (initial version)** — the first working end-to-end 3D detection pipeline: 2D-box frustums, Frustum-PointNet inference (fregu856's pretrained KITTI weights), full geometric decoding into 3D boxes, ROS 2 markers to RViz. This version made the domain gap and its limits *visible* — which is what motivated the team's move to the finalized mask + retrained-model design. (See [`src/frustum_extractor_v1/`](src/frustum_extractor_v1/), [`src/car_inference/`](src/car_inference/), and [`src/fusion_msgs/`](src/fusion_msgs/).)

- **Offline analysis tooling** — the `decode_frustum` scripts (v1/v2/v3) and `test_load`: a set of GPU-only, ROS-free tools that replay a single saved frustum through the same model and decode maths as the live node, so you can study a real frame in isolation. These were the tools that isolated the intensity vs geometry question and produced the project's first believable 3D car detection (49.9% foreground, box dimensions matching a real car within centimetres). (See [`src/offline_tools/`](src/offline_tools/) and [`results/offline_test_49pct.md`](results/offline_test_49pct.md).)

For a full walk-through of what I built and what I learned, see [`docs/02_my_contributions.md`](docs/02_my_contributions.md).

---

## What's in this repository

```
src_finalized/        The finalized team pipeline — reference code, joint team authorship (see folder README)
src/                  Only code  — LiDAR prep, extractor v1, car inference, fusion messages, offline tools

```

The team's finalized presentation is **not** included here at the team's request. What's public here is the code and documentation I own. The finalized **code** in `src_finalized/` is included for reference and completeness — it was built collaboratively by the team, and I am not the sole author. See [`src_finalized/README.md`](src_finalized/README.md) for the authorship details.

---

## Stack

- **Middleware:** ROS 2 Humble  ·  DDS
- **AI:** PyTorch (inference )
- **Runtime:** Docker on Ubuntu 22.04 / WSL2  ·  GTX 1650 (2 GB VRAM)
- **Sensors used (team):** monocular camera, Ouster OS2 LiDAR, Continental ARS548 radar (ANTON platform)
- **Datasets referenced:** KITTI (initial pretrained weights, via fregu856), nuScenes (finalized team model)

## Requirements

- ROS 2 Humble
- Python 3.10, PyTorch (CUDA-enabled build)
- LiDAR bag files from the ANTON platform (not distributed — see `docs/` for the input contract)
- fregu856's Frustum-PointNet weights (not distributed)

## Model weights

**Pretrained weights are **not included** in this repository. The AI inference backbone in `src/car_inference/` builds on [fregu856/3DOD_thesis](https://github.com/fregu856/3DOD_thesis) — please download the KITTI Frustum-PointNet weights from that repository directly. The finalized team model in `src_finalized/` was trained on nuScenes and those weights are also not distributed here.**

## Data

Recorded sensor data from the ANTON platform is **not included** for data-protection reasons (recorded scenes contain identifiable people and vehicles).

---

## Team & acknowledgements

Project was a team effort at THI Ingolstadt / CARISSMA C-ISAFE. — YOLO26-seg integration, radar preprocessing, the finalized 7-channel frustum extractor, the custom nuScenes-trained model, and the visualization node. The pipeline that exists today exists because of what the team built together.

- **Supervisor:** Dr. Thiago de Borba (CARISSMA C-ISAFE, THI Ingolstadt)
- **Open-source foundation:** [fregu856/3DOD_thesis](https://github.com/fregu856/3DOD_thesis) — the pretrained Frustum-PointNet that the initial AI inference backbone is built on.

See [`acknowledgements.md`](acknowledgements.md) for the full credit sheet.

## License

[MIT](LICENSE) — for the code I authored. Third-party components retain their original licenses. Code in `src_finalized/` is jointly attributed to the Project 008 team.

