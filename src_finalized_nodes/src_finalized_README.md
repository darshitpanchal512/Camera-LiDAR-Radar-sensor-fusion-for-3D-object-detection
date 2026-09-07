# `src_finalized/` — The Finalized Team Pipeline

**Read this section first.** This folder contains the finalized version of the Project 008 sensor-fusion pipeline — the version that ran in real time on the ANTON platform and produced the results described in the top-level README. **This code is the collective work of the Project 008 team; it is not solely my authorship.** I've included it here because it is the necessary counterpart to the initial-version work in [`src/`](../src/) — you cannot fully understand my contribution without seeing what it fed into. Individual file authorship is listed below.

If you are evaluating this repository for my portfolio, the primary code showing what I built is in [`src/`](../src/). This folder is the *context* for that work.

---

## What's in this folder

Five Python nodes that make up the finalized ROS 2 pipeline:

| File | Role |
|---|---|
| `yolo_seg_node_1.py` | Runs YOLO26-seg (TensorRT-accelerated) on the camera stream. Publishes three parallel outputs: a debug-visualization image, a hidden `mono8` instance-mask stream, and a `Detection2DArray` of 2D boxes with class IDs. |
| `lidar_radar_preprocessor_node.py` | Cleans and unifies both LiDAR and radar streams. LiDAR side: forward-facing ROI crop, 10 cm voxel downsampling, ground removal, intensity preservation. Radar side: Doppler velocity threshold filter (\|v\| > 0.05 m/s) removing static clutter, compensated velocity retention. |
| `live_frustum_extractor_node.py` | The finalized frustum extractor — the piece that replaced my initial 2D-box version. Uses pixel-perfect instance masks (with dynamic mask dilation: ~20 px for cars, ~5 px for pedestrians) to carve mask-shaped frustums from the merged LiDAR + radar cloud. Constructs the custom 7-channel tensor `[X, Y, Z, Intensity/RCS, Vx, Vy, Modality flag]` and samples to exactly 1024 points via radar-preserving sampling. Publishes as `Float32MultiArray` on `/fusion/ready_frustums`. |
| `frustum_model_7d.py` | The finalized model definition — a custom 7-dimensional Frustum-PointNet architecture with T-Net (spatial alignment), SegNet (per-point foreground/background segmentation), and BoxNet (amodal 3D box regression). Trained on the nuScenes multi-modal dataset. |
| `frustum_inference_node.py` | The finalized inference node. Loads the trained `.pth` checkpoint, runs `forward()` on incoming 7D frustums, decodes the model outputs into 3D bounding boxes with class-aware size blending, applies geometric overrides (asphalt clamp, class-specific Z-offsets, occlusion-aware tracking), and publishes `MarkerArray` messages for RViz. |

## Authorship — please read

Project 008 was a four-person team effort. My personal ownership within the pipeline covered the AI backbone: LiDAR preprocessing on the initial version, the initial frustum extractor and its analysis, the KITTI-weights Frustum-PointNet inference node, and the offline analysis tooling. Those pieces live in [`src/`](../src/) and are the primary evidence of my authorship.

The finalized files in this folder — the mask-based extractor, the 7D model architecture, the finalized inference node, the YOLO26-seg integration, and the radar preprocessing — were built collaboratively by the team, with different pieces owned primarily by different teammates. I contributed to the design discussions and to shared debugging, but I am not the sole author of these files.

I am publishing this folder for **reference and completeness**, not as a portfolio artefact of my personal work. If you are assessing my technical contribution, please look at [`src/`](../src/) and [`docs/02_my_contributions.md`](../docs/02_my_contributions.md) — those describe what I actually built.

If any of my teammates would prefer I take this folder down, I will do so immediately — please contact me.

## What the finalized pipeline replaces from the initial version

The finalized system addresses the three honest limits of the initial version documented in [`docs/02_my_contributions.md`](../docs/02_my_contributions.md):

- **The 2D bounding box replaced by pixel-perfect instance masks** — the frustum now hugs the exact car silhouette rather than a rectangle behind it. No more background scooped in with the vehicle. The initial version's ~20 m depth cap is no longer needed.
- **KITTI-trained weights replaced by nuScenes-trained weights on a custom 7-channel input** — the model now sees the same distribution the data comes from. The domain gap the depth cap was hiding is closed at the source rather than worked around.
- **LiDAR-only frustums replaced by fused LiDAR + radar frustums** — the model now reasons over velocity, not just position, which helps with occlusion and cross-traffic cases where geometry alone is ambiguous.

The succession is the point: the initial version made these limitations *visible* and *measurable*, which is what motivated the design of the finalized system. Both versions are needed to see the full engineering arc.

## Running this code

I have **not** made the finalized scripts portable in the way I did for the initial version — the environment variables, standalone launch configuration, and dependency documentation in [`CONFIG.md`](../CONFIG.md) apply only to `src/`. The finalized scripts in this folder retain their original in-container paths and were designed to run inside the team's specific development environment.

If you want to actually execute the finalized pipeline, expect meaningful integration work: the finalized model weights are not distributed, the ANTON bag data is not distributable, and the calibration files and TensorRT engines are specific to that hardware. Treat this folder as **reference code**, not a runnable release.

## Third-party dependencies

The finalized pipeline builds on the same open-source foundations as the initial version — see [`acknowledgements.md`](../acknowledgements.md) for full credit — plus:

- **nuScenes dataset** — for training the finalized 7D model.
- **YOLO26 / Ultralytics** — for the 2D instance-segmentation frontend.
- **OpenPCDet** — architectural conventions used in the exploratory Path B work referenced in the presentation.
- **TensorRT** — for accelerating the YOLO frontend.

## License

The code in this folder is subject to the same [MIT license](../LICENSE) as the rest of this repository, applied jointly to all contributors. I am not asserting sole ownership over these files, and I acknowledge each teammate's contribution as their own.
