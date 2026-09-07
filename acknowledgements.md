# Acknowledgements

Project 008 was a team effort at Technische Hochschule Ingolstadt / CARISSMA C-ISAFE. This repository publishes only the code and documentation I personally wrote. The finalized system that we demonstrated together also includes work by my teammates that is not included here.

## Supervisor

**Dr. Thiago de Borba** — CARISSMA C-ISAFE, THI Ingolstadt.
For giving the space to work through problems rather than around them, and for trusting the process even when the process was ugly.

## Team

Project 008 was carried out by four students at THI Ingolstadt. The full finalized pipeline exists because of what we built together. Specific credit for the parts not covered in this repository — the 2D camera perception (YOLO26-seg integration), radar preprocessing, the finalized mask-based 7-channel frustum extractor, the custom nuScenes-trained Frustum-PointNet, and the RViz visualization node — belongs to whole team. I've kept their names off the public README for privacy.

## Open-source foundations

**[fregu856/3DOD_thesis](https://github.com/fregu856/3DOD_thesis)** — Fridrik Kristjansson Gustavsson's open-source Frustum-PointNet implementation and pretrained KITTI weights (`model_37_2_epoch_400.pth`). The AI inference backbone in `src/car_inference/` and the offline analysis tooling in `src/offline_tools/` are built on this repository. Without it, the initial version of this pipeline would not have existed.

**[PyTorch](https://pytorch.org/)** — the framework the inference runs on.

## Reference material

- The **KITTI Vision Benchmark Suite** — the dataset on which Fridrik's weights were trained, and the reference distribution the initial model expects.
- The **nuScenes dataset** — used by the team for the finalized model.
- The **Velodyne and Ouster manuals** — for sensor specifications referenced during debugging (intensity scaling, beam patterns).
