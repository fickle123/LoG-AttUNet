# LoG-AttUNet

**LoG-Derived Structural Response-Guided Attention for Fetal Head Ultrasound Segmentation**

This repository provides the implementation of **LoG-AttUNet**, a residual
attention U-Net designed for fetal-head ultrasound segmentation.

The method introduces a fixed Laplacian-of-Gaussian (LoG)-derived local
structural response into shallow skip-attention gates, allowing explicit
second-order structural information to participate in feature selection before
learned attention weights are generated. Deeper skip connections retain
standard learned attention for semantic feature selection.

The framework further incorporates signed-distance-field (SDF) supervision,
deep supervision, and boundary-residual refinement for contour-aware
segmentation.

This repository contains the core source code and evaluation pipeline.
Pretrained weights, prediction files, figures, and experimental results are not
included.

---

## Method overview

LoG-AttUNet combines four main components:

- **LoG-derived structural response-guided attention** at shallow skip levels;
- **standard learned attention** at deeper semantic levels;
- **signed-distance-field supervision and deep supervision**;
- **boundary-residual refinement** for final contour prediction.

The LoG branch is implemented using fixed Gaussian and Laplacian operators and
does not introduce additional trainable parameters.

The structural response used in this work is a normalized LoG-derived
second-order response and is not intended to represent
differential-geometric curvature.

---

## Repository structure

```text
LoG-AttUNet/
├── README.md
├── requirements.txt
├── configs/
│   └── ...
├── curvrevision/
│   ├── ...
│   └── ...
├── scripts/
│   ├── 00_preflight.sh
│   ├── 01_make_splits.sh
│   ├── 02_annotation_audit.sh
│   ├── 10_run_core_stage1.sh
│   ├── 11_run_core_matched48.sh
│   ├── 12_run_prior_ablation.sh
│   ├── 13_run_depth_ablation.sh
│   ├── 14_run_original_method48.sh
│   ├── 15_run_selection_sensitivity.sh
│   ├── 19_run_all_outer_tests.sh
│   ├── 20_collect_oof.sh
│   ├── 21_statistics.sh
│   ├── 22_efficiency.sh
│   ├── 23_hc_analysis.sh
│   └── ...
└── tests/
    └── ...
