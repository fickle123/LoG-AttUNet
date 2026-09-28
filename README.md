# Curvature-AttUNet revision v2 — independent Scientific Reports experiment project

This project is deliberately isolated from all historical `Attention-ResUnet-main` result directories. It was built for the Scientific Reports major revision and addresses the central methodological concerns: selection leakage, small-effect statistics, unmatched training budget, LoG isolation, fixed-prior/depth ablations, checkpoint sensitivity, HD95/failure handling, physical-unit HC validation, annotation/ellipse audit, practical efficiency, and reproducibility provenance.

## 1. Expected location

Extract inside the existing server project root:

```text
~/experiments/Attention-ResUnet-main/
├── fetal_dataset/                         # existing HC18 data; not included here
└── curvature_revision_v2_20260905/        # this archive
```

Default data root from the new project is therefore `../fetal_dataset`.

## 2. The formal protocol

1. The 999 labelled HC18 images are divided into five fixed **outer folds**.
2. For each fold, only outer-training images are split into inner-train/inner-val.
3. Inner validation selects the number of training epochs. The outer-test fold is never consulted.
4. A fresh model is then **refit from scratch on all outer-training images** for the selected epoch count.
5. All intended models/ablations must finish refit before any outer-test evaluation.
6. `scripts/18_freeze_protocol.sh` records hashes of code, splits, selections and final outer-train checkpoints.
7. Only after that freeze does `scripts/19_run_all_outer_tests.sh` permit a one-time outer-test evaluation.
8. Any code/split/checkpoint change after freeze blocks testing.

This is stricter than the submitted fixed 799/200 protocol and is designed to prevent both code-level and human-in-the-loop use of final folds for tuning.

## 3. Fixed evaluation rules

- input: `256×256`, single-channel grayscale;
- segmentation threshold: `0.5`;
- main segmentation metrics: raw thresholded mask, **no LCC/hole filling**;
- metrics: Dice, IoU, maximum HD, HD95 and ASD;
- empty prediction when GT is non-empty: explicitly flagged as failure and assigned the image-diagonal penalty for HD/HD95/ASD;
- HC only: restore to original image dimensions, largest connected component + hole filling, physical-coordinate ellipse fitting, explicit fit-failure log;
- primary checkpoint rule: best inner-validation Dice;
- sensitivity rule: historical Dice-constrained lowest-HD selector (`0.0003` Dice tolerance), never the primary result.

## 4. Clean LoG isolation

The package separates three questions that were previously entangled:

- `full_standard`: full multi-task architecture with standard learned attention and a fixed zero auxiliary channel to the boundary head;
- `full_log_attention_only`: **identical trainable capacity and heads**, but LoG guidance is injected into shallow attention only; the boundary path is still held fixed. This is the clean component-isolation experiment requested by the editor;
- `full_log_shallow`: submitted final architecture, where shallow LoG guidance is used in attention and the shallow LoG response is also supplied to the boundary residual head.

“no prior” in the prior ablation still means a learned standard attention gate; it is **not** confused with “no attention”. The no-attention residual control uses an explicit pass-through gate.

## 5. Controlled principal baselines

The formal nested-CV core includes controlled reimplementations of `unet`, `attention_unet`, `resunet`, and `attention_resunet`, in addition to the matched full-architecture controls. These four baselines use the same data folds, augmentation, threshold and optimization budget. They are labeled as **controlled reimplementations**, not upstream-official implementations. This directly covers the principal U-Net-family comparators discussed in the submitted manuscript while avoiding the old ambiguity about non-U-Net “style” implementations.

## 6. Training budgets

Two main comparisons are produced:

- **architecture-only**: stage 1 only (up to 40 epochs, epoch selected inside outer-training data);
- **matched-budget**: every core comparator receives the same additional low-LR stage-2 opportunity, with full-model continuation for fairness.

For this matched-budget branch, every model keeps all parameters trainable and uses the same two-rate stage-2 optimizer convention (`1e-5` for attention/output/multi-task modules and `5e-6` for the remaining encoder/decoder parameters; weight decay `1e-6`, cosine decay, no augmentation). The submitted frozen-encoder schedule is kept only in the separate `original48` analysis.

The submitted frozen-encoder second-stage design is retained as a separate `original48` analysis for the proposed final architecture, not mixed into the fair architecture comparison.

## 7. First commands

```bash
cd ~/experiments/Attention-ResUnet-main/curvature_revision_v2_20260905

# use the existing environment if dependencies are already present
python -m pip install -r requirements.txt

bash scripts/00_preflight.sh
bash scripts/01_make_splits.sh
bash scripts/02_annotation_audit.sh
```

If a genuine patient/pregnancy mapping is available:

```bash
GROUPS_CSV=/absolute/path/case_to_subject.csv bash scripts/01_make_splits.sh
```

If no such mapping exists, the code uses official-HC-stratified image-level nested CV and prints/logs that limitation rather than claiming patient-level splitting.

## 8. Formal run order

```bash
# Development + refit only; NO outer-test yet
bash scripts/10_run_core_stage1.sh
bash scripts/11_run_core_matched48.sh
bash scripts/12_run_prior_ablation.sh
bash scripts/13_run_depth_ablation.sh
bash scripts/14_run_original_method48.sh
bash scripts/15_run_selection_sensitivity.sh

# verify the complete pre-specified 3-seed × 5-fold formal matrix (exactly 405 refit runs; extras also fail)
bash scripts/16_check_formal_matrix.sh

# after stage-1 checkpoints have been reused by the stage-2 branches
bash scripts/17_cleanup_development_checkpoints.sh

# one-time pre-test freeze
bash scripts/18_freeze_protocol.sh

# now—and only now—evaluate every frozen outer fold
bash scripts/19_run_all_outer_tests.sh

# aggregate and produce revision evidence
bash scripts/20_collect_oof.sh
bash scripts/21_statistics.sh
bash scripts/22_efficiency.sh
bash scripts/23_hc_analysis.sh
bash scripts/24_prior_depth_statistics.sh
bash scripts/25_selection_statistics.sh
```

See `docs/RUNBOOK.md` for monitoring, resume and pilot instructions.

## 9. Efficiency of the experiment workflow

The core stage-1 inner-development history/checkpoint is temporarily retained and reused by matched-budget and original frozen-stage-2 branches. This avoids retraining the exact same 40-epoch inner-development run. The selection-sensitivity branch also reuses the same deterministic inner-validation history and changes only the pre-specified selection rule. Final outer-training models are still refit independently from scratch for each formal branch.

Interrupted develop/refit phases save a single rolling checkpoint and resume from the next epoch. Completed phases are skipped safely when scripts are rerun.

## 10. Statistical outputs

For formal paired testing, the three seed-specific metrics for each image are averaged first. Seeds are **not** treated as 3× independent samples. The package reports:

- per-image paired mean/median differences;
- 10,000-resample bootstrap 95% CIs;
- two-sided Wilcoxon signed-rank tests;
- paired sign-flip permutation tests;
- Holm multiplicity correction;
- rank-biserial effect size;
- five-outer-fold variability and fold-level 95% CIs.

The collector refuses to regard a model/seed/rule as complete unless it has exactly 999 unique OOF cases across five folds.

## 11. Annotation and HC evidence

`annotation_audit.py` records for every case:

- ellipse fit or fallback status;
- annotation border contact;
- physical ellipse-derived HC vs official HC;
- point-to-ellipse geometric deviation;
- whether the raw annotation encloses an area;
- ellipse-mask vs direct closed-annotation fill overlap/distance where applicable.

`hc_analysis.py` uses the model's OOF predictions, restores each to original resolution, performs HC-specific LCC/hole filling, fits in physical coordinates using HC18 pixel spacing, compares to official HC in mm, and outputs Bland–Altman plots and explicit fit-failure counts.

## 12. Baseline naming warning

The controlled baselines included here are **not claimed to be official external implementations**. The prior audit showed the old non-U-Net models were local style/uniform-framework reimplementations. If Segmenter, BiSeNetV2 or STDC-Seg remain in the revised manuscript, either label those runs accurately as controlled reimplementations or separately rerun verified upstream implementations. See `docs/OFFICIAL_BASELINE_POLICY.md`.

## 13. Code release

The editor requires a DOI-assigning code repository for the bespoke pipeline. This project writes exact split files, environment data, run configs, checkpoint hashes and a pre-test frozen protocol manifest so the final cleaned package can be deposited on Zenodo. See `docs/ZENODO_RELEASE_CHECKLIST.md`.
