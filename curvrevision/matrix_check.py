from __future__ import annotations

import argparse
from pathlib import Path

from .utils import dump_json


def expected_runs(runs_root: Path, seeds: list[int]) -> list[Path]:
    folds = range(5)
    paths: list[Path] = []

    def add(experiment: str, label: str, budget: str, selection: str, seed: int, fold: int) -> None:
        paths.append(
            runs_root
            / experiment
            / label
            / f"budget_{budget}"
            / f"selection_{selection}"
            / f"seed_{seed}"
            / f"fold_{fold}"
        )

    core = ["unet", "attention_unet", "resunet", "attention_resunet", "full_standard", "full_log_attention_only", "full_log_shallow"]
    priors = ["prior_none", "prior_sobel", "prior_laplacian", "prior_gaussian_gradient", "prior_log"]
    depths = ["depth_shallow", "depth_middle", "depth_deep", "depth_all"]
    sensitivity = ["full_standard", "full_log_attention_only", "full_log_shallow"]

    for seed in seeds:
        for fold in folds:
            for model in core:
                add("core", model, "stage1", "best_dice", seed, fold)
                add("core", model, "matched48", "best_dice", seed, fold)
            for label in priors:
                add("prior", label, "stage1", "best_dice", seed, fold)
            for label in depths:
                add("depth", label, "stage1", "best_dice", seed, fold)
            add("core", "full_log_shallow", "original48", "best_dice", seed, fold)
            for model in sensitivity:
                add("core", model, "stage1", "constrained_hd", seed, fold)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify that the complete pre-specified formal revision matrix is refit before protocol freeze.")
    ap.add_argument("--runs-root", default="outputs/runs")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    ap.add_argument("--report", default="outputs/formal_matrix_check.json")
    args = ap.parse_args()

    root = Path(args.runs_root)
    expected = expected_runs(root, args.seeds)
    missing: list[str] = []
    malformed: list[str] = []
    for rd in expected:
        if not (rd / "REFIT_COMPLETE").exists():
            missing.append(str(rd))
            continue
        required = ["run_config.json", "selection.json", "refit_manifest.json", "final_outer_train_model.pth"]
        absent = [name for name in required if not (rd / name).exists()]
        if absent:
            malformed.append(f"{rd}: missing {', '.join(absent)}")

    discovered = sorted(p.parent for p in root.rglob("REFIT_COMPLETE")) if root.exists() else []
    expected_set = {str(p) for p in expected}
    extras = [str(p) for p in discovered if str(p) not in expected_set]
    report = {
        # The formal root must contain exactly the pre-specified matrix. Pilot
        # runs belong under outputs/pilot_runs and are not allowed here.
        "status": "PASS" if not missing and not malformed and not extras else "FAIL",
        "seeds": args.seeds,
        "expected_refit_runs": len(expected),
        "completed_expected_runs": len(expected) - len(missing) - len(malformed),
        "missing_count": len(missing),
        "malformed_count": len(malformed),
        "extra_refit_runs_not_in_formal_matrix": len(extras),
        "missing": missing,
        "malformed": malformed,
        "extras": extras,
    }
    dump_json(report, args.report)
    print(f"Expected formal refit runs: {len(expected)}")
    print(f"Completed expected runs:    {report['completed_expected_runs']}")
    print(f"Missing:                    {len(missing)}")
    print(f"Malformed:                  {len(malformed)}")
    print(f"Extra completed runs:       {len(extras)}")
    print(f"Report: {args.report}")
    if missing:
        print("First missing runs:")
        for x in missing[:20]:
            print("  ", x)
    if malformed:
        print("First malformed runs:")
        for x in malformed[:20]:
            print("  ", x)
    if extras:
        print("Unexpected completed runs in formal root (move/delete before freeze):")
        for x in extras[:20]:
            print("  ", x)
    if missing or malformed or extras:
        raise SystemExit(2)
    print("FORMAL MATRIX COMPLETE — safe to clean temporary development checkpoints and freeze.")


if __name__ == "__main__":
    main()
