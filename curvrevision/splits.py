from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split

from .data import discover_hc18_cases, find_metadata_columns, robust_metadata_table
from .utils import dump_json, sha256_file


def load_groups(groups_csv: str | None, case_ids: list[str]) -> dict[str, str] | None:
    if not groups_csv:
        return None
    df = pd.read_csv(groups_csv)
    if df.shape[1] < 2:
        raise ValueError("groups CSV needs case_id,group_id columns")
    mapping = dict(zip(df.iloc[:, 0].astype(str), df.iloc[:, 1].astype(str)))
    missing = [c for c in case_ids if c not in mapping]
    if missing:
        raise ValueError(f"groups CSV missing {len(missing)} cases, first={missing[:5]}")
    return mapping


def _metadata_hc_map(data_root: str | Path) -> dict[str, float]:
    df = robust_metadata_table(data_root)
    cols = find_metadata_columns(df)
    if cols["name"] is None or cols["hc"] is None:
        raise RuntimeError("training_pixel_size.csv must contain image filename and official head-circumference columns for stratification")
    out: dict[str, float] = {}
    for _, row in df.iterrows():
        name = Path(str(row[cols["name"]]).strip()).stem
        try:
            hc = float(row[cols["hc"]])
        except Exception:
            continue
        if np.isfinite(hc):
            out[name] = hc
    return out


def hc_strata(data_root: str | Path, case_ids: list[str], bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    mapping = _metadata_hc_map(data_root)
    missing = [c for c in case_ids if c not in mapping]
    if missing:
        raise RuntimeError(f"Official HC metadata missing for {len(missing)} cases; first={missing[:5]}")
    hc = np.asarray([mapping[c] for c in case_ids], dtype=float)
    # Quantile strata reduce fold-to-fold differences in fetal-head size without using image outcomes.
    # duplicates='drop' handles repeated HC values safely.
    y = np.asarray(pd.qcut(hc, q=min(bins, len(np.unique(hc))), labels=False, duplicates="drop"), dtype=int)
    if len(np.unique(y)) < 2:
        raise RuntimeError("Could not form at least two HC strata")
    return hc, y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="../fetal_dataset")
    ap.add_argument("--output", default="outputs/splits")
    ap.add_argument("--outer-folds", type=int, default=5)
    ap.add_argument("--inner-val-fraction", type=float, default=0.20)
    ap.add_argument("--split-seed", type=int, default=2026)
    ap.add_argument("--strata-bins", type=int, default=10)
    ap.add_argument("--groups-csv", default=None)
    args = ap.parse_args()

    cases = discover_hc18_cases(args.data_root)
    ids = [c.case_id for c in cases]
    hc, strata = hc_strata(args.data_root, ids, args.strata_bins)
    groups = load_groups(args.groups_csv, ids)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    ids_arr = np.asarray(ids)

    if groups:
        g = np.asarray([groups[i] for i in ids])
        splitter = StratifiedGroupKFold(n_splits=args.outer_folds, shuffle=True, random_state=args.split_seed)
        outer = splitter.split(ids_arr, strata, groups=g)
        mode = "stratified_grouped"
    else:
        splitter = StratifiedKFold(n_splits=args.outer_folds, shuffle=True, random_state=args.split_seed)
        outer = splitter.split(ids_arr, strata)
        mode = "stratified_image_level"

    for fold, (train_idx, test_idx) in enumerate(outer):
        outer_train = ids_arr[train_idx].tolist()
        outer_test = ids_arr[test_idx].tolist()
        train_strata = strata[train_idx]

        if groups:
            train_groups = np.asarray([groups[x] for x in outer_train])
            # A single fold of a seeded 5-way stratified grouped split gives ~20% inner validation
            # while ensuring no subject group appears in both inner-train and inner-val.
            n_inner = max(2, int(round(1.0 / args.inner_val_fraction)))
            inner_splitter = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=args.split_seed + 100 + fold)
            ii_train, ii_val = next(inner_splitter.split(np.asarray(outer_train), train_strata, groups=train_groups))
            inner_train = np.asarray(outer_train)[ii_train].tolist()
            inner_val = np.asarray(outer_train)[ii_val].tolist()
        else:
            inner_train, inner_val = train_test_split(
                outer_train,
                test_size=args.inner_val_fraction,
                random_state=args.split_seed + 100 + fold,
                shuffle=True,
                stratify=train_strata,
            )

        tr_set, va_set, te_set = set(inner_train), set(inner_val), set(outer_test)
        if tr_set & va_set or tr_set & te_set or va_set & te_set:
            raise RuntimeError(f"Leakage detected while constructing fold {fold}")
        if set(outer_train) != tr_set | va_set:
            raise RuntimeError(f"Inner split does not partition outer_train for fold {fold}")
        if groups:
            gt = {groups[x] for x in inner_train}
            gv = {groups[x] for x in inner_val}
            go = {groups[x] for x in outer_test}
            if gt & gv or gt & go or gv & go:
                raise RuntimeError(f"Subject/group leakage detected in fold {fold}")

        id_to_hc = dict(zip(ids, hc))
        record = {
            "fold": fold,
            "mode": mode,
            "stratification": "official_HC_mm_quantiles",
            "strata_bins_requested": args.strata_bins,
            "inner_train": sorted(inner_train),
            "inner_val": sorted(inner_val),
            "outer_train": sorted(outer_train),
            "outer_test": sorted(outer_test),
        }
        p = out / f"fold_{fold}.json"
        dump_json(record, p)
        rows.append(
            {
                "fold": fold,
                "mode": mode,
                "inner_train": len(inner_train),
                "inner_val": len(inner_val),
                "outer_train": len(outer_train),
                "outer_test": len(outer_test),
                "outer_test_hc_mean_mm": float(np.mean([id_to_hc[x] for x in outer_test])),
                "outer_test_hc_sd_mm": float(np.std([id_to_hc[x] for x in outer_test], ddof=1)),
                "sha256": sha256_file(p),
            }
        )

    manifest = pd.DataFrame(rows)
    manifest.to_csv(out / "split_manifest.csv", index=False)
    dump_json(
        {
            "n_cases": len(ids),
            "outer_folds": args.outer_folds,
            "inner_val_fraction": args.inner_val_fraction,
            "split_seed": args.split_seed,
            "strata_bins_requested": args.strata_bins,
            "mode": mode,
            "stratification": "official_HC_mm_quantiles",
            "groups_csv": args.groups_csv,
        },
        out / "protocol.json",
    )
    print(manifest.to_string(index=False))
    if not groups:
        print("WARNING: No subject/pregnancy IDs supplied. Splits are stratified at image level; manuscript must state that patient-level grouping could not be verified.")


if __name__ == "__main__":
    main()
