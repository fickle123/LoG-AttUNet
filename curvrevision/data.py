from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import math
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage as ndi
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Case:
    case_id: str
    image_path: str
    annotation_path: str
    width: int
    height: int


def discover_hc18_cases(data_root: str | Path) -> list[Case]:
    root = Path(data_root)
    train_dir = root / "training_set"
    if not train_dir.exists():
        raise FileNotFoundError(f"HC18 training_set not found: {train_dir}")
    anns = sorted(train_dir.glob("*_Annotation.png"))
    cases: list[Case] = []
    for ann in anns:
        stem = ann.name.replace("_Annotation.png", "")
        candidates = [train_dir / f"{stem}.png", train_dir / f"{stem}.jpg", train_dir / f"{stem}.jpeg"]
        image = next((p for p in candidates if p.exists()), None)
        if image is None:
            continue
        with Image.open(image) as im:
            w, h = im.size
        cases.append(Case(stem, str(image), str(ann), w, h))
    if len(cases) != 999:
        raise RuntimeError(f"Expected 999 labelled HC18 cases, found {len(cases)} in {train_dir}")
    return cases


def annotation_points(annotation_path: str | Path) -> np.ndarray:
    arr = cv2.imread(str(annotation_path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise FileNotFoundError(annotation_path)
    ys, xs = np.nonzero(arr > 0)
    return np.column_stack([xs, ys]).astype(np.float32)


def ellipse_mask_from_annotation(annotation_path: str | Path, return_info: bool = False):
    arr = cv2.imread(str(annotation_path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise FileNotFoundError(annotation_path)
    h, w = arr.shape
    pts = annotation_points(annotation_path)
    mask = np.zeros((h, w), np.uint8)
    info = {"fit_method": None, "n_points": int(len(pts)), "touches_border": False, "fit_error": None}
    if len(pts):
        info["touches_border"] = bool(np.any((pts[:,0] <= 0) | (pts[:,0] >= w-1) | (pts[:,1] <= 0) | (pts[:,1] >= h-1)))
    try:
        if len(pts) < 5:
            raise ValueError("fewer than 5 contour pixels")
        ellipse = cv2.fitEllipse(pts.reshape(-1,1,2))
        cv2.ellipse(mask, ellipse, 1, thickness=-1)
        info["fit_method"] = "ellipse"
        info["ellipse"] = {
            "center": [float(ellipse[0][0]), float(ellipse[0][1])],
            "axes": [float(ellipse[1][0]), float(ellipse[1][1])],
            "angle": float(ellipse[2]),
        }
    except Exception as e:
        info["fit_error"] = repr(e)
        if len(pts) >= 3:
            hull = cv2.convexHull(pts.astype(np.int32).reshape(-1,1,2))
            cv2.fillConvexPoly(mask, hull, 1)
            info["fit_method"] = "convex_hull"
        else:
            info["fit_method"] = "failed"
    return (mask, info) if return_info else mask


def signed_distance(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    if not m.any():
        return np.ones(mask.shape, np.float32)
    outside = ndi.distance_transform_edt(~m)
    inside = ndi.distance_transform_edt(m)
    sdf = outside.astype(np.float32)
    sdf[m] = -inside[m].astype(np.float32)
    denom = float(np.max(np.abs(sdf)))
    if denom > 0:
        sdf /= denom
    return sdf


def _augment(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator):
    h, w = image.shape
    if rng.random() < 0.5:
        image = np.fliplr(image).copy(); mask = np.fliplr(mask).copy()
    angle = rng.uniform(-20, 20)
    scale = rng.uniform(0.90, 1.10)
    tx = rng.uniform(-0.05, 0.05) * w
    ty = rng.uniform(-0.05, 0.05) * h
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
    M[:,2] += [tx, ty]
    image = cv2.warpAffine(image, M, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    mask = cv2.warpAffine(mask.astype(np.uint8), M, (w,h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    alpha = rng.uniform(0.90, 1.10)
    beta = rng.uniform(-0.05, 0.05)
    image = np.clip(image * alpha + beta, 0, 1)
    gamma = rng.uniform(0.90, 1.10)
    image = np.clip(image, 1e-6, 1.0) ** gamma
    if rng.random() < 0.20:
        noise = rng.normal(0.0, 0.03, size=image.shape).astype(np.float32)
        image = np.clip(image + image * noise, 0, 1)
    return image.astype(np.float32), (mask > 0).astype(np.uint8)


class HC18Dataset(Dataset):
    def __init__(self, cases: Sequence[Case], image_size: int = 256, augment: bool = False, seed: int = 0, with_sdf: bool = True):
        self.cases = list(cases)
        self.image_size = int(image_size)
        self.augment = augment
        self.seed = int(seed)
        self.with_sdf = with_sdf
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self): return len(self.cases)

    def __getitem__(self, idx: int):
        c = self.cases[idx]
        image = cv2.imread(c.image_path, cv2.IMREAD_GRAYSCALE)
        if image is None: raise FileNotFoundError(c.image_path)
        mask = ellipse_mask_from_annotation(c.annotation_path)
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        if self.augment:
            rng = np.random.default_rng(self.seed * 1000003 + self.epoch * 1009 + idx)
            image, mask = _augment(image, mask, rng)
        sdf = signed_distance(mask) if self.with_sdf else np.zeros_like(image, np.float32)
        return {
            "image": torch.from_numpy(image[None].astype(np.float32)),
            "mask": torch.from_numpy(mask[None].astype(np.float32)),
            "sdf": torch.from_numpy(sdf[None].astype(np.float32)),
            "case_id": c.case_id,
            "orig_hw": torch.tensor([c.height, c.width], dtype=torch.int64),
        }


def robust_metadata_table(data_root: str | Path) -> pd.DataFrame:
    root = Path(data_root)
    path = root / "training_pixel_size.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_metadata_columns(df: pd.DataFrame) -> dict[str, str | None]:
    lower = {c: c.lower().strip() for c in df.columns}
    def pick(tokens):
        for c, lc in lower.items():
            if all(t in lc for t in tokens): return c
        return None
    name = pick(["filename"]) or pick(["file"]) or pick(["image"])
    pixel = pick(["pixel", "size"]) or pick(["spacing"])
    hc = pick(["head", "circumference"]) or pick(["circumference"])
    sx = pick(["pixel", "x"]) or pick(["spacing", "x"])
    sy = pick(["pixel", "y"]) or pick(["spacing", "y"])
    return {"name": name, "pixel": pixel, "hc": hc, "sx": sx, "sy": sy}
