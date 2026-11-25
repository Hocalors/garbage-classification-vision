"""Headless Google Colab execution script for Garbage Classification v2 (Step 1).

Preserves the exact Step 1 high-quality test-set locking, near-duplicate exclusion,
leakage firewall, two-stage ConvNeXt-Tiny training, Grad-CAM error analysis,
and optional RealWaste zero-shot OOD evaluation.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys


def ensure_dependencies() -> None:
    required = [
        ("timm", "timm"),
        ("kagglehub", "kagglehub"),
        ("imagehash", "imagehash"),
        ("pytorch_grad_cam", "grad-cam"),
        ("sklearn", "scikit-learn"),
        ("cv2", "opencv-python-headless"),
        ("seaborn", "seaborn"),
        ("PIL", "Pillow"),
        ("tqdm", "tqdm"),
    ]
    missing = [pkg for mod, pkg in required if importlib.util.find_spec(mod) is None]
    if missing:
        print(f"[Setup] Installing missing packages: {missing}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


ensure_dependencies()

import matplotlib

matplotlib.use("Agg")

import hashlib
import json
import math
import os
import random
import shutil
import tarfile
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import imagehash
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import timm
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from tqdm.auto import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

REALWASTE_TO_V2 = {
    "Cardboard": "cardboard",
    "Food Organics": "biological",
    "Vegetation": "biological",
    "Glass": "glass",
    "Metal": "metal",
    "Miscellaneous Trash": "trash",
    "Paper": "paper",
    "Plastic": "plastic",
    "Textile Trash": "clothes",
}


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def locate_class_root(download_dir: str | Path, expected_classes: list[str]) -> Path:
    """Recursively locate the directory that directly contains the expected class folders."""
    root = Path(download_dir)
    expected_set = set(expected_classes)
    candidates = [root] + [p for p in root.rglob("*") if p.is_dir()]
    best_dir = None
    best_overlap = 0
    for cand in candidates:
        subdirs = {d.name for d in cand.iterdir() if d.is_dir()}
        overlap = len(subdirs & expected_set)
        if overlap == len(expected_set):
            return cand
        if overlap > best_overlap:
            best_overlap = overlap
            best_dir = cand
    if best_dir is not None and best_overlap >= max(2, len(expected_set) // 2):
        print(f"[Warning] Matched {best_overlap}/{len(expected_set)} class dirs at: {best_dir}")
        return best_dir
    raise FileNotFoundError(f"Could not locate expected classes {expected_classes} under {download_dir}")


def compute_image_record(path_str: str, label: str) -> dict:
    """Verify image integrity, compute MD5, 64-bit pHash, and visual quality metrics."""
    path = Path(path_str)
    try:
        raw_bytes = path.read_bytes()
        md5_hex = hashlib.md5(raw_bytes).hexdigest()

        with Image.open(path) as img_verify:
            img_verify.verify()

        with Image.open(path) as img_raw:
            rgb_img = ImageOps.exif_transpose(img_raw).convert("RGB")
            width, height = rgb_img.size
            if width <= 0 or height <= 0:
                raise ValueError(f"Invalid image dimensions: {width}x{height}")

            phash_int = int(str(imagehash.phash(rgb_img)), 16)
            phash_hex = f"{phash_int:016x}"

            rgb_np = np.asarray(rgb_img, dtype=np.uint8)
            gray = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY)
            laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            brightness_mean = float(gray.mean())
            contrast_std = float(gray.std())

        return {
            "orig_path": str(path),
            "orig_name": path.name,
            "label": label,
            "status": "ok",
            "error": "",
            "md5": md5_hex,
            "phash_hex": phash_hex,
            "width": int(width),
            "height": int(height),
            "min_side": int(min(width, height)),
            "aspect_ratio": float(width / height),
            "laplacian_var": laplacian_var,
            "brightness_mean": brightness_mean,
            "contrast_std": contrast_std,
        }
    except Exception as exc:
        return {
            "orig_path": str(path),
            "orig_name": path.name,
            "label": label,
            "status": "corrupted",
            "error": repr(exc),
            "md5": "",
            "phash_hex": "",
            "width": 0,
            "height": 0,
            "min_side": 0,
            "aspect_ratio": 0.0,
            "laplacian_var": 0.0,
            "brightness_mean": 0.0,
            "contrast_std": 0.0,
        }


def scan_dataset(
    dataset_root: str | Path,
    expected_classes: list[str],
    max_workers: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scan all class subdirectories in parallel and separate valid vs corrupted images."""
    root = Path(dataset_root)
    tasks: list[tuple[str, str]] = []
    for cls_name in expected_classes:
        cls_dir = root / cls_name
        if not cls_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {cls_dir}")
        files = sorted(
            p for p in cls_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        for fpath in files:
            tasks.append((str(fpath), cls_name))

    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(compute_image_record, p, lbl) for p, lbl in tasks]
        for fut in tqdm(futures, total=len(futures), desc="Scanning images"):
            records.append(fut.result())

    full_df = pd.DataFrame(records)
    valid_df = full_df[full_df["status"] == "ok"].reset_index(drop=True)
    corrupted_df = full_df[full_df["status"] != "ok"].reset_index(drop=True)
    return valid_df, corrupted_df


def _hex_to_uint64(hex_values) -> np.ndarray:
    return np.array([int(h, 16) for h in hex_values], dtype=np.uint64)


def min_hamming_between(
    query_hex_values,
    ref_hex_values,
    chunk_size: int = 512,
) -> np.ndarray:
    """Compute minimum 64-bit Hamming distance from each query pHash to any reference pHash."""
    q_arr = _hex_to_uint64(query_hex_values)
    r_arr = _hex_to_uint64(ref_hex_values)
    if len(q_arr) == 0:
        return np.empty(0, dtype=np.int32)
    if len(r_arr) == 0:
        return np.full(len(q_arr), 64, dtype=np.int32)

    min_dists = np.full(len(q_arr), 64, dtype=np.int32)
    for start in range(0, len(q_arr), chunk_size):
        end = min(start + chunk_size, len(q_arr))
        q_chunk = q_arr[start:end]
        row_min = np.full(len(q_chunk), 64, dtype=np.int32)
        for r_start in range(0, len(r_arr), chunk_size * 4):
            r_end = min(r_start + chunk_size * 4, len(r_arr))
            xor = q_chunk[:, None] ^ r_arr[None, r_start:r_end]
            dists = POPCOUNT_LUT[xor.view(np.uint8).reshape(len(q_chunk), r_end - r_start, 8)].sum(axis=-1)
            row_min = np.minimum(row_min, dists.min(axis=1))
        min_dists[start:end] = row_min
    return min_dists


def cluster_phash_vectorized(
    phash_hex_list,
    threshold: int = 6,
    chunk_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster near-duplicate images via vectorized uint64 XOR + POPCOUNT_LUT + Union-Find."""
    arr = _hex_to_uint64(phash_hex_list)
    n = len(arr)
    if n == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    parent = np.arange(n, dtype=np.int64)
    rank = np.zeros(n, dtype=np.int32)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        sub = arr[i:end]
        tail = arr[i:]
        xor = sub[:, None] ^ tail[None, :]
        dists = POPCOUNT_LUT[xor.view(np.uint8).reshape(end - i, n - i, 8)].sum(axis=-1)

        row_offsets = np.arange(end - i)[:, None]
        col_offsets = np.arange(n - i)[None, :]
        upper_mask = col_offsets > row_offsets
        match_rows, match_cols = np.where((dists <= threshold) & upper_mask)
        for r_off, c_off in zip(match_rows.tolist(), match_cols.tolist()):
            union(i + r_off, i + c_off)

    roots = np.array([find(idx) for idx in range(n)], dtype=np.int64)
    _, cluster_ids, counts = np.unique(roots, return_inverse=True, return_counts=True)
    cluster_sizes = counts[cluster_ids]
    return cluster_ids.astype(np.int64), cluster_sizes.astype(np.int64)


def _compute_quality_metrics(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    work = df.copy()
    cls_p20 = work.groupby("label")["laplacian_var"].transform(lambda s: float(np.percentile(s, 20)))
    lap_floor = np.maximum(cfg["min_laplacian_var"], cls_p20)

    work["is_high_quality"] = (
        (work["min_side"] >= cfg["min_test_res"])
        & (work["aspect_ratio"] >= cfg["min_aspect"])
        & (work["aspect_ratio"] <= cfg["max_aspect"])
        & (work["brightness_mean"] >= cfg["min_brightness"])
        & (work["brightness_mean"] <= cfg["max_brightness"])
        & (work["contrast_std"] >= cfg["min_contrast"])
        & (work["laplacian_var"] >= lap_floor)
    )

    sharp_norm = np.clip(np.log1p(work["laplacian_var"]) / np.log1p(1000.0), 0.0, 1.0)
    res_norm = np.clip(work["min_side"] / 512.0, 0.0, 1.0)
    aspect_balance = 1.0 - np.clip(np.abs(np.log(work["aspect_ratio"].clip(lower=1e-3))) / math.log(2.5), 0.0, 1.0)
    work["quality_score"] = 0.50 * sharp_norm + 0.30 * res_norm + 0.20 * aspect_balance
    return work


def _materialize_isolated_split(df: pd.DataFrame, target_split_dir: Path) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    if target_split_dir.exists():
        shutil.rmtree(target_split_dir)
    target_split_dir.mkdir(parents=True, exist_ok=True)

    isolated_paths: list[str] = []
    for idx, row in out.iterrows():
        cls_dir = target_split_dir / str(row["label"])
        cls_dir.mkdir(parents=True, exist_ok=True)
        src = Path(row["orig_path"])
        dst = cls_dir / f"{idx:06d}_{src.name}"
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        isolated_paths.append(str(dst))

    out["isolated_path"] = isolated_paths
    return out


def _compute_lock_sha256(test_df: pd.DataFrame) -> str:
    payload = "\n".join(sorted(f"{lbl}:{md5}" for lbl, md5 in zip(test_df["label"], test_df["md5"])))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_high_quality_test_and_exclude(
    df: pd.DataFrame,
    cfg: dict,
    clean_root: str | Path,
    manifest_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select and lock a high-quality test set first, then strictly exclude it and all near-duplicates."""
    clean_path = Path(clean_root)
    manifest_path = Path(manifest_dir)
    manifest_path.mkdir(parents=True, exist_ok=True)

    work = _compute_quality_metrics(df, cfg)
    test_csv_path = manifest_path / "test_locked.csv"
    sha_path = manifest_path / "test_lock_sha256.txt"

    reused_test_indices: list[int] | None = None
    if cfg.get("reuse_existing_test_lock", True) and test_csv_path.exists():
        prev_test = pd.read_csv(test_csv_path)
        prev_pairs = set(zip(prev_test["label"].astype(str), prev_test["md5"].astype(str)))
        matched_rows = []
        used_pair = set()
        for idx, row in work.iterrows():
            pair = (str(row["label"]), str(row["md5"]))
            if pair in prev_pairs and pair not in used_pair:
                matched_rows.append(idx)
                used_pair.add(pair)
        if len(matched_rows) == len(prev_pairs) and len(matched_rows) > 0:
            candidate_test = work.loc[matched_rows]
            recomputed_sha = _compute_lock_sha256(candidate_test)
            expected_sha = sha_path.read_text(encoding="utf-8").strip() if sha_path.exists() else recomputed_sha
            if recomputed_sha == expected_sha:
                reused_test_indices = matched_rows
                print(f"[Lock Reuse] Reusing existing locked test set ({len(reused_test_indices)} images).")

    if reused_test_indices is not None:
        selected_indices = reused_test_indices
    else:
        selected_indices = []
        used_clusters: set[int] = set()
        for cls_idx, cls_name in enumerate(cfg["expected_classes"]):
            cls_df = work[work["label"] == cls_name]
            n_clusters = int(cls_df["cluster_id"].nunique())
            cap_n = max(1, int(n_clusters * 0.30))
            target_n = min(max(cfg["min_test_per_class"], int(round(n_clusters * cfg["test_ratio"]))), cap_n)
            rng = np.random.default_rng(int(cfg["seed"]) + cls_idx)

            cls_picked: list[int] = []

            p1_df = cls_df[
                cls_df["is_high_quality"]
                & (cls_df["cluster_size"] == 1)
                & (~cls_df["cluster_id"].isin(used_clusters))
            ]
            if len(p1_df) > 0:
                take_n = min(target_n, len(p1_df))
                chosen = rng.choice(p1_df.index.to_numpy(), size=take_n, replace=False).tolist()
                cls_picked.extend(chosen)
                used_clusters.update(work.loc[chosen, "cluster_id"].astype(int).tolist())

            if len(cls_picked) < target_n:
                need = target_n - len(cls_picked)
                p2_df = cls_df[cls_df["is_high_quality"] & (~cls_df["cluster_id"].isin(used_clusters))]
                if len(p2_df) > 0:
                    p2_rep = p2_df.sort_values("quality_score", ascending=False).drop_duplicates(
                        subset=["cluster_id"], keep="first"
                    )
                    take_n = min(need, len(p2_rep))
                    chosen = p2_rep.index.to_numpy()[:take_n].tolist()
                    cls_picked.extend(chosen)
                    used_clusters.update(work.loc[chosen, "cluster_id"].astype(int).tolist())

            if len(cls_picked) < target_n:
                need = target_n - len(cls_picked)
                p3_df = cls_df[~cls_df["cluster_id"].isin(used_clusters)]
                if len(p3_df) > 0:
                    p3_rep = p3_df.sort_values("quality_score", ascending=False).drop_duplicates(
                        subset=["cluster_id"], keep="first"
                    )
                    take_n = min(need, len(p3_rep))
                    chosen = p3_rep.index.to_numpy()[:take_n].tolist()
                    cls_picked.extend(chosen)
                    used_clusters.update(work.loc[chosen, "cluster_id"].astype(int).tolist())
                    print(f"[Warning] {cls_name}: fell back to {take_n} standard-quality clusters.")

            selected_indices.extend(cls_picked)

    test_raw_df = work.loc[selected_indices].copy()
    locked_clusters = set(test_raw_df["cluster_id"].astype(int).tolist())

    not_test_mask = ~work.index.isin(test_raw_df.index)
    same_cluster_mask = not_test_mask & work["cluster_id"].isin(locked_clusters)
    candidate_mask = not_test_mask & (~same_cluster_mask)

    boundary_dup_mask = pd.Series(False, index=work.index)
    if candidate_mask.any() and len(test_raw_df) > 0:
        cand_indices = work.index[candidate_mask].to_numpy()
        min_dists = min_hamming_between(
            work.loc[cand_indices, "phash_hex"].tolist(),
            test_raw_df["phash_hex"].tolist(),
            chunk_size=512,
        )
        hit_indices = cand_indices[min_dists <= int(cfg["phash_dup_threshold"])]
        boundary_dup_mask.loc[hit_indices] = True

    excluded_mask = same_cluster_mask | boundary_dup_mask
    excluded_raw_df = work.loc[excluded_mask].copy()
    excluded_raw_df["exclusion_reason"] = np.where(
        same_cluster_mask.loc[excluded_mask],
        "same_cluster_as_locked_test",
        "cross_boundary_phash_near_dup",
    )

    pool_raw_df = work.loc[not_test_mask & (~excluded_mask)].copy()

    test_df = _materialize_isolated_split(test_raw_df, clean_path / "test_locked")
    pool_df = _materialize_isolated_split(pool_raw_df, clean_path / "train_val_pool")

    classes_list = list(cfg["expected_classes"])
    class_to_idx = {c: i for i, c in enumerate(classes_list)}
    test_df["label_idx"] = test_df["label"].map(class_to_idx)
    pool_df["label_idx"] = pool_df["label"].map(class_to_idx)

    test_df.to_csv(test_csv_path, index=False)
    excluded_raw_df.to_csv(manifest_path / "excluded_near_dups_of_test.csv", index=False)
    pool_df.to_csv(manifest_path / "train_val_pool.csv", index=False)
    with open(manifest_path / "classes.json", "w", encoding="utf-8") as f:
        json.dump({"classes": classes_list, "class_to_idx": class_to_idx}, f, ensure_ascii=False, indent=2)

    lock_sha = _compute_lock_sha256(test_df)
    sha_path.write_text(lock_sha + "\n", encoding="utf-8")
    return test_df, pool_df, excluded_raw_df


def split_train_val_grouped(
    pool_df: pd.DataFrame, cfg: dict, manifest_dir: str | Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sgkf = StratifiedGroupKFold(n_splits=int(cfg["val_folds"]), shuffle=True, random_state=int(cfg["seed"]))
    train_idx, val_idx = next(
        sgkf.split(pool_df, pool_df["label_idx"].to_numpy(), groups=pool_df["cluster_id"].to_numpy())
    )
    train_df = pool_df.iloc[train_idx].copy().reset_index(drop=True)
    val_df = pool_df.iloc[val_idx].copy().reset_index(drop=True)

    manifest_path = Path(manifest_dir)
    train_df.to_csv(manifest_path / "train.csv", index=False)
    val_df.to_csv(manifest_path / "val.csv", index=False)
    return train_df, val_df


def verify_no_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    clean_root: str | Path,
    phash_dup_threshold: int = 6,
) -> int:
    clean_path = Path(clean_root).resolve()
    pool_dir = (clean_path / "train_val_pool").resolve()
    test_dir = (clean_path / "test_locked").resolve()

    splits = [("train", train_df), ("val", val_df), ("test", test_df)]
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            name_a, df_a = splits[i]
            name_b, df_b = splits[j]
            for col in ("orig_path", "isolated_path", "md5", "cluster_id"):
                overlap = set(df_a[col]) & set(df_b[col])
                assert len(overlap) == 0, f"Leakage between {name_a} and {name_b} on {col}: {len(overlap)}"

    for name, sub_df in (("train", train_df), ("val", val_df)):
        for p_str in sub_df["isolated_path"]:
            p_res = Path(p_str).resolve()
            assert p_res.is_relative_to(pool_dir), f"{name} outside train_val_pool: {p_res}"
            assert not p_res.is_relative_to(test_dir), f"{name} inside test_locked: {p_res}"

    for p_str in test_df["isolated_path"]:
        p_res = Path(p_str).resolve()
        assert p_res.is_relative_to(test_dir), f"test outside test_locked: {p_res}"

    tv_hex = pd.concat([train_df["phash_hex"], val_df["phash_hex"]], ignore_index=True).tolist()
    min_dists = min_hamming_between(tv_hex, test_df["phash_hex"].tolist(), chunk_size=512)
    min_overall = int(min_dists.min()) if len(min_dists) > 0 else 64
    assert min_overall > phash_dup_threshold, (
        f"Cross-boundary pHash leakage: {min_overall} <= {phash_dup_threshold}"
    )
    print(f"[Firewall Passed] Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)} | min_hamming={min_overall}")
    return min_overall


class GarbageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None, path_col: str = "isolated_path"):
        self.paths = df[path_col].tolist()
        self.labels = df["label_idx"].astype(int).tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        with Image.open(self.paths[idx]) as img:
            rgb = ImageOps.exif_transpose(img).convert("RGB")
        tensor = self.transform(rgb) if self.transform is not None else v2.ToDtype(torch.float32, scale=True)(v2.ToImage()(rgb))
        return tensor, self.labels[idx]


def set_backbone_trainable(model: nn.Module, trainable: bool) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    classifier = model.get_classifier() if hasattr(model, "get_classifier") else None
    head_param_ids: set[int] = set()
    if isinstance(classifier, nn.Module):
        head_param_ids = {id(p) for p in classifier.parameters()}
    else:
        for name, param in model.named_parameters():
            if any(k in name for k in ("head", "fc", "classifier")):
                head_param_ids.add(id(param))

    backbone_params, head_params = [], []
    for param in model.parameters():
        if id(param) in head_param_ids:
            param.requires_grad = True
            head_params.append(param)
        else:
            param.requires_grad = trainable
            backbone_params.append(param)
    return backbone_params, head_params


def evaluate_loader(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device, amp_dtype, tta_hflip: bool = False):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_probs, all_targets = [], []

    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=amp_dtype):
                logits = model(images)
                if tta_hflip:
                    logits = 0.5 * (logits + model(torch.flip(images, dims=[3])))
                loss = loss_fn(logits, targets)

            bs = targets.size(0)
            total_loss += float(loss.item()) * bs
            total_samples += bs
            all_probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    y_pred = probs.argmax(axis=1)
    return {
        "loss": total_loss / max(total_samples, 1),
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "probs": probs,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    scaler: torch.amp.GradScaler,
    device,
    amp_dtype,
    grad_accum_steps: int = 1,
    desc: str = "Train",
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    total_samples = 0

    for step, (images, targets) in enumerate(tqdm(loader, total=len(loader), desc=desc, leave=False)):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=amp_dtype):
            logits = model(images)
            raw_loss = loss_fn(logits, targets)
            scaled_loss = raw_loss / grad_accum_steps

        scaler.scale(scaled_loss).backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        bs = targets.size(0)
        running_loss += float(raw_loss.item()) * bs
        total_samples += bs

    return running_loss / max(total_samples, 1)


def find_gradcam_target_layer(net: nn.Module) -> nn.Module:
    if hasattr(net, "stages") and len(net.stages) > 0:
        last_stage = net.stages[-1]
        if hasattr(last_stage, "blocks") and len(last_stage.blocks) > 0:
            return last_stage.blocks[-1]
        return last_stage
    if hasattr(net, "conv_head"):
        return net.conv_head
    last_conv = None
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise ValueError("No Conv2d layer found for Grad-CAM")
    return last_conv


def download_realwaste_root(target_dir: Path) -> Path | None:
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        rw_path = Path(kagglehub.dataset_download("joarvidsson/realwaste-image-classification"))
        return locate_class_root(rw_path, list(REALWASTE_TO_V2.keys()))
    except Exception as exc:
        print(f"[Info] KaggleHub RealWaste fetch skipped ({exc}), trying UCI ZIP...")

    zip_path = target_dir / "realwaste.zip"
    extract_dir = target_dir / "extracted"
    if not extract_dir.exists():
        try:
            urllib.request.urlretrieve("https://archive.ics.uci.edu/static/public/908/realwaste.zip", zip_path)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
        except Exception as exc:
            print(f"[Warning] RealWaste download failed (OOD eval skipped): {exc}")
            return None
    try:
        return locate_class_root(extract_dir, list(REALWASTE_TO_V2.keys()))
    except Exception as exc:
        print(f"[Warning] Failed to locate RealWaste classes: {exc}")
        return None


def main() -> None:
    out_root = Path("/content/step1_outputs")
    cfg = {
        "dataset_slug": "sumn2u/garbage-classification-v2",
        "expected_classes": [
            "battery", "biological", "cardboard", "clothes", "glass",
            "metal", "paper", "plastic", "shoes", "trash",
        ],
        "model_name": "convnext_tiny.fb_in22k_ft_in1k_384",
        "seed": 42,
        "phash_dup_threshold": 6,
        "test_ratio": 0.15,
        "min_test_per_class": 30,
        "min_test_res": 224,
        "min_aspect": 0.4,
        "max_aspect": 2.5,
        "min_laplacian_var": 30.0,
        "min_brightness": 25.0,
        "max_brightness": 235.0,
        "min_contrast": 15.0,
        "reuse_existing_test_lock": True,
        "val_folds": 6,
        "stage1_epochs": 2,
        "stage1_lr": 1e-3,
        "stage2_epochs": 12,
        "backbone_lr": 1e-4,
        "head_lr": 5e-4,
        "weight_decay": 1e-2,
        "batch_size": 32,
        "grad_accum_steps": 2,
        "num_workers": 4,
        "drop_path_rate": 0.2,
        "label_smoothing": 0.05,
        "early_stop_patience": 4,
        "clean_root": "/content/garbage_v2_clean",
        "manifest_dir": str(out_root / "manifests"),
        "ckpt_dir": str(out_root / "checkpoints"),
        "eval_dir": str(out_root / "eval_outputs"),
    }

    for key in ("clean_root", "manifest_dir", "ckpt_dir", "eval_dir"):
        Path(cfg[key]).mkdir(parents=True, exist_ok=True)

    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    total_gb = (
        torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if torch.cuda.is_available()
        else 0.0
    )
    if 0.0 < total_gb < 20.0:
        cfg["model_name"] = "convnext_tiny.fb_in22k_ft_in1k"
        cfg["batch_size"] = 32
        cfg["grad_accum_steps"] = 2
        cfg["num_workers"] = 2
    print(
        f"PyTorch {torch.__version__} | Device={device} | GPU={gpu_name} ({total_gb:.1f} GB) | "
        f"model={cfg['model_name']} | batch_size={cfg['batch_size']}, grad_accum={cfg['grad_accum_steps']}, num_workers={cfg['num_workers']}"
    )

    # 1. Download & locate raw dataset
    raw_dl = kagglehub.dataset_download(cfg["dataset_slug"])
    raw_root = locate_class_root(raw_dl, cfg["expected_classes"])
    print(f"Dataset root: {raw_root}")

    # 2. Scan & cluster pHash
    valid_df, corrupted_df = scan_dataset(raw_root, cfg["expected_classes"], max_workers=8)
    cluster_ids, cluster_sizes = cluster_phash_vectorized(
        valid_df["phash_hex"].tolist(), threshold=cfg["phash_dup_threshold"], chunk_size=512
    )
    valid_df["cluster_id"] = cluster_ids
    valid_df["cluster_size"] = cluster_sizes
    if len(corrupted_df) > 0:
        corrupted_df.to_csv(Path(cfg["manifest_dir"]) / "corrupted_images.csv", index=False)

    # 3. Lock high-quality test set & strictly exclude from pool
    test_df, pool_df, excluded_df = select_high_quality_test_and_exclude(
        valid_df, cfg, clean_root=cfg["clean_root"], manifest_dir=cfg["manifest_dir"]
    )
    print(
        f"Valid={len(valid_df)}, Corrupted={len(corrupted_df)}, "
        f"TestLocked={len(test_df)}, ExcludedNearDups={len(excluded_df)}, Pool={len(pool_df)}"
    )

    # Visual audit of test_locked
    fig, axes = plt.subplots(len(cfg["expected_classes"]), 5, figsize=(12, 22))
    rng_vis = np.random.default_rng(cfg["seed"])
    for r, cls_name in enumerate(cfg["expected_classes"]):
        cls_samples = test_df[test_df["label"] == cls_name]
        pick_idx = rng_vis.choice(len(cls_samples), size=min(5, len(cls_samples)), replace=False)
        for c in range(5):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(pick_idx):
                row = cls_samples.iloc[pick_idx[c]]
                ax.imshow(Image.open(row["isolated_path"]).convert("RGB"))
                ax.set_title(f"{cls_name}\n{row['width']}x{row['height']} | lap={row['laplacian_var']:.0f}", fontsize=8)
    plt.tight_layout()
    plt.savefig(Path(cfg["eval_dir"]) / "test_locked_visual_audit.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # 4. Split train/val & verify firewall
    train_df, val_df = split_train_val_grouped(pool_df, cfg, cfg["manifest_dir"])
    min_cross_hamming = verify_no_leakage(
        train_df, val_df, test_df, cfg["clean_root"], cfg["phash_dup_threshold"]
    )

    # 5. Datasets & Loaders
    _tmp = timm.create_model(cfg["model_name"], pretrained=False, num_classes=len(cfg["expected_classes"]))
    data_cfg = timm.data.resolve_model_data_config(_tmp)
    del _tmp

    img_size = data_cfg["input_size"][-2:]
    img_mean = list(data_cfg["mean"])
    img_std = list(data_cfg["std"])

    train_transforms = v2.Compose([
        v2.ToImage(),
        v2.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.8, 1.25), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.2),
        v2.RandomRotation(degrees=20),
        v2.RandomApply([v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03)], p=0.4),
        v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.2),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=img_mean, std=img_std),
        v2.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])
    eval_transforms = v2.Compose([
        v2.ToImage(),
        v2.Resize(int(round(img_size[0] / 0.875)), antialias=True),
        v2.CenterCrop(img_size),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=img_mean, std=img_std),
    ])

    train_loader = DataLoader(
        GarbageDataset(train_df, transform=train_transforms),
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        GarbageDataset(val_df, transform=eval_transforms),
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        GarbageDataset(test_df, transform=eval_transforms),
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    train_counts = train_df["label_idx"].value_counts().sort_index().to_numpy(dtype=np.float32)
    raw_weights = (train_counts.sum() / np.maximum(train_counts, 1.0)) ** 0.5
    class_weights = torch.tensor(raw_weights / raw_weights.mean(), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg["label_smoothing"])

    # 6. Two-Stage Training
    model = timm.create_model(
        cfg["model_name"],
        pretrained=True,
        num_classes=len(cfg["expected_classes"]),
        drop_path_rate=cfg["drop_path_rate"],
    ).to(device)

    use_bf16 = (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability(0)[0] >= 8
        and torch.cuda.is_bf16_supported()
    )
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"[AMP] use_bf16={use_bf16}, amp_dtype={amp_dtype}")
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available() and (not use_bf16))

    stage1_opt = torch.optim.AdamW(set_backbone_trainable(model, False)[1], lr=cfg["stage1_lr"], weight_decay=cfg["weight_decay"])
    backbone_p, head_p = set_backbone_trainable(model, True)
    stage2_opt = torch.optim.AdamW(
        [{"params": backbone_p, "lr": cfg["backbone_lr"]}, {"params": head_p, "lr": cfg["head_lr"]}],
        weight_decay=cfg["weight_decay"],
    )
    warmup_sch = torch.optim.lr_scheduler.LinearLR(stage2_opt, start_factor=0.1, total_iters=1)
    cosine_sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        stage2_opt, T_max=max(1, cfg["stage2_epochs"] - 1), eta_min=1e-6
    )
    stage2_sch = torch.optim.lr_scheduler.SequentialLR(stage2_opt, schedulers=[warmup_sch, cosine_sch], milestones=[1])

    best_ckpt_path = Path(cfg["ckpt_dir"]) / "best.pt"
    history_csv_path = Path(cfg["ckpt_dir"]) / "history.csv"

    total_epochs = cfg["stage1_epochs"] + cfg["stage2_epochs"]
    best_val_f1 = -1.0
    no_improve_epochs = 0
    history: list[dict] = []

    for epoch in range(total_epochs):
        if epoch < cfg["stage1_epochs"]:
            stage = 1
            set_backbone_trainable(model, False)
            active_opt, active_sch = stage1_opt, None
        else:
            stage = 2
            set_backbone_trainable(model, True)
            active_opt, active_sch = stage2_opt, stage2_sch

        train_loss = train_one_epoch(
            model, train_loader, active_opt, criterion, scaler, device, amp_dtype,
            grad_accum_steps=cfg["grad_accum_steps"], desc=f"Epoch {epoch + 1}/{total_epochs} [S{stage}]"
        )
        if active_sch is not None:
            active_sch.step()

        val_res = evaluate_loader(model, val_loader, criterion, device, amp_dtype, tta_hflip=False)
        history.append({
            "epoch": epoch + 1,
            "stage": stage,
            "train_loss": train_loss,
            "val_loss": val_res["loss"],
            "val_acc": val_res["acc"],
            "val_macro_f1": val_res["macro_f1"],
        })
        pd.DataFrame(history).to_csv(history_csv_path, index=False)

        is_best = val_res["macro_f1"] > best_val_f1
        if is_best:
            best_val_f1 = val_res["macro_f1"]
            no_improve_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_name": cfg["model_name"],
                    "classes": cfg["expected_classes"],
                    "model_state": model.state_dict(),
                    "best_val_f1": best_val_f1,
                    "img_size": img_size,
                    "mean": img_mean,
                    "std": img_std,
                },
                best_ckpt_path,
            )
        elif stage == 2:
            no_improve_epochs += 1

        mark = " *BEST*" if is_best else ""
        print(
            f"Epoch {epoch + 1:02d}/{total_epochs} (S{stage}) | "
            f"TrainLoss={train_loss:.4f} | ValLoss={val_res['loss']:.4f} | "
            f"ValAcc={val_res['acc']:.4f} | ValMacroF1={val_res['macro_f1']:.4f}{mark}",
            flush=True,
        )

        if stage == 2 and no_improve_epochs >= cfg["early_stop_patience"]:
            print(f"[Early Stop] Stopped at epoch {epoch + 1}.")
            break

    # Save training curves
    hist_df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(hist_df["epoch"], hist_df["train_loss"], label="Train Loss")
    axes[0].plot(hist_df["epoch"], hist_df["val_loss"], label="Val Loss")
    axes[0].axvline(cfg["stage1_epochs"] + 0.5, color="gray", linestyle="--")
    axes[0].set_title("Loss Curve")
    axes[0].legend()
    axes[1].plot(hist_df["epoch"], hist_df["val_acc"], label="Val Acc")
    axes[1].plot(hist_df["epoch"], hist_df["val_macro_f1"], label="Val Macro-F1")
    axes[1].axvline(cfg["stage1_epochs"] + 0.5, color="gray", linestyle="--")
    axes[1].set_title("Validation Metrics")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(Path(cfg["eval_dir"]) / "training_curves.png", dpi=150)
    plt.close(fig)

    # 7. Final evaluation on locked test set
    best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state"])
    model.eval()

    single_res = evaluate_loader(model, test_loader, criterion, device, amp_dtype, tta_hflip=False)
    tta_res = evaluate_loader(model, test_loader, criterion, device, amp_dtype, tta_hflip=True)
    cls_report_dict = classification_report(
        tta_res["y_true"], tta_res["y_pred"], target_names=cfg["expected_classes"], output_dict=True, digits=4
    )
    print(f"Test Single-View: Acc={single_res['acc']:.4f}, Macro-F1={single_res['macro_f1']:.4f}")
    print(f"Test 2-View TTA : Acc={tta_res['acc']:.4f}, Macro-F1={tta_res['macro_f1']:.4f}")
    print(classification_report(tta_res["y_true"], tta_res["y_pred"], target_names=cfg["expected_classes"], digits=4))

    cm = confusion_matrix(tta_res["y_true"], tta_res["y_pred"], normalize="true")
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=cfg["expected_classes"], yticklabels=cfg["expected_classes"], ax=ax,
    )
    ax.set_title("Locked Test Set Normalized Confusion Matrix (2-View TTA)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(Path(cfg["eval_dir"]) / "test_confusion_matrix.png", dpi=150)
    plt.close(fig)

    probs = tta_res["probs"]
    y_true = tta_res["y_true"]
    y_pred = tta_res["y_pred"]
    wrong_idx = np.where(y_true != y_pred)[0]
    if len(wrong_idx) > 0:
        conf_wrong = probs[wrong_idx, y_pred[wrong_idx]]
        top_wrong = wrong_idx[np.argsort(-conf_wrong)[:16]]
        cols = 4
        rows = math.ceil(len(top_wrong) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(13, 3.2 * rows))
        axes_flat = np.array(axes).reshape(-1)
        for k, ax in enumerate(axes_flat):
            ax.axis("off")
            if k < len(top_wrong):
                idx = int(top_wrong[k])
                row_meta = test_df.iloc[idx]
                ax.imshow(Image.open(row_meta["isolated_path"]).convert("RGB"))
                ax.set_title(
                    f"T: {cfg['expected_classes'][y_true[idx]]}\n"
                    f"P: {cfg['expected_classes'][y_pred[idx]]} ({probs[idx, y_pred[idx]]:.2f})",
                    fontsize=9, color="crimson",
                )
        plt.tight_layout()
        plt.savefig(Path(cfg["eval_dir"]) / "test_top_errors.png", dpi=140)
        plt.close(fig)

    # Grad-CAM
    target_layer = find_gradcam_target_layer(model)
    cam = GradCAM(model=model, target_layers=[target_layer])
    rng_cam = np.random.default_rng(cfg["seed"])
    sample_cam_idx = rng_cam.choice(len(test_df), size=min(8, len(test_df)), replace=False)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for ax, idx in zip(axes.flat, sample_cam_idx):
        row_meta = test_df.iloc[int(idx)]
        with Image.open(row_meta["isolated_path"]) as im:
            rgb = ImageOps.exif_transpose(im).convert("RGB")
        input_tensor = eval_transforms(rgb).unsqueeze(0).to(device)
        vis_np = np.asarray(rgb.resize(img_size[::-1]), dtype=np.float32) / 255.0
        pred_cls = int(y_pred[int(idx)])
        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_cls)])[0]
        overlay = show_cam_on_image(vis_np, grayscale_cam, use_rgb=True)
        ax.imshow(overlay)
        ax.set_title(f"T: {row_meta['label']} | P: {cfg['expected_classes'][pred_cls]}", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(Path(cfg["eval_dir"]) / "test_gradcam_samples.png", dpi=140)
    plt.close(fig)
    del cam

    # 8. Optional RealWaste Zero-Shot OOD Evaluation
    ood_summary: dict = {"status": "skipped"}
    try:
        rw_root = download_realwaste_root(Path("/content/realwaste_ood"))
        if rw_root is not None:
            rw_records = []
            class_to_idx = {c: i for i, c in enumerate(cfg["expected_classes"])}
            for rw_cls, v2_cls in REALWASTE_TO_V2.items():
                cls_dir = rw_root / rw_cls
                if not cls_dir.is_dir():
                    continue
                for p in cls_dir.rglob("*"):
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                        rw_records.append({
                            "isolated_path": str(p),
                            "label": v2_cls,
                            "label_idx": class_to_idx[v2_cls],
                        })
            rw_df = pd.DataFrame(rw_records)
            if len(rw_df) > 0:
                rw_loader = DataLoader(
                    GarbageDataset(rw_df, transform=eval_transforms),
                    batch_size=cfg["batch_size"],
                    shuffle=False,
                    num_workers=cfg["num_workers"],
                )
                rw_res = evaluate_loader(model, rw_loader, criterion, device, amp_dtype, tta_hflip=True)
                overlapping_labels = sorted(rw_df["label_idx"].unique().tolist())
                overlapping_names = [cfg["expected_classes"][i] for i in overlapping_labels]
                rw_f1 = float(f1_score(rw_res["y_true"], rw_res["y_pred"], labels=overlapping_labels, average="macro"))
                rw_report = classification_report(
                    rw_res["y_true"], rw_res["y_pred"], labels=overlapping_labels,
                    target_names=overlapping_names, output_dict=True, digits=4, zero_division=0,
                )
                print(f"RealWaste Zero-Shot OOD ({len(rw_df)} images): Acc={rw_res['acc']:.4f}, Macro-F1={rw_f1:.4f}")
                ood_summary = {
                    "status": "completed",
                    "num_images": int(len(rw_df)),
                    "acc": float(rw_res["acc"]),
                    "overlapping_macro_f1": rw_f1,
                    "per_class": rw_report,
                }
    except Exception as exc:
        print(f"[Warning] RealWaste OOD eval failed non-fatally: {exc}")
        ood_summary = {"status": "failed", "error": repr(exc)}

    summary = {
        "gpu": gpu_name,
        "model_name": cfg["model_name"],
        "counts": {
            "valid": int(len(valid_df)),
            "corrupted": int(len(corrupted_df)),
            "unique_clusters": int(valid_df["cluster_id"].nunique()),
            "test_locked": int(len(test_df)),
            "excluded_near_dups": int(len(excluded_df)),
            "train_val_pool": int(len(pool_df)),
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "min_cross_boundary_hamming": int(min_cross_hamming),
        },
        "test_lock_sha256": (Path(cfg["manifest_dir"]) / "test_lock_sha256.txt").read_text(encoding="utf-8").strip(),
        "training": {
            "epochs_completed": int(len(history)),
            "best_val_macro_f1": float(best_val_f1),
            "last_epoch": history[-1] if history else None,
        },
        "test_locked_metrics": {
            "single_view_acc": float(single_res["acc"]),
            "single_view_macro_f1": float(single_res["macro_f1"]),
            "tta_acc": float(tta_res["acc"]),
            "tta_macro_f1": float(tta_res["macro_f1"]),
            "classification_report": cls_report_dict,
        },
        "realwaste_ood": ood_summary,
    }
    summary_path = Path(cfg["eval_dir"]) / "step1_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Done] Saved summary to {summary_path}", flush=True)

    archive_path = Path("/content/step1_results.tar.gz")
    with tarfile.open(archive_path, "w:gz") as tf:
        for rel in ("manifests", "eval_outputs", "checkpoints/history.csv"):
            src = out_root / rel
            if src.exists():
                tf.add(src, arcname=rel)
    print(f"[ALL_DONE] Results archived to {archive_path}", flush=True)


if __name__ == "__main__":
    main()
