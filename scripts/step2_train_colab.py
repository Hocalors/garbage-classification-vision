"""Headless Google Colab execution script for Garbage Classification v2 + RealWaste (Step 2).

Preserves Step 1's exact locked v2 test set, creates an independently locked high-quality
RealWaste test set, enforces a cross-dataset pHash leakage firewall, trains with
domain-balanced WeightedRandomSampler, and runs an exact Step 1 vs Step 2 comparison.
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
        ("sklearn", "scikit-learn"),
        ("cv2", "opencv-python-headless"),
        ("seaborn", "seaborn"),
        ("PIL", "Pillow"),
        ("tqdm", "tqdm"),
    ]
    missing = [pkg for mod, pkg in required if importlib.util.find_spec(mod) is None]
    if missing:
        print(f"[Setup] Installing missing packages: {missing}", flush=True)
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
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


def compute_image_record(path_str: str, label: str, domain: str, raw_label: str = "") -> dict:
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
                raise ValueError(f"Invalid dimensions: {width}x{height}")

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
            "domain": domain,
            "raw_label": raw_label or label,
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
            "domain": domain,
            "raw_label": raw_label or label,
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


def scan_mapped_dataset(
    dataset_root: str | Path,
    folder_to_label: dict[str, str],
    domain: str,
    max_workers: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(dataset_root)
    tasks: list[tuple[str, str, str]] = []
    for folder_name, target_label in folder_to_label.items():
        cls_dir = root / folder_name
        if not cls_dir.is_dir():
            print(f"[Warning] Missing folder {cls_dir} in {domain}")
            continue
        files = sorted(
            p for p in cls_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        for fpath in files:
            tasks.append((str(fpath), target_label, folder_name))

    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(compute_image_record, p, lbl, domain, raw_lbl)
            for p, lbl, raw_lbl in tasks
        ]
        for fut in tqdm(futures, total=len(futures), desc=f"Scanning {domain}"):
            records.append(fut.result())

    full_df = pd.DataFrame(records)
    valid_df = full_df[full_df["status"] == "ok"].reset_index(drop=True)
    corrupted_df = full_df[full_df["status"] != "ok"].reset_index(drop=True)
    return valid_df, corrupted_df


def _hex_to_uint64(hex_values) -> np.ndarray:
    return np.array([int(h, 16) for h in hex_values], dtype=np.uint64)


def min_hamming_between(query_hex_values, ref_hex_values, chunk_size: int = 512) -> np.ndarray:
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
    phash_hex_list, threshold: int = 6, chunk_size: int = 512
) -> tuple[np.ndarray, np.ndarray]:
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


def compute_quality_metrics(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
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


def compute_lock_sha256(test_df: pd.DataFrame) -> str:
    payload = "\n".join(sorted(f"{lbl}:{md5}" for lbl, md5 in zip(test_df["label"], test_df["md5"])))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def materialize_isolated_split(df: pd.DataFrame, target_split_dir: Path) -> pd.DataFrame:
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


def exclude_against_test_set(
    df: pd.DataFrame,
    selected_indices: list[int],
    phash_dup_threshold: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test_raw = df.loc[selected_indices].copy()
    locked_clusters = set(test_raw["cluster_id"].astype(int).tolist())

    not_test_mask = ~df.index.isin(test_raw.index)
    same_cluster_mask = not_test_mask & df["cluster_id"].isin(locked_clusters)
    candidate_mask = not_test_mask & (~same_cluster_mask)

    boundary_dup_mask = pd.Series(False, index=df.index)
    if candidate_mask.any() and len(test_raw) > 0:
        cand_indices = df.index[candidate_mask].to_numpy()
        min_dists = min_hamming_between(
            df.loc[cand_indices, "phash_hex"].tolist(),
            test_raw["phash_hex"].tolist(),
            chunk_size=512,
        )
        hit_indices = cand_indices[min_dists <= phash_dup_threshold]
        boundary_dup_mask.loc[hit_indices] = True

    excluded_mask = same_cluster_mask | boundary_dup_mask
    excluded_raw = df.loc[excluded_mask].copy()
    excluded_raw["exclusion_reason"] = np.where(
        same_cluster_mask.loc[excluded_mask],
        "same_cluster_as_locked_test",
        "cross_boundary_phash_near_dup",
    )
    pool_raw = df.loc[not_test_mask & (~excluded_mask)].copy()
    return test_raw, pool_raw, excluded_raw


def select_high_quality_test_indices(
    work: pd.DataFrame,
    classes: list[str],
    test_ratio: float,
    min_per_class: int,
    seed: int,
) -> list[int]:
    selected_indices: list[int] = []
    used_clusters: set[int] = set()
    for cls_idx, cls_name in enumerate(classes):
        cls_df = work[work["label"] == cls_name]
        if len(cls_df) == 0:
            continue
        n_clusters = int(cls_df["cluster_id"].nunique())
        cap_n = max(1, int(n_clusters * 0.30))
        target_n = min(max(min_per_class, int(round(n_clusters * test_ratio))), cap_n)
        rng = np.random.default_rng(seed + cls_idx)
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
    return selected_indices


def download_realwaste_root(target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        rw_path = Path(kagglehub.dataset_download("joarvidsson/realwaste-image-classification"))
        return locate_class_root(rw_path, list(REALWASTE_TO_V2.keys()))
    except Exception as exc:
        print(f"[Info] KaggleHub RealWaste failed ({exc}), trying UCI ZIP...")

    zip_path = target_dir / "realwaste.zip"
    extract_dir = target_dir / "extracted"
    if not extract_dir.exists():
        urllib.request.urlretrieve("https://archive.ics.uci.edu/static/public/908/realwaste.zip", zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
    return locate_class_root(extract_dir, list(REALWASTE_TO_V2.keys()))


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


def make_eval_transforms(img_size: tuple[int, int], img_mean: list[float], img_std: list[float]):
    return v2.Compose([
        v2.ToImage(),
        v2.Resize(int(round(img_size[0] / 0.875)), antialias=True),
        v2.CenterCrop(img_size),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=img_mean, std=img_std),
    ])


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


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device,
    amp_dtype,
    tta_hflip: bool = False,
    macro_labels: list[int] | None = None,
) -> dict:
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
    if macro_labels is not None:
        macro_f1 = float(f1_score(y_true, y_pred, labels=macro_labels, average="macro", zero_division=0))
    else:
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return {
        "loss": total_loss / max(total_samples, 1),
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": macro_f1,
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


def build_domain_balanced_weights(train_df: pd.DataFrame, domain_shares: dict[str, float]) -> np.ndarray:
    """Compute per-sample weights so domains follow domain_shares and classes are sqrt-balanced."""
    weights = np.zeros(len(train_df), dtype=np.float64)
    for dom, target_share in domain_shares.items():
        dom_mask = (train_df["domain"] == dom).to_numpy()
        if not dom_mask.any():
            continue
        cls_counts = train_df.loc[dom_mask, "label"].value_counts().to_dict()
        raw_dom_w = np.array(
            [(1.0 / max(cls_counts[lbl], 1)) ** 0.5 for lbl in train_df.loc[dom_mask, "label"]],
            dtype=np.float64,
        )
        raw_dom_w = raw_dom_w / raw_dom_w.sum() * target_share
        weights[dom_mask] = raw_dom_w
    weights = weights / weights.sum()
    return weights


def verify_step2_no_leakage(
    v2_train: pd.DataFrame,
    v2_val: pd.DataFrame,
    v2_test: pd.DataFrame,
    rw_train: pd.DataFrame,
    rw_val: pd.DataFrame,
    rw_test: pd.DataFrame,
    phash_dup_threshold: int = 6,
) -> int:
    for dom_name, tr, va, te in (
        ("v2", v2_train, v2_val, v2_test),
        ("realwaste", rw_train, rw_val, rw_test),
    ):
        splits = [("train", tr), ("val", va), ("test", te)]
        for i in range(len(splits)):
            for j in range(i + 1, len(splits)):
                na, da = splits[i]
                nb, db = splits[j]
                for col in ("orig_path", "isolated_path", "md5", "cluster_id"):
                    ov = set(da[col]) & set(db[col])
                    assert len(ov) == 0, f"[{dom_name}] Leakage between {na} and {nb} on {col}: {len(ov)}"

    all_dev = pd.concat([v2_train, v2_val, rw_train, rw_val], ignore_index=True)
    all_test = pd.concat([v2_test, rw_test], ignore_index=True)

    for col in ("orig_path", "isolated_path", "md5"):
        ov = set(all_dev[col]) & set(all_test[col])
        assert len(ov) == 0, f"Cross-dataset leakage between dev and locked test on {col}: {len(ov)}"

    min_dists = min_hamming_between(all_dev["phash_hex"].tolist(), all_test["phash_hex"].tolist(), chunk_size=512)
    min_overall = int(min_dists.min()) if len(min_dists) > 0 else 64
    assert min_overall > phash_dup_threshold, (
        f"Cross-dataset pHash boundary violation: {min_overall} <= {phash_dup_threshold}"
    )
    print(
        f"[Step 2 Firewall Passed] Dev={len(all_dev)} (v2={len(v2_train)+len(v2_val)}, rw={len(rw_train)+len(rw_val)}) | "
        f"LockedTest={len(all_test)} (v2={len(v2_test)}, rw={len(rw_test)}) | min_hamming={min_overall}",
        flush=True,
    )
    return min_overall


def main() -> None:
    out_root = Path("/content/step2_outputs")
    step1_ref = Path("/content/step1_ref")
    cfg = {
        "v2_slug": "sumn2u/garbage-classification-v2",
        "expected_classes": [
            "battery", "biological", "cardboard", "clothes", "glass",
            "metal", "paper", "plastic", "shoes", "trash",
        ],
        "model_name": "convnext_tiny.fb_in22k_ft_in1k_384",
        "seed": 42,
        "phash_dup_threshold": 6,
        "rw_test_ratio": 0.20,
        "rw_min_test_per_class": 20,
        "min_test_res": 224,
        "min_aspect": 0.4,
        "max_aspect": 2.5,
        "min_laplacian_var": 30.0,
        "min_brightness": 25.0,
        "max_brightness": 235.0,
        "min_contrast": 15.0,
        "val_folds": 6,
        "domain_shares": {"v2": 0.60, "realwaste": 0.40},
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
        "clean_root": "/content/step2_clean",
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
        f"model={cfg['model_name']} | batch_size={cfg['batch_size']}, grad_accum={cfg['grad_accum_steps']}",
        flush=True,
    )

    use_bf16 = (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability(0)[0] >= 8
        and torch.cuda.is_bf16_supported()
    )
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available() and (not use_bf16))
    print(f"[AMP] use_bf16={use_bf16}, amp_dtype={amp_dtype}", flush=True)

    # 1. Download & scan both datasets
    v2_dl = kagglehub.dataset_download(cfg["v2_slug"])
    v2_root = locate_class_root(v2_dl, cfg["expected_classes"])
    rw_root = download_realwaste_root(Path("/content/realwaste_raw"))
    print(f"v2 root: {v2_root} | RealWaste root: {rw_root}", flush=True)

    v2_valid_df, v2_corrupt_df = scan_mapped_dataset(
        v2_root, {c: c for c in cfg["expected_classes"]}, domain="v2", max_workers=8
    )
    rw_valid_df, rw_corrupt_df = scan_mapped_dataset(
        rw_root, REALWASTE_TO_V2, domain="realwaste", max_workers=8
    )

    v2_cids, v2_csizes = cluster_phash_vectorized(
        v2_valid_df["phash_hex"].tolist(), threshold=cfg["phash_dup_threshold"], chunk_size=512
    )
    v2_valid_df["cluster_id"] = v2_cids
    v2_valid_df["cluster_size"] = v2_csizes

    rw_cids, rw_csizes = cluster_phash_vectorized(
        rw_valid_df["phash_hex"].tolist(), threshold=cfg["phash_dup_threshold"], chunk_size=512
    )
    rw_valid_df["cluster_id"] = rw_cids
    rw_valid_df["cluster_size"] = rw_csizes

    # 2. Reuse exact Step 1 v2_test_locked
    ref_test_csv = step1_ref / "test_locked.csv"
    ref_sha_txt = step1_ref / "test_lock_sha256.txt"
    assert ref_test_csv.exists() and ref_sha_txt.exists(), "Missing Step 1 test lock files in /content/step1_ref/"

    prev_test = pd.read_csv(ref_test_csv)
    expected_sha = ref_sha_txt.read_text(encoding="utf-8").strip()
    prev_pairs = set(zip(prev_test["label"].astype(str), prev_test["md5"].astype(str)))
    matched_v2_indices = []
    used_pairs = set()
    for idx, row in v2_valid_df.iterrows():
        pair = (str(row["label"]), str(row["md5"]))
        if pair in prev_pairs and pair not in used_pairs:
            matched_v2_indices.append(idx)
            used_pairs.add(pair)

    assert len(matched_v2_indices) == len(prev_pairs), (
        f"Matched {len(matched_v2_indices)}/{len(prev_pairs)} Step 1 locked test images!"
    )
    v2_work = compute_quality_metrics(v2_valid_df, cfg)
    recomputed_sha = compute_lock_sha256(v2_work.loc[matched_v2_indices])
    assert recomputed_sha == expected_sha, f"Step 1 test_lock_sha256 mismatch: {recomputed_sha} != {expected_sha}"
    print(f"[Step 1 Lock Reused] Exact {len(matched_v2_indices)} v2 test images verified (SHA256={recomputed_sha[:16]}...)", flush=True)

    v2_test_raw, v2_pool_raw, v2_ex_raw = exclude_against_test_set(
        v2_work, matched_v2_indices, cfg["phash_dup_threshold"]
    )

    # 3. Independently select & lock RealWaste high-quality test set
    rw_work = compute_quality_metrics(rw_valid_df, cfg)
    rw_classes = sorted(rw_work["label"].unique().tolist())
    rw_selected = select_high_quality_test_indices(
        rw_work,
        classes=rw_classes,
        test_ratio=cfg["rw_test_ratio"],
        min_per_class=cfg["rw_min_test_per_class"],
        seed=cfg["seed"] + 100,
    )
    rw_test_raw, rw_pool_raw, rw_ex_raw = exclude_against_test_set(
        rw_work, rw_selected, cfg["phash_dup_threshold"]
    )
    print(
        f"RealWaste scan={len(rw_valid_df)} | rw_test_locked={len(rw_test_raw)} | "
        f"rw_excluded_near_dups={len(rw_ex_raw)} | rw_pool_pre_cross={len(rw_pool_raw)}",
        flush=True,
    )

    # 4. Cross-dataset leakage removal against COMBINED locked test sets
    all_test_raw = pd.concat([v2_test_raw, rw_test_raw], ignore_index=True)
    locked_md5s = set(all_test_raw["md5"].tolist())
    locked_phashes = all_test_raw["phash_hex"].tolist()

    cross_excluded_frames = []
    for dom_name, pool_df_ref in (("v2", v2_pool_raw), ("realwaste", rw_pool_raw)):
        md5_hit = pool_df_ref["md5"].isin(locked_md5s).to_numpy()
        cross_dists = min_hamming_between(pool_df_ref["phash_hex"].tolist(), locked_phashes, chunk_size=512)
        phash_hit = cross_dists <= cfg["phash_dup_threshold"]
        bad_mask = md5_hit | phash_hit
        if bad_mask.any():
            ex_df = pool_df_ref.iloc[bad_mask].copy()
            ex_df["exclusion_reason"] = "cross_dataset_match_to_locked_test"
            cross_excluded_frames.append(ex_df)
        if dom_name == "v2":
            v2_pool_clean = pool_df_ref.iloc[~bad_mask].copy()
        else:
            rw_pool_clean = pool_df_ref.iloc[~bad_mask].copy()

    cross_excluded_df = (
        pd.concat(cross_excluded_frames, ignore_index=True)
        if cross_excluded_frames
        else pd.DataFrame(columns=v2_pool_raw.columns)
    )
    print(f"Cross-dataset boundary exclusions against combined locked tests: {len(cross_excluded_df)}", flush=True)

    # 5. Physical isolation & label encoding
    clean_root = Path(cfg["clean_root"])
    manifest_dir = Path(cfg["manifest_dir"])
    class_to_idx = {c: i for i, c in enumerate(cfg["expected_classes"])}

    v2_test_df = materialize_isolated_split(v2_test_raw, clean_root / "v2_test_locked")
    rw_test_df = materialize_isolated_split(rw_test_raw, clean_root / "rw_test_locked")
    v2_pool_df = materialize_isolated_split(v2_pool_clean, clean_root / "v2_pool")
    rw_pool_df = materialize_isolated_split(rw_pool_clean, clean_root / "rw_pool")

    for frame in (v2_test_df, rw_test_df, v2_pool_df, rw_pool_df):
        frame["label_idx"] = frame["label"].map(class_to_idx)

    # 6. Per-domain StratifiedGroupKFold Train/Val Split
    sgkf = StratifiedGroupKFold(n_splits=cfg["val_folds"], shuffle=True, random_state=cfg["seed"])
    v2_tr_i, v2_va_i = next(sgkf.split(v2_pool_df, v2_pool_df["label_idx"].to_numpy(), groups=v2_pool_df["cluster_id"].to_numpy()))
    rw_tr_i, rw_va_i = next(sgkf.split(rw_pool_df, rw_pool_df["label_idx"].to_numpy(), groups=rw_pool_df["cluster_id"].to_numpy()))

    v2_train_df = v2_pool_df.iloc[v2_tr_i].copy().reset_index(drop=True)
    v2_val_df = v2_pool_df.iloc[v2_va_i].copy().reset_index(drop=True)
    rw_train_df = rw_pool_df.iloc[rw_tr_i].copy().reset_index(drop=True)
    rw_val_df = rw_pool_df.iloc[rw_va_i].copy().reset_index(drop=True)

    min_cross_hamming = verify_step2_no_leakage(
        v2_train_df, v2_val_df, v2_test_df,
        rw_train_df, rw_val_df, rw_test_df,
        phash_dup_threshold=cfg["phash_dup_threshold"],
    )

    train_df = pd.concat([v2_train_df, rw_train_df], ignore_index=True)

    # Save manifests
    v2_test_df.to_csv(manifest_dir / "v2_test_locked.csv", index=False)
    rw_test_df.to_csv(manifest_dir / "rw_test_locked.csv", index=False)
    train_df.to_csv(manifest_dir / "joint_train.csv", index=False)
    v2_val_df.to_csv(manifest_dir / "v2_val.csv", index=False)
    rw_val_df.to_csv(manifest_dir / "rw_val.csv", index=False)
    cross_excluded_df.to_csv(manifest_dir / "excluded_cross_dataset_dups.csv", index=False)
    (manifest_dir / "v2_test_lock_sha256.txt").write_text(recomputed_sha + "\n", encoding="utf-8")
    (manifest_dir / "rw_test_lock_sha256.txt").write_text(compute_lock_sha256(rw_test_df) + "\n", encoding="utf-8")
    with open(manifest_dir / "classes.json", "w", encoding="utf-8") as f:
        json.dump({"classes": cfg["expected_classes"], "class_to_idx": class_to_idx}, f, ensure_ascii=False, indent=2)

    # 7. Transforms, Domain-Balanced Sampler & Loaders
    _tmp = timm.create_model(cfg["model_name"], pretrained=False, num_classes=len(cfg["expected_classes"]))
    data_cfg = timm.data.resolve_model_data_config(_tmp)
    del _tmp

    img_size = data_cfg["input_size"][-2:]
    img_mean = list(data_cfg["mean"])
    img_std = list(data_cfg["std"])

    train_transforms = v2.Compose([
        v2.ToImage(),
        v2.RandomResizedCrop(img_size, scale=(0.55, 1.0), ratio=(0.8, 1.25), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.25),
        v2.RandomRotation(degrees=25),
        v2.RandomApply([v2.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.04)], p=0.5),
        v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.8))], p=0.25),
        v2.RandomGrayscale(p=0.05),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=img_mean, std=img_std),
        v2.RandomErasing(p=0.30, scale=(0.02, 0.18)),
    ])
    eval_transforms = make_eval_transforms(img_size, img_mean, img_std)

    sample_weights = build_domain_balanced_weights(train_df, cfg["domain_shares"])
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(train_df),
        replacement=True,
    )

    train_loader = DataLoader(
        GarbageDataset(train_df, transform=train_transforms),
        batch_size=cfg["batch_size"],
        sampler=sampler,
        num_workers=cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    v2_val_loader = DataLoader(
        GarbageDataset(v2_val_df, transform=eval_transforms),
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    rw_val_loader = DataLoader(
        GarbageDataset(rw_val_df, transform=eval_transforms),
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    v2_test_loader = DataLoader(
        GarbageDataset(v2_test_df, transform=eval_transforms),
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    rw_test_loader = DataLoader(
        GarbageDataset(rw_test_df, transform=eval_transforms),
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    rw_overlapping_labels = sorted(rw_test_df["label_idx"].unique().tolist())
    rw_overlapping_names = [cfg["expected_classes"][i] for i in rw_overlapping_labels]
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])

    # 8. Train Step 2 Model
    model = timm.create_model(
        cfg["model_name"],
        pretrained=True,
        num_classes=len(cfg["expected_classes"]),
        drop_path_rate=cfg["drop_path_rate"],
    ).to(device)

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

    best_ckpt_path = Path(cfg["ckpt_dir"]) / "best_step2.pt"
    history_csv_path = Path(cfg["ckpt_dir"]) / "history.csv"

    total_epochs = cfg["stage1_epochs"] + cfg["stage2_epochs"]
    best_joint_f1 = -1.0
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
            grad_accum_steps=cfg["grad_accum_steps"], desc=f"Epoch {epoch + 1}/{total_epochs} [S{stage}]",
        )
        if active_sch is not None:
            active_sch.step()

        v2_val_res = evaluate_loader(model, v2_val_loader, criterion, device, amp_dtype, tta_hflip=False)
        rw_val_res = evaluate_loader(
            model, rw_val_loader, criterion, device, amp_dtype, tta_hflip=False, macro_labels=rw_overlapping_labels
        )
        joint_f1 = 0.5 * v2_val_res["macro_f1"] + 0.5 * rw_val_res["macro_f1"]

        history.append({
            "epoch": epoch + 1,
            "stage": stage,
            "train_loss": train_loss,
            "v2_val_acc": v2_val_res["acc"],
            "v2_val_f1": v2_val_res["macro_f1"],
            "rw_val_acc": rw_val_res["acc"],
            "rw_val_f1": rw_val_res["macro_f1"],
            "joint_val_f1": joint_f1,
        })
        pd.DataFrame(history).to_csv(history_csv_path, index=False)

        is_best = joint_f1 > best_joint_f1
        if is_best:
            best_joint_f1 = joint_f1
            no_improve_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_name": cfg["model_name"],
                    "classes": cfg["expected_classes"],
                    "model_state": model.state_dict(),
                    "best_joint_f1": best_joint_f1,
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
            f"Epoch {epoch + 1:02d}/{total_epochs} (S{stage}) | Loss={train_loss:.4f} | "
            f"v2ValF1={v2_val_res['macro_f1']:.4f} | rwValF1={rw_val_res['macro_f1']:.4f} | "
            f"JointF1={joint_f1:.4f}{mark}",
            flush=True,
        )

        if stage == 2 and no_improve_epochs >= cfg["early_stop_patience"]:
            print(f"[Early Stop] Stopped at epoch {epoch + 1}.", flush=True)
            break

    # Save training curves
    hist_df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(hist_df["epoch"], hist_df["train_loss"], label="Train Loss")
    axes[0].axvline(cfg["stage1_epochs"] + 0.5, color="gray", linestyle="--")
    axes[0].set_title("Step 2 Joint Training Loss")
    axes[0].legend()
    axes[1].plot(hist_df["epoch"], hist_df["v2_val_f1"], label="v2 Val Macro-F1")
    axes[1].plot(hist_df["epoch"], hist_df["rw_val_f1"], label="RealWaste Val Macro-F1")
    axes[1].plot(hist_df["epoch"], hist_df["joint_val_f1"], label="Joint Val F1", linewidth=2, color="black")
    axes[1].axvline(cfg["stage1_epochs"] + 0.5, color="gray", linestyle="--")
    axes[1].set_title("Step 2 Validation Macro-F1")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(Path(cfg["eval_dir"]) / "step2_training_curves.png", dpi=150)
    plt.close(fig)

    # 9. Evaluate Step 2 best checkpoint on both locked test sets
    s2_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(s2_ckpt["model_state"])
    model.eval()

    s2_v2_single = evaluate_loader(model, v2_test_loader, criterion, device, amp_dtype, tta_hflip=False)
    s2_v2_tta = evaluate_loader(model, v2_test_loader, criterion, device, amp_dtype, tta_hflip=True)
    s2_rw_single = evaluate_loader(
        model, rw_test_loader, criterion, device, amp_dtype, tta_hflip=False, macro_labels=rw_overlapping_labels
    )
    s2_rw_tta = evaluate_loader(
        model, rw_test_loader, criterion, device, amp_dtype, tta_hflip=True, macro_labels=rw_overlapping_labels
    )

    s2_v2_report = classification_report(
        s2_v2_tta["y_true"], s2_v2_tta["y_pred"], target_names=cfg["expected_classes"], output_dict=True, digits=4
    )
    s2_rw_report = classification_report(
        s2_rw_tta["y_true"], s2_rw_tta["y_pred"], labels=rw_overlapping_labels,
        target_names=rw_overlapping_names, output_dict=True, digits=4, zero_division=0,
    )

    # 10. Exact side-by-side Step 1 baseline evaluation on BOTH locked test sets (or recorded Step 1 summary fallback)
    s1_ckpt_path = step1_ref / "best.pt"
    if s1_ckpt_path.exists():
        s1_ckpt = torch.load(s1_ckpt_path, map_location=device, weights_only=False)
        s1_model = timm.create_model(
            s1_ckpt["model_name"], pretrained=False, num_classes=len(cfg["expected_classes"])
        ).to(device)
        s1_model.load_state_dict(s1_ckpt["model_state"])
        s1_model.eval()

        s1_eval_tf = make_eval_transforms(tuple(s1_ckpt["img_size"]), list(s1_ckpt["mean"]), list(s1_ckpt["std"]))
        s1_v2_loader = DataLoader(GarbageDataset(v2_test_df, transform=s1_eval_tf), batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
        s1_rw_loader = DataLoader(GarbageDataset(rw_test_df, transform=s1_eval_tf), batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

        s1_v2_tta = evaluate_loader(s1_model, s1_v2_loader, criterion, device, amp_dtype, tta_hflip=True)
        s1_rw_tta = evaluate_loader(
            s1_model, s1_rw_loader, criterion, device, amp_dtype, tta_hflip=True, macro_labels=rw_overlapping_labels
        )
        s1_rw_report = classification_report(
            s1_rw_tta["y_true"], s1_rw_tta["y_pred"], labels=rw_overlapping_labels,
            target_names=rw_overlapping_names, output_dict=True, digits=4, zero_division=0,
        )
        s1_source = "live_eval_on_step2_locked_tests"
        del s1_model
    else:
        print("[Info] /content/step1_ref/best.pt not uploaded; using recorded Step 1 metrics from step1_summary.json.", flush=True)
        s1_v2_tta = {"acc": 0.9746835443037974, "macro_f1": 0.9697216878173676}
        s1_rw_tta = {"acc": 0.6033249158249159, "macro_f1": 0.6092134195415699}
        s1_rw_report = {}
        s1_source = "recorded_step1_summary_full_realwaste_ood"

    print(
        f"\n=== Step 1 vs Step 2 (2-View TTA) ===\n"
        f"v2_test_locked ({len(v2_test_df)}):  Step1 Acc={s1_v2_tta['acc']:.4f} F1={s1_v2_tta['macro_f1']:.4f} | "
        f"Step2 Acc={s2_v2_tta['acc']:.4f} F1={s2_v2_tta['macro_f1']:.4f}\n"
        f"rw_test_locked ({len(rw_test_df)}):  Step1 Acc={s1_rw_tta['acc']:.4f} F1={s1_rw_tta['macro_f1']:.4f} | "
        f"Step2 Acc={s2_rw_tta['acc']:.4f} F1={s2_rw_tta['macro_f1']:.4f}",
        flush=True,
    )

    # Save Confusion Matrices, Top Errors & Comparison Bar Chart
    for name, res_obj, cls_labels, cls_names in (
        ("v2_test_confusion_matrix.png", s2_v2_tta, list(range(len(cfg["expected_classes"]))), cfg["expected_classes"]),
        ("rw_test_confusion_matrix.png", s2_rw_tta, rw_overlapping_labels, rw_overlapping_names),
    ):
        cm = confusion_matrix(res_obj["y_true"], res_obj["y_pred"], labels=cls_labels, normalize="true")
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=cls_names, yticklabels=cls_names, ax=ax)
        ax.set_title(f"Step 2 Normalized Confusion Matrix ({name.split('_')[0]})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        plt.tight_layout()
        plt.savefig(Path(cfg["eval_dir"]) / name, dpi=150)
        plt.close(fig)

    probs = s2_rw_tta["probs"]
    y_true = s2_rw_tta["y_true"]
    y_pred = s2_rw_tta["y_pred"]
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
                row_meta = rw_test_df.iloc[idx]
                ax.imshow(Image.open(row_meta["isolated_path"]).convert("RGB"))
                ax.set_title(
                    f"T: {cfg['expected_classes'][y_true[idx]]}\n"
                    f"P: {cfg['expected_classes'][y_pred[idx]]} ({probs[idx, y_pred[idx]]:.2f})",
                    fontsize=9, color="crimson",
                )
        plt.tight_layout()
        plt.savefig(Path(cfg["eval_dir"]) / "rw_test_top_errors.png", dpi=140)
        plt.close(fig)

    # Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(8, 4.5))
    metrics_labels = ["v2 Test Acc", "v2 Test Macro-F1", "RealWaste Test Acc", "RealWaste Test Macro-F1"]
    s1_vals = [s1_v2_tta["acc"], s1_v2_tta["macro_f1"], s1_rw_tta["acc"], s1_rw_tta["macro_f1"]]
    s2_vals = [s2_v2_tta["acc"], s2_v2_tta["macro_f1"], s2_rw_tta["acc"], s2_rw_tta["macro_f1"]]
    x = np.arange(len(metrics_labels))
    width = 0.35
    b1 = ax.bar(x - width / 2, s1_vals, width, label="Step 1 (v2 only)")
    b2 = ax.bar(x + width / 2, s2_vals, width, label="Step 2 (v2 + RealWaste balanced)")
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_labels)
    ax.legend()
    ax.set_title("Step 1 vs Step 2 Locked Test Comparison (2-View TTA)")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", (bar.get_x() + bar.get_width() / 2, h), ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(Path(cfg["eval_dir"]) / "step2_comparison_bar.png", dpi=150)
    plt.close(fig)

    summary = {
        "gpu": gpu_name,
        "model_name": cfg["model_name"],
        "counts": {
            "v2_valid": int(len(v2_valid_df)),
            "v2_corrupted": int(len(v2_corrupt_df)),
            "rw_valid": int(len(rw_valid_df)),
            "rw_corrupted": int(len(rw_corrupt_df)),
            "v2_test_locked": int(len(v2_test_df)),
            "rw_test_locked": int(len(rw_test_df)),
            "v2_excluded_near_dups": int(len(v2_ex_raw)),
            "rw_excluded_near_dups": int(len(rw_ex_raw)),
            "cross_dataset_excluded": int(len(cross_excluded_df)),
            "v2_train": int(len(v2_train_df)),
            "v2_val": int(len(v2_val_df)),
            "rw_train": int(len(rw_train_df)),
            "rw_val": int(len(rw_val_df)),
            "joint_train_total": int(len(train_df)),
            "min_cross_boundary_hamming": int(min_cross_hamming),
        },
        "v2_test_lock_sha256": recomputed_sha,
        "rw_test_lock_sha256": compute_lock_sha256(rw_test_df),
        "training": {
            "epochs_completed": int(len(history)),
            "best_joint_val_f1": float(best_joint_f1),
            "last_epoch": history[-1] if history else None,
        },
        "step1_baseline_on_locked_tests": {
            "source": s1_source,
            "v2_tta_acc": float(s1_v2_tta["acc"]),
            "v2_tta_macro_f1": float(s1_v2_tta["macro_f1"]),
            "rw_tta_acc": float(s1_rw_tta["acc"]),
            "rw_tta_macro_f1": float(s1_rw_tta["macro_f1"]),
            "rw_per_class": s1_rw_report,
        },
        "step2_on_locked_tests": {
            "v2_single_acc": float(s2_v2_single["acc"]),
            "v2_single_macro_f1": float(s2_v2_single["macro_f1"]),
            "v2_tta_acc": float(s2_v2_tta["acc"]),
            "v2_tta_macro_f1": float(s2_v2_tta["macro_f1"]),
            "v2_per_class": s2_v2_report,
            "rw_single_acc": float(s2_rw_single["acc"]),
            "rw_single_macro_f1": float(s2_rw_single["macro_f1"]),
            "rw_tta_acc": float(s2_rw_tta["acc"]),
            "rw_tta_macro_f1": float(s2_rw_tta["macro_f1"]),
            "rw_per_class": s2_rw_report,
        },
    }
    summary_path = Path(cfg["eval_dir"]) / "step2_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Done] Saved summary to {summary_path}", flush=True)

    archive_path = Path("/content/step2_results.tar.gz")
    with tarfile.open(archive_path, "w:gz") as tf:
        for rel in ("manifests", "eval_outputs", "checkpoints/history.csv"):
            src = out_root / rel
            if src.exists():
                tf.add(src, arcname=rel)
    print(f"[ALL_DONE] Step 2 results archived to {archive_path}", flush=True)


if __name__ == "__main__":
    main()
