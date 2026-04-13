"""
ddpm_dataset.py
PanNuke 数据集读取（判别器标签纠正任务版）。

【变更说明】
─────────────────────────────────────────────────────────────────────────────
- 去掉 LR 字段：任务不再需要超分，推理时直接输入 HR
- 去掉 degradation 调用：不做任何图像退化或颜色增广
- 保留 GT label_map 和 nuc_mask：用于训练时的交集监督
- 新增 split_train_val()：从合并的多折数据中按 tissue type 分层抽样，
  保证验证集每种类别都有代表，抽出的样本从训练集中剔除
- NCTDataset 保留以兼容旧代码
"""

import os
import cv2
import random
import math
from typing import List, Optional, Sequence, Tuple, Dict
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


_VALID_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
_PANNUKE_REQUIRED_FILES = ("images.npy", "types.npy", "masks.npy")


# ─────────────────────────────────────────────────────────────────────────────
# 折目录查找工具
# ─────────────────────────────────────────────────────────────────────────────

def _find_pannuke_fold_files(folder: str) -> Optional[Tuple[str, str, str]]:
    if not folder or not os.path.isdir(folder):
        return None

    direct = tuple(os.path.join(folder, name) for name in _PANNUKE_REQUIRED_FILES)
    if all(os.path.isfile(p) for p in direct):
        return direct  # type: ignore[return-value]

    found = {name: [] for name in _PANNUKE_REQUIRED_FILES}
    for root, _, files in os.walk(folder):
        file_set = set(files)
        for name in _PANNUKE_REQUIRED_FILES:
            if name in file_set:
                found[name].append(os.path.join(root, name))

    if not all(found[name] for name in _PANNUKE_REQUIRED_FILES):
        return None

    def _score_triplet(img_p, type_p, mask_p):
        img_dir  = os.path.dirname(img_p)
        type_dir = os.path.dirname(type_p)
        mask_dir = os.path.dirname(mask_p)
        same_img_type    = 0 if img_dir == type_dir else 1
        has_images_token = 0 if "images" in img_dir.lower() else 1
        has_masks_token  = 0 if "masks"  in mask_dir.lower() else 1
        common_all       = os.path.commonpath([img_p, type_p, mask_p])
        common_img_type  = os.path.commonpath([img_p, type_p])
        return (same_img_type,
                has_images_token + has_masks_token,
                -len(common_all),
                -len(common_img_type))

    candidates = []
    for img_p in found["images.npy"]:
        for type_p in found["types.npy"]:
            for mask_p in found["masks.npy"]:
                candidates.append(
                    (_score_triplet(img_p, type_p, mask_p), (img_p, type_p, mask_p))
                )
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1] if candidates else None


def _is_pannuke_fold_dir(folder: str) -> bool:
    return _find_pannuke_fold_files(folder) is not None


def _normalize_fold_dirs(
    fold_dirs: Optional[Sequence[str]] = None,
    root_dir:  Optional[str] = None,
) -> List[str]:
    dirs: List[str] = []
    if fold_dirs:
        dirs.extend([str(d) for d in fold_dirs if d])
    if root_dir:
        if _is_pannuke_fold_dir(root_dir):
            dirs.append(root_dir)
        elif os.path.isdir(root_dir):
            for name in sorted(os.listdir(root_dir)):
                cand = os.path.join(root_dir, name)
                if _is_pannuke_fold_dir(cand):
                    dirs.append(cand)
    out, seen = [], set()
    for d in dirs:
        if d not in seen:
            out.append(d)
            seen.add(d)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# PanNuke masks → 语义空间
# ─────────────────────────────────────────────────────────────────────────────

def pannuke_mask_to_semantic(
    mask: np.ndarray,
    target_hw: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将 PanNuke masks.npy 单个样本转换为像素级标签图与核掩膜。

    返回：
      label_map : [H, W] int32   0=背景, 1-5=细胞类型
      nuc_mask  : [H, W] float32 有核像素为 1.0
    """
    m = np.asarray(mask)
    if m.ndim != 3:
        raise ValueError(f"PanNuke mask 期望 3 维，得到 shape={m.shape}")
    if m.shape[0] in (5, 6) and m.shape[-1] not in (5, 6):
        m = np.transpose(m, (1, 2, 0))

    h, w, c = m.shape
    if c not in (5, 6):
        raise ValueError(f"PanNuke mask 通道数应为 5 或 6，得到 shape={m.shape}")

    m = m.astype(np.int32, copy=False)
    label_map  = np.zeros((h, w), dtype=np.int32)
    nuc_binary = np.zeros((h, w), dtype=np.int32)

    for ch in range(5):
        cell_mask = m[..., ch] > 0
        label_map[cell_mask]  = ch + 1
        nuc_binary[cell_mask] = 1

    if target_hw is not None:
        th, tw = target_hw
        if (h, w) != (th, tw):
            label_map = cv2.resize(
                label_map.astype(np.float32), (tw, th),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.int32)
            nuc_binary = cv2.resize(
                nuc_binary.astype(np.float32), (tw, th),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.int32)

    return label_map, nuc_binary.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 分层抽样工具
# ─────────────────────────────────────────────────────────────────────────────

def split_train_val(
    dataset: 'PanNukeDataset',
    n_val:   int  = 100,
    seed:    int  = 42,
    verbose: bool = True,
) -> Tuple[Subset, Subset]:
    """
    从 PanNukeDataset 中按 tissue type 分层抽样 n_val 张作为验证集，
    其余作为训练集。验证集样本从训练集中剔除。

    分层策略：
      1. 统计每种 tissue type 的样本数
      2. 每种 type 按比例分配验证集数量，至少分配 1 张
      3. 若某类样本数不足，取该类全部样本
      4. 总数凑到 n_val（从样本最多的类中补充或裁剪）

    Args:
        dataset : PanNukeDataset 实例（包含全部折数据）
        n_val   : 验证集总样本数
        seed    : 随机种子，保证可复现
        verbose : 是否打印分层信息

    Returns:
        train_subset : Subset（训练集，已剔除验证集样本）
        val_subset   : Subset（验证集）
    """
    rng = random.Random(seed)
    n_total = len(dataset)

    # ── 1. 按 tissue type 分组，收集每组的全局索引 ──────────────────
    type_to_indices: Dict[str, List[int]] = defaultdict(list)
    for global_idx in range(n_total):
        fold_id, local_idx = dataset.index[global_idx]
        type_arr  = np.asarray(dataset.types[fold_id][local_idx])
        type_name = dataset._normalize_type_name(type_arr)
        type_to_indices[type_name].append(global_idx)

    all_types   = sorted(type_to_indices.keys())
    n_types     = len(all_types)
    n_val_real  = min(n_val, n_total)

    if verbose:
        print(f"\n分层抽样：共 {n_total} 张，目标验证集 {n_val_real} 张，"
              f"tissue type 数量 {n_types}")

    # ── 2. 按比例分配验证集数量，每类至少 1 张 ──────────────────────
    type_counts = {t: len(idxs) for t, idxs in type_to_indices.items()}
    total_count = sum(type_counts.values())

    # 初始分配（按比例，向下取整，至少 1）
    alloc: Dict[str, int] = {}
    for t in all_types:
        quota = max(1, int(math.floor(n_val_real * type_counts[t] / total_count)))
        alloc[t] = min(quota, type_counts[t])  # 不超过该类总数

    # ── 3. 调整总数到 n_val_real ─────────────────────────────────────
    current_total = sum(alloc.values())

    if current_total < n_val_real:
        # 不足则从样本最多的类中依次补充
        deficit = n_val_real - current_total
        sorted_by_remaining = sorted(
            all_types,
            key=lambda t: type_counts[t] - alloc[t],
            reverse=True,
        )
        for t in sorted_by_remaining:
            if deficit <= 0:
                break
            can_add = type_counts[t] - alloc[t]
            add     = min(can_add, deficit)
            alloc[t] += add
            deficit  -= add

    elif current_total > n_val_real:
        # 超出则从样本最多的类中依次裁剪（但每类至少保留 1）
        surplus = current_total - n_val_real
        sorted_by_alloc = sorted(
            all_types,
            key=lambda t: alloc[t],
            reverse=True,
        )
        for t in sorted_by_alloc:
            if surplus <= 0:
                break
            can_remove = alloc[t] - 1   # 至少保留 1
            remove     = min(can_remove, surplus)
            alloc[t]  -= remove
            surplus   -= remove

    # ── 4. 从每类中随机抽样 ─────────────────────────────────────────
    val_indices  = []
    for t in all_types:
        pool     = list(type_to_indices[t])
        rng.shuffle(pool)
        selected = pool[:alloc[t]]
        val_indices.extend(selected)
        if verbose:
            print(f"  {t:<20} total={type_counts[t]:>5}  "
                  f"val_alloc={alloc[t]:>3}  "
                  f"train_remain={type_counts[t]-alloc[t]:>5}")

    val_set  = set(val_indices)
    train_indices = [i for i in range(n_total) if i not in val_set]

    actual_val = len(val_indices)
    if verbose:
        print(f"\n实际验证集：{actual_val} 张  训练集：{len(train_indices)} 张")
        print(f"验证集随机种子：{seed}（固定种子保证可复现）")

    train_subset = Subset(dataset, train_indices)
    val_subset   = Subset(dataset, val_indices)
    return train_subset, val_subset


# ─────────────────────────────────────────────────────────────────────────────
# PanNukeDataset
# ─────────────────────────────────────────────────────────────────────────────

class PanNukeDataset(Dataset):
    """
    PanNuke 数据集，支持多折合并加载。

    返回字段：
      hr           : [3, H, W] float32 [0,1]
      label        : int   patch 级组织类型标签
      type_name    : str   patch 级组织类型名称
      gt_label_map : [H, W] int64   像素级细胞类型（0=bg, 1-5）
      gt_nuc_mask  : [H, W] float32 核掩膜

    注意：不再返回 lr 字段。
    """

    def __init__(
        self,
        fold_dirs:   Optional[Sequence[str]] = None,
        root_dir:    Optional[str] = None,
        target_size: Optional[int] = 256,
        max_samples: Optional[int] = None,
        verbose:     bool = True,
        # 以下参数保留接口兼容，不再使用
        scale:             int   = 4,
        blur_sigma_range:  tuple = (2.0, 3.0),
        noise_std_range:   tuple = (0.03, 0.08),
        stain_jitter:      float = 0.15,
    ):
        self.target_size = target_size

        self.fold_dirs = _normalize_fold_dirs(fold_dirs=fold_dirs, root_dir=root_dir)
        if not self.fold_dirs:
            raise ValueError(
                "No valid PanNuke fold dirs found. "
                "Expect folders containing images.npy / types.npy / masks.npy."
            )

        self.images: List[np.ndarray] = []
        self.types:  List[np.ndarray] = []
        self.masks:  List[np.ndarray] = []
        self.index:  List[Tuple[int, int]] = []

        for fold_id, fold_dir in enumerate(self.fold_dirs):
            fold_files = _find_pannuke_fold_files(fold_dir)
            if fold_files is None:
                raise ValueError(
                    f"Cannot locate images.npy/types.npy/masks.npy under: {fold_dir}"
                )

            images_path, types_path, masks_path = fold_files
            images = np.load(images_path, mmap_mode="r")
            types  = np.load(types_path,  mmap_mode="r")
            masks  = np.load(masks_path,  mmap_mode="r")

            if len(images) != len(types) or len(images) != len(masks):
                raise ValueError(
                    f"PanNuke fold size mismatch in {fold_dir}: "
                    f"images={len(images)}, types={len(types)}, masks={len(masks)}"
                )

            self.images.append(images)
            self.types.append(types)
            self.masks.append(masks)
            self.index.extend((fold_id, i) for i in range(len(images)))

            if verbose:
                print(
                    f"  Loaded PanNuke fold: {fold_dir} | samples={len(images)}\n"
                    f"    images: {images_path}\n"
                    f"    types : {types_path}\n"
                    f"    masks : {masks_path}"
                )

        self._build_type_label_mapping()

        if max_samples is not None:
            self.index = self.index[:max_samples]

        if verbose:
            print(f"PanNukeDataset loaded: folds={len(self.fold_dirs)}, "
                  f"total={len(self.index)}")
            print("  Tissue label mapping:")
            for lid, tname in sorted(self.label_to_type_name.items()):
                print(f"    {lid}: {tname}")

    def __len__(self):
        return len(self.index)

    # ── 静态工具 ─────────────────────────────────────────────────────

    @staticmethod
    def _to_hwc3_uint8(img: np.ndarray) -> np.ndarray:
        img = np.asarray(img)
        if img.ndim != 3:
            raise ValueError(f"Expected 3-dim image, got shape={img.shape}")
        if img.shape[-1] == 3:
            out = img
        elif img.shape[0] == 3:
            out = np.transpose(img, (1, 2, 0))
        else:
            raise ValueError(f"Cannot infer channel axis for shape={img.shape}")
        if out.dtype != np.uint8:
            if np.issubdtype(out.dtype, np.floating):
                vmax = float(out.max()) if out.size > 0 else 1.0
                out  = (out * 255.0).clip(0, 255).astype(np.uint8) if vmax <= 1.0 \
                       else out.clip(0, 255).astype(np.uint8)
            else:
                out = out.clip(0, 255).astype(np.uint8)
        return out

    @staticmethod
    def _normalize_type_name(type_value) -> str:
        t = np.squeeze(np.asarray(type_value))
        if t.ndim == 0:
            value = t.item()
        else:
            flat = t.reshape(-1)
            if flat.size == 0:
                value = "Unknown"
            elif flat.size == 1:
                value = flat[0].item() if hasattr(flat[0], "item") else flat[0]
            else:
                value = (f"class_{int(np.argmax(flat))}"
                         if np.issubdtype(flat.dtype, np.number)
                         else (flat[0].item() if hasattr(flat[0], "item") else flat[0]))
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="ignore")
        return str(value).strip() or "Unknown"

    def _build_type_label_mapping(self) -> None:
        type_names = []
        for fold_types in self.types:
            for i in range(len(fold_types)):
                type_names.append(self._normalize_type_name(fold_types[i]))
        unique_names = sorted(set(type_names))
        self.type_name_to_label = {n: i for i, n in enumerate(unique_names)}
        self.label_to_type_name = {i: n for n, i in self.type_name_to_label.items()}

    def _to_patch_label(self, type_arr: np.ndarray) -> Tuple[int, str]:
        type_name = self._normalize_type_name(type_arr)
        if type_name not in self.type_name_to_label:
            new_id = len(self.type_name_to_label)
            self.type_name_to_label[type_name] = new_id
            self.label_to_type_name[new_id] = type_name
        return self.type_name_to_label[type_name], type_name

    # ── __getitem__ ──────────────────────────────────────────────────

    def __getitem__(self, idx):
        fold_id, local_idx = self.index[idx]

        img      = self._to_hwc3_uint8(self.images[fold_id][local_idx])
        type_arr = np.asarray(self.types[fold_id][local_idx])

        if self.target_size is not None:
            h, w = img.shape[:2]
            if h != self.target_size or w != self.target_size:
                img = cv2.resize(
                    img, (self.target_size, self.target_size),
                    interpolation=cv2.INTER_LANCZOS4,
                )

        hr = torch.from_numpy(img).float() / 255.0
        hr = hr.permute(2, 0, 1).contiguous()   # [3, H, W]

        label_id, type_name = self._to_patch_label(type_arr)

        raw_mask  = np.asarray(self.masks[fold_id][local_idx])
        target_hw = (self.target_size, self.target_size) \
                    if self.target_size is not None else None
        gt_label_map, gt_nuc_mask = pannuke_mask_to_semantic(
            raw_mask, target_hw=target_hw
        )

        return {
            "hr":           hr,
            "label":        torch.tensor(label_id, dtype=torch.long),
            "type_name":    type_name,
            "gt_label_map": torch.from_numpy(gt_label_map).long(),
            "gt_nuc_mask":  torch.from_numpy(gt_nuc_mask).float(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# NCTDataset（旧版，保留兼容）
# ─────────────────────────────────────────────────────────────────────────────

class NCTDataset(Dataset):
    """旧版 NCT-CRC-HE 数据集，保留兼容。"""

    def __init__(
        self,
        tum_dir,
        norm_dir,
        oversample:  bool          = True,
        target_size: Optional[int] = 256,
        scale:             int   = 2,
        blur_sigma_range:  tuple = (0.5, 1.5),
        noise_std_range:   tuple = (0.0, 0.02),
        stain_jitter:      float = 0.05,
    ):
        self.target_size = target_size
        self.files  = []
        self.labels = []

        tum_files = []
        if tum_dir and os.path.exists(tum_dir):
            tum_files = [
                os.path.join(tum_dir, f)
                for f in os.listdir(tum_dir)
                if f.lower().endswith(_VALID_EXT)
            ]

        norm_files = []
        if norm_dir and os.path.exists(norm_dir):
            norm_files = [
                os.path.join(norm_dir, f)
                for f in os.listdir(norm_dir)
                if f.lower().endswith(_VALID_EXT)
            ]

        if not tum_files and not norm_files:
            raise ValueError("No image files found. Check tum_dir and norm_dir.")

        if oversample and tum_files and norm_files:
            ratio = len(norm_files) / max(len(tum_files), 1)
            if ratio > 1.5:
                tum_files = tum_files * int(round(ratio))
            elif ratio < 0.67:
                norm_files = norm_files * int(round(1.0 / ratio))

        pairs = list(zip(tum_files + norm_files,
                         [1] * len(tum_files) + [0] * len(norm_files)))
        random.shuffle(pairs)
        self.files, self.labels = zip(*pairs) if pairs else ([], [])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path  = self.files[idx]
        label = self.labels[idx]
        img   = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.target_size is not None:
            img = cv2.resize(img, (self.target_size, self.target_size),
                             interpolation=cv2.INTER_LANCZOS4)
        hr = torch.from_numpy(img).float() / 255.0
        hr = hr.permute(2, 0, 1).contiguous()
        return {"hr": hr, "label": torch.tensor(label, dtype=torch.long)}


# ─────────────────────────────────────────────────────────────────────────────
# 工厂函数
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(
    dataset_type: str,
    *,
    tum_dir:       Optional[str] = None,
    norm_dir:      Optional[str] = None,
    oversample:    bool = True,
    pannuke_root:  Optional[str] = None,
    pannuke_folds: Optional[Sequence[str]] = None,
    target_size:   Optional[int] = 256,
    max_samples:   Optional[int] = None,
    # 保留接口兼容，不再使用
    scale:             int   = 4,
    blur_sigma_range:  tuple = (2.0, 3.0),
    noise_std_range:   tuple = (0.03, 0.08),
    stain_jitter:      float = 0.15,
) -> Dataset:
    dataset_type = dataset_type.lower()

    if dataset_type == "pannuke":
        return PanNukeDataset(
            fold_dirs   = pannuke_folds,
            root_dir    = pannuke_root,
            target_size = target_size,
            max_samples = max_samples,
        )

    if dataset_type == "nct":
        return NCTDataset(
            tum_dir     = tum_dir,
            norm_dir    = norm_dir,
            oversample  = oversample,
            target_size = target_size,
        )

    raise ValueError(f"Unsupported dataset_type: {dataset_type}")