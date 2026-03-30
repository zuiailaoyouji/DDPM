import os
import cv2
import math
import random
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from degradation import degrade


_VALID_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
_PANNUKE_REQUIRED_FILES = ("images.npy", "types.npy", "masks.npy")


def _find_pannuke_fold_files(folder: str) -> Optional[Tuple[str, str, str]]:
    """
    在给定目录下递归查找 PanNuke fold 对应的 npy 文件。

    兼容两类结构：
    1) 规整结构：
       Fold 1/
         images.npy
         types.npy
         masks.npy
    2) 你现在的真实结构：
       Fold 1/images/fold1/images.npy
       Fold 1/images/fold1/types.npy
       Fold 1/masks/fold1/masks.npy

    返回:
        (images_path, types_path, masks_path)
    若找不到则返回 None。
    """
    if not folder or not os.path.isdir(folder):
        return None

    # 先查当前目录直下
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

    def _score_triplet(img_p: str, type_p: str, mask_p: str) -> Tuple[int, int, int, int]:
        # 分数越小越好
        img_dir = os.path.dirname(img_p)
        type_dir = os.path.dirname(type_p)
        mask_dir = os.path.dirname(mask_p)

        same_img_type = 0 if img_dir == type_dir else 1
        has_images_token = 0 if "images" in img_dir.lower() else 1
        has_masks_token = 0 if "masks" in mask_dir.lower() else 1

        common_img_type = os.path.commonpath([img_p, type_p])
        common_all = os.path.commonpath([img_p, type_p, mask_p])

        # 越接近同一 fold 子目录越好，因此用负长度作为优先项
        return (
            same_img_type,
            has_images_token + has_masks_token,
            -len(common_all),
            -len(common_img_type),
        )

    candidates = []
    for img_p in found["images.npy"]:
        for type_p in found["types.npy"]:
            for mask_p in found["masks.npy"]:
                candidates.append((_score_triplet(img_p, type_p, mask_p), (img_p, type_p, mask_p)))

    candidates.sort(key=lambda x: x[0])
    return candidates[0][1] if candidates else None


def _is_pannuke_fold_dir(folder: str) -> bool:
    return _find_pannuke_fold_files(folder) is not None


def _normalize_fold_dirs(
    fold_dirs: Optional[Sequence[str]] = None,
    root_dir: Optional[str] = None,
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

    out: List[str] = []
    seen = set()
    for d in dirs:
        if d not in seen:
            out.append(d)
            seen.add(d)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# PanNuke masks.npy → HoVer-Net 语义空间 转换工具
# ─────────────────────────────────────────────────────────────────────────────

def pannuke_mask_to_semantic(
    mask: np.ndarray,
    target_hw: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将 PanNuke masks.npy 单个样本转换为 HoVer-Net 语义空间的标签图与核掩膜。

    PanNuke masks 格式：
        shape (H, W, 6) 或 (H, W, 5) 或转置版 (5/6, H, W)
        ch 0: Neoplastic    → HoVer-Net tp class 1
        ch 1: Inflammatory  → HoVer-Net tp class 2
        ch 2: Connective    → HoVer-Net tp class 3
        ch 3: Dead          → HoVer-Net tp class 4
        ch 4: Epithelial    → HoVer-Net tp class 5
        ch 5: 背景通道（若存在），忽略

    返回：
        label_map : np.ndarray [H, W] int32
                    0=背景, 1=Neoplastic, 2=Inflammatory,
                    3=Connective, 4=Dead, 5=Epithelial
                    （与 HoVer-Net tp 分支的 6 类索引一致）
        nuc_mask  : np.ndarray [H, W] float32
                    任意细胞核通道有实例（>0）的像素为 1.0，其余为 0.0
    """
    m = np.asarray(mask)
    if m.ndim != 3:
        raise ValueError(f"PanNuke mask 期望 3 维，得到 shape={m.shape}")

    # 兼容 (C, H, W) 或 (H, W, C)
    if m.shape[0] in (5, 6) and m.shape[-1] not in (5, 6):
        m = np.transpose(m, (1, 2, 0))   # → (H, W, C)

    h, w, c = m.shape
    if c not in (5, 6):
        raise ValueError(f"PanNuke mask 通道数应为 5 或 6，得到 shape={m.shape}")

    m = m.astype(np.int32, copy=False)

    # label_map：逐通道写入 HoVer-Net 类别索引（后写的会覆盖前写的，保持"最后类型优先"；
    # 若像素同时属于多个类型的实例则不常见，但以防万一用 argmax 更严谨——此处保留简单赋值）
    label_map = np.zeros((h, w), dtype=np.int32)
    nuc_binary = np.zeros((h, w), dtype=np.int32)
    n_type_ch = 5   # ch 0..4 对应 5 种细胞类型
    for ch in range(n_type_ch):
        cell_mask = m[..., ch] > 0          # 该通道有实例的像素
        label_map[cell_mask] = ch + 1       # HoVer-Net class = pannuke_ch + 1
        nuc_binary[cell_mask] = 1

    if target_hw is not None:
        th, tw = target_hw
        if (h, w) != (th, tw):
            label_map = cv2.resize(
                label_map.astype(np.float32),
                (tw, th),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.int32)
            nuc_binary = cv2.resize(
                nuc_binary.astype(np.float32),
                (tw, th),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.int32)

    nuc_mask = nuc_binary.astype(np.float32)
    return label_map, nuc_mask



class NCTDataset(Dataset):
    """
    旧版 NCT-CRC-HE 数据集读取逻辑。
    先保留，方便你后续做 legacy 备份或结果对照。
    """

    def __init__(
        self,
        tum_dir,
        norm_dir,
        oversample: bool = True,
        scale: int = 2,
        blur_sigma_range: tuple = (0.5, 1.5),
        noise_std_range: tuple = (0.0, 0.02),
        stain_jitter: float = 0.05,
        target_size: Optional[int] = 256,
    ):
        self.scale = scale
        self.blur_sigma_range = blur_sigma_range
        self.noise_std_range = noise_std_range
        self.stain_jitter = stain_jitter
        self.target_size = target_size
        self.files = []

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

        orig_tum = len(tum_files)
        orig_norm = len(norm_files)

        if oversample and tum_files and norm_files:
            if len(norm_files) < len(tum_files):
                reps = math.ceil(len(tum_files) / len(norm_files))
                norm_files = (norm_files * reps)[: len(tum_files)]
                random.shuffle(norm_files)
                print(f"  Oversample: NORM {orig_norm} -> {len(norm_files)}")
            elif len(tum_files) < len(norm_files):
                reps = math.ceil(len(norm_files) / len(tum_files))
                tum_files = (tum_files * reps)[: len(norm_files)]
                random.shuffle(tum_files)
                print(f"  Oversample: TUM {orig_tum} -> {len(tum_files)}")

        self.files.extend([(f, 1) for f in tum_files])
        self.files.extend([(f, 0) for f in norm_files])
        random.shuffle(self.files)

        n_tum = sum(1 for _, l in self.files if l == 1)
        n_norm = sum(1 for _, l in self.files if l == 0)
        print(f"NCTDataset loaded: TUM={n_tum}, NORM={n_norm}, total={len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path, label = self.files[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.target_size is not None:
            img = cv2.resize(
                img,
                (self.target_size, self.target_size),
                interpolation=cv2.INTER_LANCZOS4,
            )

        hr = torch.from_numpy(img).float() / 255.0
        hr = hr.permute(2, 0, 1)
        lr = degrade(
            hr,
            scale=self.scale,
            blur_sigma_range=self.blur_sigma_range,
            noise_std_range=self.noise_std_range,
            stain_jitter_strength=self.stain_jitter,
        )
        return {"hr": hr, "lr": lr, "label": torch.tensor(label, dtype=torch.long)}


class PanNukeDataset(Dataset):
    """
    PanNuke 数据集读取逻辑。

    兼容：
    1) 规整结构
       Fold 1/images.npy
       Fold 1/types.npy
       Fold 1/masks.npy

    2) 当前真实结构
       Fold 1/images/fold1/images.npy
       Fold 1/images/fold1/types.npy
       Fold 1/masks/fold1/masks.npy

    新增返回字段（相比旧版）
    ─────────────────────────
    'gt_label_map' : LongTensor [H, W]
        像素级 GT 细胞类型标签，与 HoVer-Net tp 分支空间一致：
            0 = 背景
            1 = Neoplastic
            2 = Inflammatory
            3 = Connective
            4 = Dead
            5 = Epithelial

    'gt_nuc_mask'  : FloatTensor [H, W]
        GT 细胞核二值掩膜（任意类型细胞核所在像素=1.0）

    原有返回字段
    ─────────────
    'hr'        : FloatTensor [3, H, W]
    'lr'        : FloatTensor [3, H, W]
    'label'     : LongTensor []   — patch 级组织类型整数标签（来自 types.npy）
    'type_name' : str             — patch 级组织类型名称
    """

    def __init__(
        self,
        fold_dirs: Optional[Sequence[str]] = None,
        root_dir: Optional[str] = None,
        scale: int = 2,
        blur_sigma_range: Tuple[float, float] = (0.5, 1.5),
        noise_std_range: Tuple[float, float] = (0.0, 0.02),
        stain_jitter: float = 0.05,
        target_size: Optional[int] = 256,
        max_samples: Optional[int] = None,
        verbose: bool = True,
    ):
        self.scale = scale
        self.blur_sigma_range = blur_sigma_range
        self.noise_std_range = noise_std_range
        self.stain_jitter = stain_jitter
        self.target_size = target_size

        self.fold_dirs = _normalize_fold_dirs(fold_dirs=fold_dirs, root_dir=root_dir)
        if not self.fold_dirs:
            raise ValueError(
                "No valid PanNuke fold dirs found. Expect fold folders that contain "
                "images.npy / types.npy / masks.npy, either directly or in nested subfolders."
            )

        self.images: List[np.ndarray] = []
        self.types: List[np.ndarray] = []
        self.masks: List[np.ndarray] = []
        self.index: List[Tuple[int, int]] = []
        self.fold_file_map: List[Tuple[str, str, str]] = []

        for fold_id, fold_dir in enumerate(self.fold_dirs):
            fold_files = _find_pannuke_fold_files(fold_dir)
            if fold_files is None:
                raise ValueError(f"Cannot locate images.npy/types.npy/masks.npy under fold dir: {fold_dir}")

            images_path, types_path, masks_path = fold_files
            images = np.load(images_path, mmap_mode="r")
            types = np.load(types_path, mmap_mode="r")
            masks = np.load(masks_path, mmap_mode="r")

            if len(images) != len(types) or len(images) != len(masks):
                raise ValueError(
                    f"PanNuke fold size mismatch in {fold_dir}: "
                    f"images={len(images)}, types={len(types)}, masks={len(masks)}"
                )

            self.images.append(images)
            self.types.append(types)
            self.masks.append(masks)
            self.fold_file_map.append((images_path, types_path, masks_path))
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
            print(f"PanNukeDataset loaded: folds={len(self.fold_dirs)}, total={len(self.index)}")
            print("  Tissue label mapping:")
            for label_id, type_name in sorted(self.label_to_type_name.items()):
                print(f"    {label_id}: {type_name}")

    def __len__(self):
        return len(self.index)

    @staticmethod
    def _to_hwc3_uint8(img: np.ndarray) -> np.ndarray:
        img = np.asarray(img)
        if img.ndim != 3:
            raise ValueError(f"Expected image with 3 dims, got shape={img.shape}")

        if img.shape[-1] == 3:
            out = img
        elif img.shape[0] == 3:
            out = np.transpose(img, (1, 2, 0))
        else:
            raise ValueError(f"Cannot infer channel axis for image with shape={img.shape}")

        if out.dtype != np.uint8:
            if np.issubdtype(out.dtype, np.floating):
                vmax = float(out.max()) if out.size > 0 else 1.0
                if vmax <= 1.0:
                    out = (out * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    out = out.clip(0, 255).astype(np.uint8)
            else:
                out = out.clip(0, 255).astype(np.uint8)

        return out

    @staticmethod
    def _normalize_type_name(type_value) -> str:
        t = np.asarray(type_value)
        t = np.squeeze(t)

        if t.ndim == 0:
            value = t.item()
        else:
            flat = t.reshape(-1)
            if flat.size == 0:
                value = "Unknown"
            elif flat.size == 1:
                value = flat[0].item() if hasattr(flat[0], "item") else flat[0]
            else:
                # 若不是字符串标签而是 one-hot / soft label，则退化到 argmax 的字符串表达。
                if np.issubdtype(flat.dtype, np.number):
                    value = f"class_{int(np.argmax(flat))}"
                else:
                    value = flat[0].item() if hasattr(flat[0], "item") else flat[0]

        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="ignore")

        value = str(value).strip()
        return value if value else "Unknown"

    def _build_type_label_mapping(self) -> None:
        type_names = []
        for fold_types in self.types:
            names = []
            for i in range(len(fold_types)):
                names.append(self._normalize_type_name(fold_types[i]))
            type_names.extend(names)

        unique_names = sorted(set(type_names))
        self.type_name_to_label = {name: idx for idx, name in enumerate(unique_names)}
        self.label_to_type_name = {idx: name for name, idx in self.type_name_to_label.items()}

    def _to_patch_label(self, type_arr: np.ndarray) -> Tuple[int, str]:
        type_name = self._normalize_type_name(type_arr)
        if type_name not in self.type_name_to_label:
            new_id = len(self.type_name_to_label)
            self.type_name_to_label[type_name] = new_id
            self.label_to_type_name[new_id] = type_name
        return self.type_name_to_label[type_name], type_name

    def __getitem__(self, idx):
        fold_id, local_idx = self.index[idx]

        # ── 图像 ──────────────────────────────────────────────────────────
        img = self._to_hwc3_uint8(self.images[fold_id][local_idx])
        type_arr = np.asarray(self.types[fold_id][local_idx])

        if self.target_size is not None:
            h, w = img.shape[:2]
            if h != self.target_size or w != self.target_size:
                img = cv2.resize(
                    img,
                    (self.target_size, self.target_size),
                    interpolation=cv2.INTER_LANCZOS4,
                )

        hr = torch.from_numpy(img).float() / 255.0
        hr = hr.permute(2, 0, 1).contiguous()

        lr = degrade(
            hr,
            scale=self.scale,
            blur_sigma_range=self.blur_sigma_range,
            noise_std_range=self.noise_std_range,
            stain_jitter_strength=self.stain_jitter,
        )

        # ── Patch 级标签 ───────────────────────────────────────────────────
        label_id, type_name = self._to_patch_label(type_arr)

        # ── 像素级 GT mask → HoVer-Net 语义空间 ────────────────────────────
        raw_mask = np.asarray(self.masks[fold_id][local_idx])
        target_hw = (self.target_size, self.target_size) if self.target_size is not None else None
        gt_label_map, gt_nuc_mask = pannuke_mask_to_semantic(raw_mask, target_hw=target_hw)

        return {
            "hr": hr,
            "lr": lr,
            "label": torch.tensor(label_id, dtype=torch.long),
            "type_name": type_name,
            "gt_label_map": torch.from_numpy(gt_label_map).long(),   # [H, W]
            "gt_nuc_mask":  torch.from_numpy(gt_nuc_mask).float(),   # [H, W]
        }


def build_dataset(
    dataset_type: str,
    *,
    tum_dir: Optional[str] = None,
    norm_dir: Optional[str] = None,
    oversample: bool = True,
    pannuke_root: Optional[str] = None,
    pannuke_folds: Optional[Sequence[str]] = None,
    scale: int = 2,
    blur_sigma_range: Tuple[float, float] = (0.5, 1.5),
    noise_std_range: Tuple[float, float] = (0.0, 0.02),
    stain_jitter: float = 0.05,
    target_size: Optional[int] = 256,
    max_samples: Optional[int] = None,
) -> Dataset:
    dataset_type = dataset_type.lower()

    if dataset_type == "nct":
        return NCTDataset(
            tum_dir=tum_dir,
            norm_dir=norm_dir,
            oversample=oversample,
            scale=scale,
            blur_sigma_range=blur_sigma_range,
            noise_std_range=noise_std_range,
            stain_jitter=stain_jitter,
            target_size=target_size,
        )

    if dataset_type == "pannuke":
        return PanNukeDataset(
            fold_dirs=pannuke_folds,
            root_dir=pannuke_root,
            scale=scale,
            blur_sigma_range=blur_sigma_range,
            noise_std_range=noise_std_range,
            stain_jitter=stain_jitter,
            target_size=target_size,
            max_samples=max_samples,
        )

    raise ValueError(f"Unsupported dataset_type: {dataset_type}")