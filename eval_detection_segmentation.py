"""
eval_detection_segmentation.py
细胞核检测与分割任务评估脚本。

三组对比：
  HR 基线    : HR 直接送 CellViT
  消融模型   : 无语义监督（use_semantic=False）
  本文方法   : 交集 Focal-CE + CellViT 软标签 sem_tensor

【后处理】
  使用 CellViT 官方 model.calculate_instance_map() 方法，
  从 nuclei_binary_map + hv_map + nuclei_type_map 生成实例分割图。

【评估指标 · 检测】
  Precision / Recall / F1 / PQ (DQ × SQ)
  匹配策略：IoU ≥ 0.5 的匈牙利匹配

【评估指标 · 分割】
  AJI (Aggregated Jaccard Index)
  DICE (匹配实例平均)

【额外输出】
  按 tissue type 和 cell type 分组

推理策略：
  N_RUNS 次不同随机噪声单步推理，CellViT 各通道输出取平均后做后处理。
"""

import numpy as np
import torch
import torch.nn.functional as F
import sys
import os
import random
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import OrderedDict, defaultdict
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, '/home/xuwen/DDPM/CellViT')
sys.path.insert(0, '/home/xuwen/DDPM')

from diffusers import DDPMScheduler

from ddpm_dataset import PanNukeDataset
from unet_wrapper import create_model
from ddpm_utils import load_cellvit, predict_x0_from_noise_shared
from semantic_sr_loss import run_cellvit, build_sem_tensor_from_cellvit

device = 'cuda'
CLASS_NAMES = ['Background', 'Neoplastic', 'Inflammatory',
               'Connective', 'Dead', 'Epithelial']


def _improvement_ratio(hr_val, full_val, upper=1.0):
    """(Full − HR) / (上限 − HR)。上限默认 1.0,适用于 F1/PQ/AJI/DICE。"""
    gap = upper - hr_val
    if gap <= 1e-8 or hr_val != hr_val:  # NaN
        return float('nan')
    return (full_val - hr_val) / gap

def _fmt_pct(r):
    return f"{r*100:>+6.1f}%" if not (isinstance(r, float) and r != r) else f"{'N/A':>7}"


# ═══════════════════════════════════════════════════════════════════════════
# CellViT 完整推理 + 官方后处理
# ═══════════════════════════════════════════════════════════════════════════

def run_cellvit_full(cellvit, img_01):
    """CellViT 完整前向，返回原始输出字典（含 hv_map）。"""
    with torch.no_grad():
        return cellvit(img_01)


def cellvit_output_to_instance_map(cellvit_model, raw_output, magnification=40):
    """
    调用 CellViT 官方 calculate_instance_map。

    Returns:
        inst_map  : (H, W) int32
        inst_type : dict {inst_id: cell_type (int)}
    """
    predictions = OrderedDict()
    predictions['nuclei_binary_map'] = raw_output['nuclei_binary_map']
    predictions['nuclei_type_map'] = raw_output['nuclei_type_map']
    predictions['hv_map'] = raw_output['hv_map']

    instance_map, cell_dict_list = cellvit_model.calculate_instance_map(
        predictions, magnification=magnification
    )

    inst_map = instance_map.squeeze(0).cpu().numpy().astype(np.int32)

    # 解析 cell_dict_list → inst_type
    inst_type = {}
    if cell_dict_list and len(cell_dict_list) > 0:
        cells = cell_dict_list[0]
        if isinstance(cells, dict):
            for iid, info in cells.items():
                if isinstance(info, dict) and 'type' in info:
                    inst_type[int(iid)] = int(info['type'])
        elif isinstance(cells, list):
            for c in cells:
                if isinstance(c, dict):
                    iid = c.get('id', c.get('inst_id', 0))
                    if iid > 0:
                        inst_type[int(iid)] = int(c.get('type', 0))

    # fallback: 从 type_map 推断
    if not inst_type:
        tp = F.softmax(raw_output['nuclei_type_map'], dim=1)
        tl = tp.argmax(dim=1).squeeze(0).cpu().numpy()
        for iid in np.unique(inst_map):
            if iid == 0: continue
            r = tl[inst_map == iid]
            nz = r[r > 0]
            if len(nz) > 0:
                vals, cnts = np.unique(nz, return_counts=True)
                inst_type[int(iid)] = int(vals[cnts.argmax()])

    return inst_map, inst_type


# ═══════════════════════════════════════════════════════════════════════════
# GT 实例分割
# ═══════════════════════════════════════════════════════════════════════════

def build_gt_instance_map(mask_raw, target_hw=None):
    import cv2
    m = np.asarray(mask_raw)
    if m.ndim != 3: raise ValueError(f"mask shape error: {m.shape}")
    if m.shape[0] in (5,6) and m.shape[-1] not in (5,6):
        m = np.transpose(m, (1,2,0))
    H, W, C = m.shape; C = min(C, 5)
    m = m[..., :C].astype(np.int32)

    inst_map = np.zeros((H,W), dtype=np.int32)
    inst_type = {}; gid = 0
    for ch in range(C):
        for lid in np.unique(m[...,ch]):
            if lid == 0: continue
            gid += 1
            inst_map[m[...,ch] == lid] = gid
            inst_type[gid] = ch + 1
    if target_hw:
        th, tw = target_hw
        if (H,W) != (th,tw):
            inst_map = cv2.resize(inst_map.astype(np.float32), (tw,th),
                                  interpolation=cv2.INTER_NEAREST).astype(np.int32)
    return inst_map, inst_type


# ═══════════════════════════════════════════════════════════════════════════
# 指标
# ═══════════════════════════════════════════════════════════════════════════

def match_instances(gt_map, pred_map, iou_thresh=0.5):
    gt_ids = [x for x in np.unique(gt_map) if x>0]
    pred_ids = [x for x in np.unique(pred_map) if x>0]
    if not gt_ids and not pred_ids: return [],[],[]
    if not gt_ids: return [], [], list(pred_ids)
    if not pred_ids: return [], list(gt_ids), []
    iou_mat = np.zeros((len(gt_ids), len(pred_ids)))
    for gi, gid in enumerate(gt_ids):
        gm = (gt_map==gid)
        for pi, pid in enumerate(pred_ids):
            pm = (pred_map==pid)
            i = np.logical_and(gm,pm).sum(); u = np.logical_or(gm,pm).sum()
            iou_mat[gi,pi] = i/(u+1e-8)
    row, col = linear_sum_assignment(-iou_mat)
    matched, mg, mp = [], set(), set()
    for r,c in zip(row,col):
        if iou_mat[r,c] >= iou_thresh:
            matched.append((gt_ids[r], pred_ids[c], float(iou_mat[r,c])))
            mg.add(gt_ids[r]); mp.add(pred_ids[c])
    return matched, [g for g in gt_ids if g not in mg], [p for p in pred_ids if p not in mp]

def compute_det(gt_map, pred_map, iou_thresh=0.5):
    matched, fn, fp = match_instances(gt_map, pred_map, iou_thresh)
    TP,FP,FN = len(matched), len(fp), len(fn)
    p = TP/(TP+FP) if TP+FP>0 else 0.; r = TP/(TP+FN) if TP+FN>0 else 0.
    f1 = 2*p*r/(p+r) if p+r>0 else 0.
    DQ = TP/(TP+.5*FP+.5*FN) if TP+FP+FN>0 else 0.
    SQ = (sum(x for _,_,x in matched)/TP) if TP>0 else 0.
    return dict(precision=p,recall=r,f1=f1,PQ=DQ*SQ,DQ=DQ,SQ=SQ,TP=TP,FP=FP,FN=FN)

def compute_aji(gt_map, pred_map):
    gids = [x for x in np.unique(gt_map) if x>0]
    pids = [x for x in np.unique(pred_map) if x>0]
    if not gids and not pids: return 1.
    if not gids: return 0.
    ti,tu,used = 0.,0.,set()
    for g in gids:
        gm=(gt_map==g); bi,bp,bii,bu=0,None,0,0
        for p in pids:
            if p in used: continue
            pm=(pred_map==p); ii=np.logical_and(gm,pm).sum()
            if ii==0: continue
            uu=np.logical_or(gm,pm).sum()
            if ii/(uu+1e-8)>bi: bi,bp,bii,bu=ii/(uu+1e-8),p,ii,uu
        if bp: ti+=bii; tu+=bu; used.add(bp)
        else: tu+=gm.sum()
    for p in pids:
        if p not in used: tu+=(pred_map==p).sum()
    return ti/(tu+1e-8)

def compute_dice_m(gt_map, pred_map, th=0.5):
    matched,_,_ = match_instances(gt_map, pred_map, th)
    if not matched: return 0.
    return float(np.mean([2*np.logical_and(gt_map==g,pred_map==p).sum()
                          /(np.sum(gt_map==g)+np.sum(pred_map==p)+1e-8)
                          for g,p,_ in matched]))

def compute_det_per_class(gt_map, gt_itype, pred_map, pred_itype,
                          n_classes=6, iou_thresh=0.5):
    """
    做法 B:全局匹配 → 按 GT 类型拆 TP/FP/FN(分类错误不双倍计费)。

    - TP_c:匹配对中 GT 类型为 c 的数量(无论 pred 类型是否对)
    - FN_c:未匹配的 GT 中类型为 c 的数量
    - FP_c:未匹配的 pred 中类型为 c 的数量
    - SQ_c:类 c 的匹配对的 IoU 平均
    - DQ_c = TP_c / (TP_c + 0.5*FP_c + 0.5*FN_c)
    - PQ_c = DQ_c * SQ_c

    返回 {ct: dict(precision/recall/f1/PQ/DQ/SQ/TP/FP/FN)} 仅含出现的类。
    """
    matched, fn_ids, fp_ids = match_instances(gt_map, pred_map, iou_thresh)

    TP_c = {c: 0 for c in range(1, n_classes)}
    FP_c = {c: 0 for c in range(1, n_classes)}
    FN_c = {c: 0 for c in range(1, n_classes)}
    iou_sum_c = {c: 0.0 for c in range(1, n_classes)}

    # 匹配对:按 GT 类型归桶,IoU 用于 SQ
    for g, p, iou in matched:
        c = gt_itype.get(g, 0)
        if 1 <= c < n_classes:
            TP_c[c] += 1
            iou_sum_c[c] += iou

    for g in fn_ids:
        c = gt_itype.get(g, 0)
        if 1 <= c < n_classes:
            FN_c[c] += 1

    for p in fp_ids:
        c = pred_itype.get(p, 0)
        if 1 <= c < n_classes:
            FP_c[c] += 1

    out = {}
    for c in range(1, n_classes):
        TP, FP, FN = TP_c[c], FP_c[c], FN_c[c]
        if TP + FP + FN == 0:
            continue   # 该类不出现于此 patch
        prec = TP / (TP + FP) if TP + FP > 0 else 0.0
        rec  = TP / (TP + FN) if TP + FN > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
        DQ   = TP / (TP + 0.5*FP + 0.5*FN)
        SQ   = (iou_sum_c[c] / TP) if TP > 0 else 0.0
        PQ   = DQ * SQ
        out[c] = dict(precision=prec, recall=rec, f1=f1,
                      PQ=PQ, DQ=DQ, SQ=SQ, TP=TP, FP=FP, FN=FN)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════════════

print("加载 CellViT...")
cellvit = load_cellvit('/home/xuwen/DDPM/CellViT/CellViT-SAM-H-x40.pth',
                       '/home/xuwen/DDPM/CellViT', device, variant='sam_h')

def load_unet(p, s):
    u = create_model(use_semantic=s).to(device)
    c = torch.load(p, map_location=device)
    u.load_state_dict(c['model_state_dict']); u.eval()
    print(f"  加载 {p}  epoch={c.get('epoch','?')}"); return u

print("\n加载 UNet...")
unet_abl = load_unet("/home/xuwen/DDPM/logs/checkpoints_correction_samh/best_unet_ablation.pth", False)
unet_full = load_unet("/home/xuwen/DDPM/logs/checkpoints_correction_samh/best_unet_correction.pth", True)

scheduler = DDPMScheduler(num_train_timesteps=1000)
dataset = PanNukeDataset(fold_dirs=['/data/xuwen/PanNuke/Fold 3'], target_size=256)
print(f"\n测试集: {len(dataset)} 张")

INFER_T, N_RUNS, N_SAMPLES = 200, 5, 200
torch.manual_seed(42); random.seed(42)

# 分层采样
_ti = defaultdict(list)
for _i in range(len(dataset)):
    _s = dataset[_i]
    if _s['gt_nuc_mask'].bool().sum() < 10: continue
    _ti[_s['type_name']].append(_i)
_nt = len(_ti); _pt = max(1, math.ceil(N_SAMPLES/_nt))
si = []
for _t, _ix in sorted(_ti.items()):
    si.extend(random.sample(_ix, min(_pt, len(_ix))))
random.shuffle(si); si = si[:N_SAMPLES]
_tc = defaultdict(int)
for _i in si: _tc[dataset[_i]['type_name']] += 1
print(f'\n分层采样: {len(si)} 张, {_nt} 种 tissue')
for _t,_c in sorted(_tc.items(), key=lambda x:-x[1]):
    print(f'  {_t:<25} {_c:>4}')


def infer_ensemble_inst(unet, hr, sem=None):
    sb = sh = st = None
    t = torch.tensor([INFER_T], device=device)
    for _ in range(N_RUNS):
        n = torch.randn_like(hr)
        nh = scheduler.add_noise(hr, n, t)
        with torch.no_grad():
            np_ = unet(torch.cat([hr,nh],1), t, semantic=sem).sample
            x0 = predict_x0_from_noise_shared(nh, np_, t, scheduler)
            o = run_cellvit_full(cellvit, x0)
        sb = o['nuclei_binary_map'] if sb is None else sb+o['nuclei_binary_map']
        sh = o['hv_map'] if sh is None else sh+o['hv_map']
        st = o['nuclei_type_map'] if st is None else st+o['nuclei_type_map']
    avg = {'nuclei_binary_map':sb/N_RUNS, 'hv_map':sh/N_RUNS, 'nuclei_type_map':st/N_RUNS}
    return cellvit_output_to_instance_map(cellvit, avg, 40)


keys = ['hr','ablation','full']
gm = {k: defaultdict(list) for k in keys}
tm = defaultdict(lambda: {k: defaultdict(list) for k in keys})
cm = defaultdict(lambda: {k: defaultdict(list) for k in keys})

# ── 可视化辅助 ───────────────────────────────────────────────────────
CLASS_COLORS = np.array([
    [0,0,0],[255,0,0],[0,0,255],[255,255,0],[0,255,0],[255,0,255]
], dtype=np.float32) / 255.0

def inst_overlay(img_rgb, inst_map, inst_type, alpha=0.45):
    """实例分割 overlay：每个实例用对应细胞类型颜色描边（不填充）。"""
    overlay = img_rgb.copy()
    from scipy.ndimage import binary_dilation, binary_erosion
    for iid in np.unique(inst_map):
        if iid == 0: continue
        mask = (inst_map == iid)
        ctype = inst_type.get(iid, 0)
        ctype = min(ctype, len(CLASS_COLORS)-1)
        color = CLASS_COLORS[ctype]
        # 仅描边，不填充内部
        boundary = binary_dilation(mask, iterations=1) & ~binary_erosion(mask, iterations=1)
        overlay[boundary] = color
    return np.clip(overlay, 0, 1)

def to_rgb(t):
    return t.squeeze(0).detach().cpu().clamp(0,1).permute(1,2,0).numpy()

# 典型案例收集：按 tissue type 各存一个 PQ 提升最大的案例
N_VIS_PER_TISSUE = 1
best_vis_cases = {}    # {tissue_name: case_dict}
best_vis_delta = {}    # {tissue_name: pq_delta}

print(f"\n开始评估（t={INFER_T}, runs={N_RUNS}, N={N_SAMPLES}）...")
print(f"{'idx':>4}  {'Tissue':<18}  {'HR_F1':>7}  {'Abl_F1':>7}  {'Full_F1':>7}  "
      f"{'HR_PQ':>7}  {'Full_PQ':>7}  {'AJI_F':>7}")

nv = 0
for i in si:
    s = dataset[i]; hr = s['hr'].unsqueeze(0).to(device); tn = s['type_name']
    fi, li = dataset.index[i]
    gt_inst, gt_it = build_gt_instance_map(np.asarray(dataset.masks[fi][li]), (256,256))
    if len(gt_it) < 2: continue

    with torch.no_grad():
        ho = run_cellvit_full(cellvit, hr)
        sem = build_sem_tensor_from_cellvit(
            F.softmax(ho['nuclei_type_map'],1),
            F.softmax(ho['nuclei_binary_map'],1)[:,1])
    hr_inst, hr_it = cellvit_output_to_instance_map(cellvit, ho, 40)
    abl_inst, abl_it = infer_ensemble_inst(unet_abl, hr, None)
    full_inst, full_it = infer_ensemble_inst(unet_full, hr, sem)

    for mk, pi, pit in [('hr',hr_inst,hr_it),('ablation',abl_inst,abl_it),('full',full_inst,full_it)]:
        d = compute_det(gt_inst, pi); aji = compute_aji(gt_inst, pi)
        dice = compute_dice_m(gt_inst, pi)
        for k,v in d.items(): gm[mk][k].append(v)
        gm[mk]['AJI'].append(aji); gm[mk]['DICE'].append(dice)
        for k,v in d.items(): tm[tn][mk][k].append(v)
        tm[tn][mk]['AJI'].append(aji); tm[tn][mk]['DICE'].append(dice)
        # per-cell-type 改用做法 B:全局匹配 → 按 GT 类型拆 TP/FP/FN
        cls_metrics = compute_det_per_class(gt_inst, gt_it, pi, pit, n_classes=6)
        for ct, mdict in cls_metrics.items():
            for k2, v2 in mdict.items():
                cm[ct][mk][k2].append(v2)

    if nv % 20 == 0:
        print(f"{nv:>4}  {tn:<18}  "
              f"{gm['hr']['f1'][-1]:>7.4f}  {gm['ablation']['f1'][-1]:>7.4f}  "
              f"{gm['full']['f1'][-1]:>7.4f}  {gm['hr']['PQ'][-1]:>7.4f}  "
              f"{gm['full']['PQ'][-1]:>7.4f}  {gm['full']['AJI'][-1]:>7.4f}")

    # ── 收集可视化案例：PQ 提升最大的 patch ────────────────────────
    pq_delta = gm['full']['PQ'][-1] - gm['hr']['PQ'][-1]
    if tn not in best_vis_delta or pq_delta > best_vis_delta[tn]:
        best_vis_delta[tn] = pq_delta
        best_vis_cases[tn] = dict(
            idx=i, type_name=tn,
            hr_rgb=to_rgb(hr),
            gt_inst=gt_inst.copy(), gt_itype=dict(gt_it),
            hr_inst=hr_inst.copy(), hr_itype=dict(hr_it),
            abl_inst=abl_inst.copy(), abl_itype=dict(abl_it),
            full_inst=full_inst.copy(), full_itype=dict(full_it),
            pq_hr=gm['hr']['PQ'][-1], pq_abl=gm['ablation']['PQ'][-1],
            pq_full=gm['full']['PQ'][-1],
            f1_hr=gm['hr']['f1'][-1], f1_full=gm['full']['f1'][-1],
            aji_full=gm['full']['AJI'][-1],
        )

    nv += 1

# 汇总
def _m(l): v=[x for x in l if not(isinstance(x,float) and x!=x)]; return np.mean(v) if v else float('nan')
W = 100
print(f"\n{'='*W}\n细胞核检测与分割指标（全局）—— CellViT 官方后处理\n{'='*W}")
print(f"{'指标':<14} {'HR基线':>10} {'消融模型':>10} {'本文方法':>10} {'Δ(本文-HR)':>12} {'改善率':>9}")
print("-"*W)
for mn in ['PQ','DQ','SQ','precision','recall','f1','AJI','DICE']:
    h,a,f=_m(gm['hr'][mn]),_m(gm['ablation'][mn]),_m(gm['full'][mn])
    d=f-h; ar='↑' if d>.001 else('↓' if d<-.001 else '—')
    imp = _improvement_ratio(h, f)
    print(f"{mn:<14} {h:>10.4f} {a:>10.4f} {f:>10.4f} {d:>+10.4f} {ar} {_fmt_pct(imp):>9}")
print("说明:改善率 = (Full − HR) / (1 − HR),覆盖了 HR→1.0 差距的多少")

for mk,lb in [('hr','HR'),('ablation','消融'),('full','本文')]:
    print(f"  {lb} 平均: TP={_m(gm[mk]['TP']):.1f} FP={_m(gm[mk]['FP']):.1f} FN={_m(gm[mk]['FN']):.1f}")

W2 = 130
print(f"\n{'='*W2}\n按 Tissue Type 分组（F1/PQ 加改善率,DICE 三联,AJI 仅 Full)\n{'='*W2}")
print(f"{'Tissue':<22}{'N':>3}  "
      f"{'F1 (HR/Abl/Full)':^28}  "
      f"{'PQ (HR/Abl/Full) | 改善率':^36}  "
      f"{'DICE (HR/Abl/Full)':^26}  "
      f"{'AJI_F':>7}")
print("-"*W2)
for t in sorted(tm):
    n=len(tm[t]['hr']['f1'])
    f_h, f_a, f_f = _m(tm[t]['hr']['f1']), _m(tm[t]['ablation']['f1']), _m(tm[t]['full']['f1'])
    p_h, p_a, p_f = _m(tm[t]['hr']['PQ']), _m(tm[t]['ablation']['PQ']), _m(tm[t]['full']['PQ'])
    d_h, d_a, d_f = _m(tm[t]['hr']['DICE']), _m(tm[t]['ablation']['DICE']), _m(tm[t]['full']['DICE'])
    aji_f = _m(tm[t]['full']['AJI'])
    pq_imp = _improvement_ratio(p_h, p_f)
    print(f"{t:<22}{n:>3}  "
          f"{f_h:>7.4f}/{f_a:.4f}/{f_f:.4f}  "
          f"{p_h:>7.4f}/{p_a:.4f}/{p_f:.4f} | {_fmt_pct(pq_imp)}  "
          f"{d_h:>7.4f}/{d_a:.4f}/{d_f:.4f}  "
          f"{aji_f:>7.4f}")

print(f"\n{'='*W2}\n按细胞类型 · 检测三联(全局匹配后按 GT 类拆,做法 B)\n{'='*W2}")
print("说明:全局做检测匹配,再按 GT 类型分桶 TP/FN,按 pred 类型分桶 FP")
print("    分类错误不双倍计费(空间对了即 TP),分类质量另见 confusion_matrix_compare.py")
print(f"{'类别':<14}{'N':>4}  "
      f"{'F1 (HR/Abl/Full)':^28}  "
      f"{'Precision (HR/Abl/Full)':^32}  "
      f"{'Recall (HR/Abl/Full)':^28}")
print("-"*W2)
for ct in range(1,6):
    if ct not in cm: continue
    n = len(cm[ct]['hr']['f1'])
    f_h, f_a, f_f = _m(cm[ct]['hr']['f1']), _m(cm[ct]['ablation']['f1']), _m(cm[ct]['full']['f1'])
    p_h, p_a, p_f = _m(cm[ct]['hr']['precision']), _m(cm[ct]['ablation']['precision']), _m(cm[ct]['full']['precision'])
    r_h, r_a, r_f = _m(cm[ct]['hr']['recall']), _m(cm[ct]['ablation']['recall']), _m(cm[ct]['full']['recall'])
    print(f"{CLASS_NAMES[ct]:<14}{n:>4}  "
          f"{f_h:>7.4f}/{f_a:.4f}/{f_f:.4f}  "
          f"{p_h:>9.4f}/{p_a:.4f}/{p_f:.4f}  "
          f"{r_h:>7.4f}/{r_a:.4f}/{r_f:.4f}")

print(f"\n{'='*W2}\n按细胞类型 · 全景质量(PQ = DQ × SQ)\n{'='*W2}")
print(f"{'类别':<14}{'N':>4}  "
      f"{'PQ (HR/Abl/Full)':^28}  "
      f"{'DQ (HR/Abl/Full)':^28}  "
      f"{'SQ (HR/Abl/Full)':^28}  "
      f"{'PQ改善率':>9}")
print("-"*W2)
for ct in range(1,6):
    if ct not in cm: continue
    n = len(cm[ct]['hr']['PQ'])
    pq_h, pq_a, pq_f = _m(cm[ct]['hr']['PQ']), _m(cm[ct]['ablation']['PQ']), _m(cm[ct]['full']['PQ'])
    dq_h, dq_a, dq_f = _m(cm[ct]['hr']['DQ']), _m(cm[ct]['ablation']['DQ']), _m(cm[ct]['full']['DQ'])
    sq_h, sq_a, sq_f = _m(cm[ct]['hr']['SQ']), _m(cm[ct]['ablation']['SQ']), _m(cm[ct]['full']['SQ'])
    pq_imp = _improvement_ratio(pq_h, pq_f)
    print(f"{CLASS_NAMES[ct]:<14}{n:>4}  "
          f"{pq_h:>7.4f}/{pq_a:.4f}/{pq_f:.4f}  "
          f"{dq_h:>7.4f}/{dq_a:.4f}/{dq_f:.4f}  "
          f"{sq_h:>7.4f}/{sq_a:.4f}/{sq_f:.4f}  "
          f"{_fmt_pct(pq_imp):>9}")

print(f"\n有效样本: {nv}")

# ═══════════════════════════════════════════════════════════════════════════
# 可视化：典型检测/分割案例（每种 tissue 一行）
# ═══════════════════════════════════════════════════════════════════════════
vis_cases = [best_vis_cases[t] for t in sorted(best_vis_cases) if best_vis_cases[t] is not None]
# 按 PQ 提升降序排列，最多取 8 行
vis_cases.sort(key=lambda c: c['pq_full'] - c['pq_hr'], reverse=True)
vis_cases = vis_cases[:8]

if vis_cases:
    n_rows = len(vis_cases)
    col_titles = ['HR image', 'GT instances', 'HR pred', 'Ablation pred', 'Full pred (ours)']
    n_cols = len(col_titles)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5*n_cols, 3.8*n_rows), squeeze=False)
    fig.suptitle(
        f'Detection & Segmentation — best PQ improvement per tissue\n'
        f'(CellViT official post-processing, ensemble N={N_RUNS})',
        fontsize=12, y=1.01)

    for ci, title in enumerate(col_titles):
        axes[0, ci].set_title(title, fontsize=10, pad=6)

    for ri, c in enumerate(vis_cases):
        img = c['hr_rgb']

        # GT overlay
        gt_ov = inst_overlay(img, c['gt_inst'], c['gt_itype'])
        # HR pred overlay
        hr_ov = inst_overlay(img, c['hr_inst'], c['hr_itype'])
        # Ablation pred overlay
        abl_ov = inst_overlay(img, c['abl_inst'], c['abl_itype'])
        # Full pred overlay
        full_ov = inst_overlay(img, c['full_inst'], c['full_itype'])

        panels = [img, gt_ov, hr_ov, abl_ov, full_ov]
        for ci, panel in enumerate(panels):
            axes[ri, ci].imshow(panel)
            axes[ri, ci].axis('off')

        # 左侧标注 tissue 名称
        axes[ri, 0].text(
            0.02, 0.98, f"[{c['type_name']}]\nidx={c['idx']}",
            transform=axes[ri, 0].transAxes, fontsize=7, fontweight='bold',
            va='top', ha='left', color='white',
            bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.7))

        # 右侧标注指标
        axes[ri, -1].text(
            0.98, 0.02,
            f"PQ: {c['pq_hr']:.3f}→{c['pq_full']:.3f} (Δ={c['pq_full']-c['pq_hr']:+.3f})\n"
            f"F1: {c['f1_hr']:.3f}→{c['f1_full']:.3f}\n"
            f"AJI: {c['aji_full']:.3f}",
            transform=axes[ri, -1].transAxes, fontsize=7, va='bottom', ha='right',
            color='white',
            bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.7))

    # 图例
    legend_patches = [mpatches.Patch(color=CLASS_COLORS[c], label=CLASS_NAMES[c])
                      for c in range(1, 6)]
    fig.legend(handles=legend_patches, title='Cell type', loc='upper right',
               fontsize=8, ncol=1, bbox_to_anchor=(0.99, 0.99))

    plt.tight_layout()
    save_path = './logs/downstream_Cellvitsamh/detection_segmentation_cases.png'
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n可视化已保存: {save_path}")

    # ── 额外图：per-tissue PQ 对比条形图 ─────────────────────────────
    all_tissues_sorted = sorted(tm.keys())
    n_t = len(all_tissues_sorted)
    fig2, ax2 = plt.subplots(figsize=(max(10, n_t*1.2), 5))
    x = np.arange(n_t); w = 0.25
    pq_hr_list = [_m(tm[t]['hr']['PQ']) for t in all_tissues_sorted]
    pq_abl_list = [_m(tm[t]['ablation']['PQ']) for t in all_tissues_sorted]
    pq_ful_list = [_m(tm[t]['full']['PQ']) for t in all_tissues_sorted]
    ax2.bar(x-w, pq_hr_list, w, label='HR baseline', color='#4878D0')
    ax2.bar(x, pq_abl_list, w, label='Ablation', color='#EE854A')
    ax2.bar(x+w, pq_ful_list, w, label='Full (ours)', color='#6ACC65')
    ax2.set_xticks(x); ax2.set_xticklabels(all_tissues_sorted, rotation=35, ha='right', fontsize=9)
    ax2.set_ylabel('Panoptic Quality (PQ)')
    ax2.set_title(f'Per-tissue PQ — ensemble N={N_RUNS}')
    ax2.legend(); ax2.set_ylim(0, 1.05); ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    bar_path = './logs/downstream_Cellvitsamh/detection_pq_per_tissue.png'
    os.makedirs(os.path.dirname(bar_path) or '.', exist_ok=True)
    plt.savefig(bar_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"PQ 条形图已保存: {bar_path}")
else:
    print("\n⚠️ 无可视化案例")