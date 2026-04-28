"""
eval_nuclear_atypia.py
核异型性评分任务评估脚本。

三组对比：HR 基线 / 消融模型 / 本文方法

【后处理】
  使用 CellViT 官方 model.calculate_instance_map()

【评估体系 · 三层】
─────────────────────────────────────────────────────────────────────────────

第一层：形态学特征保真度（Feature Fidelity）
  从实例分割提取每个核的：面积、周长、圆度、离心率、凸度、纵横比
  指标：与 GT 特征中位数的相对误差

第二层：组织学指标（Histological Metrics）
  核密度、核面积占比、多形性指数（面积 CoV）、核聚集度（最近邻距离）
  指标：与 GT 的绝对误差

第三层：综合异型性评分（Atypia Score, 0-1）
  加权：面积异常度(0.3) + 形态不规则度(0.3) + 大小异质性(0.2) + 核密度(0.2)
  指标：MAE / RMSE / Spearman ρ

以上均按 tissue type 分组输出。
"""

import numpy as np
import torch
import torch.nn.functional as F
import sys
import os
import random
import math
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import OrderedDict, defaultdict
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

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


# ═══════════════════════════════════════════════════════════════════════════
# CellViT 推理 + 官方后处理（同 eval_detection_segmentation.py）
# ═══════════════════════════════════════════════════════════════════════════

def run_cellvit_full(cellvit, img_01):
    with torch.no_grad():
        return cellvit(img_01)


def cellvit_output_to_instance_map(cellvit_model, raw_output, magnification=40):
    predictions = OrderedDict()
    predictions['nuclei_binary_map'] = raw_output['nuclei_binary_map']
    predictions['nuclei_type_map'] = raw_output['nuclei_type_map']
    predictions['hv_map'] = raw_output['hv_map']

    instance_map, cell_dict_list = cellvit_model.calculate_instance_map(
        predictions, magnification=magnification
    )
    inst_map = instance_map.squeeze(0).cpu().numpy().astype(np.int32)

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
    if not inst_type:
        tp = F.softmax(raw_output['nuclei_type_map'], dim=1)
        tl = tp.argmax(dim=1).squeeze(0).cpu().numpy()
        for iid in np.unique(inst_map):
            if iid == 0: continue
            r = tl[inst_map == iid]; nz = r[r > 0]
            if len(nz) > 0:
                v, c = np.unique(nz, return_counts=True)
                inst_type[int(iid)] = int(v[c.argmax()])
    return inst_map, inst_type


def build_gt_instance_map(mask_raw, target_hw=None):
    m = np.asarray(mask_raw)
    if m.ndim != 3: raise ValueError(f"mask shape: {m.shape}")
    if m.shape[0] in (5,6) and m.shape[-1] not in (5,6):
        m = np.transpose(m, (1,2,0))
    H,W,C = m.shape; C = min(C,5); m = m[...,:C].astype(np.int32)
    inst_map = np.zeros((H,W), dtype=np.int32); inst_type = {}; gid = 0
    for ch in range(C):
        for lid in np.unique(m[...,ch]):
            if lid == 0: continue
            gid += 1; inst_map[m[...,ch]==lid] = gid; inst_type[gid] = ch+1
    if target_hw:
        th,tw = target_hw
        if (H,W) != (th,tw):
            inst_map = cv2.resize(inst_map.astype(np.float32),(tw,th),
                                  interpolation=cv2.INTER_NEAREST).astype(np.int32)
    return inst_map, inst_type


# ═══════════════════════════════════════════════════════════════════════════
# 形态学特征提取
# ═══════════════════════════════════════════════════════════════════════════

def extract_morphological_features(inst_map, inst_type=None):
    """
    从实例分割图提取每个核的形态学特征 + patch 级统计 + 异型性评分。
    """
    inst_ids = np.unique(inst_map); inst_ids = inst_ids[inst_ids > 0]
    H, W = inst_map.shape

    if len(inst_ids) == 0:
        return [], _empty_summary()

    features = []
    for iid in inst_ids:
        mask = (inst_map == iid).astype(np.uint8)
        area = int(mask.sum())
        if area < 5: continue

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours: continue
        cnt = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(cnt, True)

        circularity = min((4*np.pi*area / (perimeter**2+1e-8)), 1.0) if perimeter > 0 else 0.

        eccentricity, aspect_ratio = 0., 1.
        if len(cnt) >= 5:
            try:
                (_,_), (mi,ma), _ = cv2.fitEllipse(cnt)
                a,b = max(ma,mi)/2., min(ma,mi)/2.
                if a > 0 and b > 0:
                    eccentricity = np.sqrt(1-(b/a)**2)
                    aspect_ratio = a/(b+1e-8)
            except cv2.error: pass

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area/(hull_area+1e-8) if hull_area > 0 else 0.

        M = cv2.moments(mask)
        cx = M['m10']/(M['m00']+1e-8) if M['m00']>0 else np.mean(np.where(mask>0)[1])
        cy = M['m01']/(M['m00']+1e-8) if M['m00']>0 else np.mean(np.where(mask>0)[0])

        features.append(dict(
            inst_id=int(iid), cell_type=inst_type.get(iid,0) if inst_type else 0,
            area=area, perimeter=float(perimeter), circularity=float(circularity),
            eccentricity=float(eccentricity), solidity=float(solidity),
            aspect_ratio=float(aspect_ratio), centroid_x=float(cx), centroid_y=float(cy),
        ))

    return features, _compute_summary(features, H, W)


def _empty_summary():
    return dict(n_nuclei=0, nuclear_density=0., nuclear_area_ratio=0.,
                mean_area=0., median_area=0., std_area=0., area_cov=0.,
                mean_circularity=0., median_circularity=0.,
                mean_eccentricity=0., mean_solidity=0., mean_aspect_ratio=1.,
                mean_nnd=0., std_nnd=0., atypia_score=0.)


def _compute_summary(features, H, W):
    if not features: return _empty_summary()

    areas = np.array([f['area'] for f in features])
    circs = np.array([f['circularity'] for f in features])
    eccs  = np.array([f['eccentricity'] for f in features])
    sols  = np.array([f['solidity'] for f in features])
    ars   = np.array([f['aspect_ratio'] for f in features])
    cents = np.array([[f['centroid_x'], f['centroid_y']] for f in features])

    n = len(features); pa = H*W
    nd = n / (pa/10000.)
    nar = areas.sum() / pa
    ma, mda, sa = float(areas.mean()), float(np.median(areas)), float(areas.std())
    acov = sa/(ma+1e-8)

    mn, sn = 0., 0.
    if n > 1:
        d = cdist(cents, cents); np.fill_diagonal(d, np.inf)
        nnd = d.min(axis=1); mn, sn = float(nnd.mean()), float(nnd.std())

    # 异型性评分
    NR = (50, 200)
    ad = np.mean([max(0,a-NR[1])/NR[1]+max(0,NR[0]-a)/NR[0] for a in areas])
    area_at = min(ad/2., 1.)
    circ_at = float(np.mean(np.maximum(0, 0.7-circs)/0.7))
    sol_at  = float(np.mean(np.maximum(0, 0.9-sols)/0.9))
    shape_at = min((circ_at+sol_at)/2., 1.)
    size_het = min(acov/1., 1.)
    dens_at  = min(max(0, nd-50)/50., 1.)
    atypia = float(np.clip(0.3*area_at + 0.3*shape_at + 0.2*size_het + 0.2*dens_at, 0, 1))

    return dict(n_nuclei=n, nuclear_density=float(nd), nuclear_area_ratio=float(nar),
                mean_area=ma, median_area=mda, std_area=sa, area_cov=float(acov),
                mean_circularity=float(circs.mean()), median_circularity=float(np.median(circs)),
                mean_eccentricity=float(eccs.mean()), mean_solidity=float(sols.mean()),
                mean_aspect_ratio=float(ars.mean()), mean_nnd=mn, std_nnd=sn,
                atypia_score=atypia)


def compute_feature_fidelity(gt_s, pred_s):
    results = {}
    for key in ['median_area','mean_circularity','mean_eccentricity','mean_solidity',
                'mean_aspect_ratio','nuclear_density','nuclear_area_ratio','area_cov','mean_nnd']:
        g, p = gt_s.get(key,0.), pred_s.get(key,0.)
        results[key] = abs(p-g)/(abs(g)+1e-8) if g != 0 else abs(p-g)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════════════

print("加载 CellViT...")
cellvit = load_cellvit('/home/xuwen/DDPM/CellViT/CellViT-256-x40.pth',
                       '/home/xuwen/DDPM/CellViT', device)

def load_unet(p, s):
    u = create_model(use_semantic=s).to(device)
    c = torch.load(p, map_location=device)
    u.load_state_dict(c['model_state_dict']); u.eval()
    print(f"  加载 {p}  epoch={c.get('epoch','?')}"); return u

print("\n加载 UNet...")
unet_abl = load_unet("/home/xuwen/DDPM/logs/checkpoints_correction_v3/best_unet_ablation.pth", False)
unet_full = load_unet("/home/xuwen/DDPM/logs/checkpoints_correction_v3/best_unet_correction.pth", True)

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


# 容器
keys = ['hr','ablation','full']
gt_summaries, gt_atypia = [], []
patch_summaries = {k:[] for k in keys}
atypia_scores = {k:[] for k in keys}
feat_errs = {k: defaultdict(list) for k in keys}
tissue_atypia = defaultdict(lambda:{k:[] for k in keys})
tissue_gt_atypia = defaultdict(list)
tissue_feat_errs = defaultdict(lambda:{k:defaultdict(list) for k in keys})
tissue_n = defaultdict(int)

# ── 可视化辅助 ───────────────────────────────────────────────────────
CLASS_COLORS = np.array([
    [0,0,0],[255,0,0],[0,0,255],[255,255,0],[0,255,0],[255,0,255]
], dtype=np.float32) / 255.0

def to_rgb(t):
    return t.squeeze(0).detach().cpu().clamp(0,1).permute(1,2,0).numpy()

def draw_morphology_overlay(img_rgb, inst_map, features, inst_type, alpha=0.35):
    """
    形态学特征可视化：
    - 每个实例用对应细胞类型颜色描边（不填充）
    - 面积异常的实例用加粗描边（2 像素）
    """
    from scipy.ndimage import binary_dilation, binary_erosion
    overlay = img_rgb.copy()
    NORMAL_AREA = (50, 200)

    for iid in np.unique(inst_map):
        if iid == 0: continue
        mask = (inst_map == iid)
        ctype = inst_type.get(iid, 0)
        ctype = min(ctype, len(CLASS_COLORS)-1)
        color = CLASS_COLORS[ctype]
        area = mask.sum()
        # 面积异常：加粗描边（2 像素）
        if area < NORMAL_AREA[0] or area > NORMAL_AREA[1]:
            bd = binary_dilation(mask, iterations=2) & ~binary_erosion(mask, iterations=1)
        else:
            bd = binary_dilation(mask, iterations=1) & ~binary_erosion(mask, iterations=1)
        overlay[bd] = color
    return np.clip(overlay, 0, 1)

def draw_atypia_heatmap(inst_map, features):
    """生成核异型性热图：每个实例按其面积偏离正常范围的程度着色。"""
    H, W = inst_map.shape
    heatmap = np.zeros((H, W), dtype=np.float32)
    feat_dict = {f['inst_id']: f for f in features}
    NORMAL = (50, 200)
    for iid in np.unique(inst_map):
        if iid == 0: continue
        f = feat_dict.get(iid)
        if f is None: continue
        area = f['area']
        circ = f['circularity']
        # 综合异常度：面积偏离 + 圆度偏离
        a_dev = max(0, area-NORMAL[1])/NORMAL[1] + max(0, NORMAL[0]-area)/NORMAL[0]
        c_dev = max(0, 0.7-circ)/0.7
        score = min((a_dev + c_dev)/2., 1.0)
        heatmap[inst_map == iid] = score
    return heatmap

# 典型案例收集
best_atypia_cases = {}    # {tissue: case_dict}
best_atypia_delta = {}    # {tissue: |atypia_delta|}

print(f"\n开始评估（t={INFER_T}, runs={N_RUNS}, N={N_SAMPLES}）...")
print(f"{'idx':>4}  {'Tissue':<18}  {'GT':>8}  {'HR':>8}  {'Abl':>8}  {'Full':>8}")

nv = 0
for i in si:
    s = dataset[i]; hr = s['hr'].unsqueeze(0).to(device); tn = s['type_name']
    fi, li = dataset.index[i]
    gt_inst, gt_it = build_gt_instance_map(np.asarray(dataset.masks[fi][li]), (256,256))
    if len(gt_it) < 3: continue

    _, gt_sum = extract_morphological_features(gt_inst, gt_it)
    if gt_sum['n_nuclei'] < 3: continue
    gt_summaries.append(gt_sum); gt_atypia.append(gt_sum['atypia_score'])
    # 保存 features 用于可视化
    gt_feats, _ = extract_morphological_features(gt_inst, gt_it)

    with torch.no_grad():
        ho = run_cellvit_full(cellvit, hr)
        sem = build_sem_tensor_from_cellvit(
            F.softmax(ho['nuclei_type_map'],1),
            F.softmax(ho['nuclei_binary_map'],1)[:,1])
    hr_inst, hr_it = cellvit_output_to_instance_map(cellvit, ho, 40)
    abl_inst, abl_it = infer_ensemble_inst(unet_abl, hr, None)
    full_inst, full_it = infer_ensemble_inst(unet_full, hr, sem)

    # 保存 features 用于可视化
    hr_feats, _ = extract_morphological_features(hr_inst, hr_it)
    full_feats, _ = extract_morphological_features(full_inst, full_it)

    for mk, pi, pit in [('hr',hr_inst,hr_it),('ablation',abl_inst,abl_it),('full',full_inst,full_it)]:
        _, psum = extract_morphological_features(pi, pit)
        patch_summaries[mk].append(psum)
        atypia_scores[mk].append(psum['atypia_score'])
        fid = compute_feature_fidelity(gt_sum, psum)
        for fk, fv in fid.items(): feat_errs[mk][fk].append(fv)
        tissue_atypia[tn][mk].append(psum['atypia_score'])
        for fk, fv in fid.items(): tissue_feat_errs[tn][mk][fk].append(fv)

    tissue_gt_atypia[tn].append(gt_sum['atypia_score'])
    tissue_n[tn] += 1

    if nv % 20 == 0:
        print(f"{nv:>4}  {tn:<18}  {gt_sum['atypia_score']:>8.4f}  "
              f"{patch_summaries['hr'][-1]['atypia_score']:>8.4f}  "
              f"{patch_summaries['ablation'][-1]['atypia_score']:>8.4f}  "
              f"{patch_summaries['full'][-1]['atypia_score']:>8.4f}")

    # ── 收集可视化案例：异型性评分 MAE 改善最大的 patch ────────────
    hr_mae_i  = abs(patch_summaries['hr'][-1]['atypia_score'] - gt_sum['atypia_score'])
    full_mae_i = abs(patch_summaries['full'][-1]['atypia_score'] - gt_sum['atypia_score'])
    delta_mae = hr_mae_i - full_mae_i  # 正值 = Full 更接近 GT = 改善
    if tn not in best_atypia_delta or delta_mae > best_atypia_delta[tn]:
        best_atypia_delta[tn] = delta_mae
        best_atypia_cases[tn] = dict(
            idx=i, type_name=tn,
            hr_rgb=to_rgb(hr),
            gt_inst=gt_inst.copy(), gt_itype=dict(gt_it),
            hr_inst=hr_inst.copy(), hr_itype=dict(hr_it),
            full_inst=full_inst.copy(), full_itype=dict(full_it),
            gt_features=gt_feats, hr_features=hr_feats, full_features=full_feats,
            gt_summary=dict(gt_sum),
            hr_summary=dict(patch_summaries['hr'][-1]),
            full_summary=dict(patch_summaries['full'][-1]),
        )

    nv += 1


# ═══════════════════════════════════════════════════════════════════════════
# 汇总
# ═══════════════════════════════════════════════════════════════════════════
def _m(l): v=[x for x in l if not(isinstance(x,float) and x!=x)]; return np.mean(v) if v else float('nan')

W = 95

# 第一层
print(f"\n{'='*W}")
print("第一层：形态学特征保真度（相对误差 vs GT，↓ 越低越好）")
print(f"{'='*W}")
feat_display = [('median_area','面积中位数'),('mean_circularity','平均圆度'),
    ('mean_eccentricity','平均离心率'),('mean_solidity','平均凸度'),
    ('mean_aspect_ratio','平均纵横比'),('nuclear_density','核密度'),
    ('nuclear_area_ratio','核面积占比'),('area_cov','面积CoV(多形性)'),
    ('mean_nnd','平均最近邻距离')]
print(f"{'特征':<20} {'HR基线':>12} {'消融模型':>12} {'本文方法':>12} {'Δ(本文-HR)':>12}")
print("-"*W)
for fk, fn in feat_display:
    h,a,f=_m(feat_errs['hr'][fk]),_m(feat_errs['ablation'][fk]),_m(feat_errs['full'][fk])
    d=f-h; ar='↓改善' if d<-.001 else('↑退步' if d>.001 else '—')
    print(f"{fn:<20} {h:>12.4f} {a:>12.4f} {f:>12.4f} {d:>+10.4f} {ar}")

# 第二层
print(f"\n{'='*W}")
print("第二层：组织学指标（Patch 级均值，括号内为 vs GT 的 MAE）")
print(f"{'='*W}")
hk = [('nuclear_density','核密度(/万px)'),('nuclear_area_ratio','核面积占比'),
      ('area_cov','多形性(CoV)'),('mean_nnd','最近邻距离'),('n_nuclei','核数量')]
print(f"{'指标':<20} {'GT':>10} {'HR(MAE)':>15} {'消融(MAE)':>15} {'本文(MAE)':>15}")
print("-"*W)
for k, nm in hk:
    gv = [s[k] for s in gt_summaries]
    hv = [s[k] for s in patch_summaries['hr']]
    av = [s[k] for s in patch_summaries['ablation']]
    fv = [s[k] for s in patch_summaries['full']]
    gm = np.mean(gv)
    hm,am,fm = np.mean(hv),np.mean(av),np.mean(fv)
    hmae = np.mean(np.abs(np.array(hv)-np.array(gv)))
    amae = np.mean(np.abs(np.array(av)-np.array(gv)))
    fmae = np.mean(np.abs(np.array(fv)-np.array(gv)))
    print(f"{nm:<20} {gm:>10.2f} {hm:>7.2f}({hmae:.2f}) {am:>7.2f}({amae:.2f}) {fm:>7.2f}({fmae:.2f})")

# 第三层
print(f"\n{'='*W}")
print("第三层：综合异型性评分（Atypia Score, 0-1）")
print(f"{'='*W}")

ga = np.array(gt_atypia)
ha = np.array(atypia_scores['hr'])
aa = np.array(atypia_scores['ablation'])
fa = np.array(atypia_scores['full'])

hmae,amae,fmae = np.mean(np.abs(ha-ga)), np.mean(np.abs(aa-ga)), np.mean(np.abs(fa-ga))
hrmse = np.sqrt(np.mean((ha-ga)**2))
armse = np.sqrt(np.mean((aa-ga)**2))
frmse = np.sqrt(np.mean((fa-ga)**2))
hrho,_ = spearmanr(ga,ha); arho,_ = spearmanr(ga,aa); frho,_ = spearmanr(ga,fa)

print(f"\n{'指标':<22} {'HR基线':>12} {'消融模型':>12} {'本文方法':>12} {'Δ(本文-HR)':>12}")
print("-"*W)
for nm, hv, av, fv in [('Atypia MAE',hmae,amae,fmae),('Atypia RMSE',hrmse,armse,frmse)]:
    d=fv-hv; ar='↓改善' if d<-.001 else('↑退步' if d>.001 else '—')
    print(f"{nm:<22} {hv:>12.4f} {av:>12.4f} {fv:>12.4f} {d:>+10.4f} {ar}")
d=frho-hrho; ar='↑改善' if d>.001 else('↓退步' if d<-.001 else '—')
print(f"{'Spearman ρ':<22} {hrho:>12.4f} {arho:>12.4f} {frho:>12.4f} {d:>+10.4f} {ar}")

print(f"\n  GT Atypia: mean={ga.mean():.4f} std={ga.std():.4f} "
      f"min={ga.min():.4f} max={ga.max():.4f}")

# per tissue
print(f"\n{'='*W}")
print("按 Tissue Type —— 异型性评分 MAE & Spearman ρ")
print(f"{'='*W}")
print(f"{'Tissue':<22}{'N':>3}  {'GT_mean':>8}  {'MAE':^26}  {'ρ(Full)':>8}")
print(f"{'':<22}{'':>3}  {'':>8}  {'HR':>7}{'Abl':>8}{'Full':>8}  {'':>8}")
print("-"*W)
for t in sorted(tissue_atypia):
    n = tissue_n[t]; tg = np.array(tissue_gt_atypia[t])
    th = np.array(tissue_atypia[t]['hr']); ta = np.array(tissue_atypia[t]['ablation'])
    tf = np.array(tissue_atypia[t]['full'])
    mh = np.mean(np.abs(th-tg)); ma_ = np.mean(np.abs(ta-tg)); mf = np.mean(np.abs(tf-tg))
    tr = spearmanr(tg,tf)[0] if n>=3 else float('nan')
    d=mf-mh; ar='↓' if d<-.001 else('↑' if d>.001 else '—')
    print(f"{t:<22}{n:>3}  {tg.mean():>8.4f}  {mh:>7.4f}{ma_:>8.4f}{mf:>8.4f}  {tr:>8.4f}  Δ={d:+.4f}{ar}")

print(f"\n{'='*W}\n有效样本: {nv}\n{'='*W}")

# ═══════════════════════════════════════════════════════════════════════════
# 可视化 1：典型异型性案例（每种 tissue 一行，7 列）
# ═══════════════════════════════════════════════════════════════════════════
vis_cases = [best_atypia_cases[t] for t in sorted(best_atypia_cases)
             if best_atypia_cases[t] is not None]
vis_cases.sort(key=lambda c: best_atypia_delta.get(c['type_name'], 0), reverse=True)
vis_cases = vis_cases[:8]

if vis_cases:
    n_rows = len(vis_cases)
    col_titles = ['HR image',
                  'GT morphology', 'GT atypia heatmap',
                  'HR morphology', 'HR atypia heatmap',
                  'Full morphology', 'Full atypia heatmap']
    n_cols = len(col_titles)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.*n_cols, 3.5*n_rows), squeeze=False)
    fig.suptitle(
        f'Nuclear Atypia — best improvement cases per tissue\n'
        f'(orange boundary = abnormal area, heatmap = per-nuclei atypia score)',
        fontsize=11, y=1.01)

    for ci, title in enumerate(col_titles):
        axes[0, ci].set_title(title, fontsize=9, pad=5)

    for ri, c in enumerate(vis_cases):
        img = c['hr_rgb']

        # GT
        gt_morph = draw_morphology_overlay(img, c['gt_inst'], c['gt_features'], c['gt_itype'])
        gt_heat  = draw_atypia_heatmap(c['gt_inst'], c['gt_features'])
        # HR pred
        hr_morph = draw_morphology_overlay(img, c['hr_inst'], c['hr_features'], c['hr_itype'])
        hr_heat  = draw_atypia_heatmap(c['hr_inst'], c['hr_features'])
        # Full pred
        full_morph = draw_morphology_overlay(img, c['full_inst'], c['full_features'], c['full_itype'])
        full_heat  = draw_atypia_heatmap(c['full_inst'], c['full_features'])

        axes[ri, 0].imshow(img); axes[ri, 0].axis('off')
        axes[ri, 1].imshow(gt_morph); axes[ri, 1].axis('off')
        axes[ri, 2].imshow(gt_heat, cmap='hot', vmin=0, vmax=1); axes[ri, 2].axis('off')
        axes[ri, 3].imshow(hr_morph); axes[ri, 3].axis('off')
        axes[ri, 4].imshow(hr_heat, cmap='hot', vmin=0, vmax=1); axes[ri, 4].axis('off')
        axes[ri, 5].imshow(full_morph); axes[ri, 5].axis('off')
        axes[ri, 6].imshow(full_heat, cmap='hot', vmin=0, vmax=1); axes[ri, 6].axis('off')

        # 标注
        gs = c['gt_summary']; hs = c['hr_summary']; fs = c['full_summary']
        axes[ri, 0].text(
            0.02, 0.98, f"[{c['type_name']}]\nidx={c['idx']}",
            transform=axes[ri, 0].transAxes, fontsize=7, fontweight='bold',
            va='top', ha='left', color='white',
            bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.7))
        axes[ri, 2].text(
            0.5, 0.02, f"atypia={gs['atypia_score']:.3f}\nN={gs['n_nuclei']}",
            transform=axes[ri, 2].transAxes, fontsize=7, va='bottom', ha='center',
            color='white', bbox=dict(fc='black', alpha=0.6, pad=1.5))
        axes[ri, 4].text(
            0.5, 0.02, f"atypia={hs['atypia_score']:.3f}\nN={hs['n_nuclei']}",
            transform=axes[ri, 4].transAxes, fontsize=7, va='bottom', ha='center',
            color='white', bbox=dict(fc='black', alpha=0.6, pad=1.5))
        axes[ri, 6].text(
            0.5, 0.02, f"atypia={fs['atypia_score']:.3f}\nN={fs['n_nuclei']}",
            transform=axes[ri, 6].transAxes, fontsize=7, va='bottom', ha='center',
            color='white', bbox=dict(fc='black', alpha=0.6, pad=1.5))

    legend_patches = [mpatches.Patch(color=CLASS_COLORS[c], label=CLASS_NAMES[c])
                      for c in range(1,6)]
    fig.legend(handles=legend_patches, title='Cell type', loc='upper right',
               fontsize=7, ncol=1, bbox_to_anchor=(0.99, 0.99))

    plt.tight_layout()
    os.makedirs('./logs', exist_ok=True)
    save_path = './logs/nuclear_atypia_cases.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n异型性可视化已保存: {save_path}")

    # ── 可视化 2：Atypia Score 散点图（GT vs 各方法） ──────────────────
    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4.5))
    fig3.suptitle('Atypia Score: GT vs Predictions', fontsize=12, y=1.02)

    for ax, mk, label, color in [
        (axes3[0], 'hr',       'HR baseline',   '#4878D0'),
        (axes3[1], 'ablation', 'Ablation',      '#EE854A'),
        (axes3[2], 'full',     'Full (ours)',    '#6ACC65'),
    ]:
        pred_arr = np.array(atypia_scores[mk])
        ax.scatter(ga, pred_arr, alpha=0.4, s=15, color=color)
        ax.plot([0,1], [0,1], 'k--', alpha=0.3, linewidth=1)
        rho, _ = spearmanr(ga, pred_arr)
        mae = np.mean(np.abs(pred_arr - ga))
        ax.set_xlabel('GT Atypia Score'); ax.set_ylabel('Predicted Atypia Score')
        ax.set_title(f'{label}\nMAE={mae:.4f}  ρ={rho:.4f}', fontsize=10)
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.grid(alpha=0.2)

    plt.tight_layout()
    scatter_path = './logs/atypia_scatter.png'
    plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"散点图已保存: {scatter_path}")

    # ── 可视化 3：Per-tissue Atypia MAE 条形图 ────────────────────────
    all_t = sorted(tissue_atypia.keys())
    n_t = len(all_t)
    fig4, ax4 = plt.subplots(figsize=(max(10, n_t*1.2), 5))
    x = np.arange(n_t); w = 0.25
    mae_hr  = [np.mean(np.abs(np.array(tissue_atypia[t]['hr'])-np.array(tissue_gt_atypia[t])))
               for t in all_t]
    mae_abl = [np.mean(np.abs(np.array(tissue_atypia[t]['ablation'])-np.array(tissue_gt_atypia[t])))
               for t in all_t]
    mae_ful = [np.mean(np.abs(np.array(tissue_atypia[t]['full'])-np.array(tissue_gt_atypia[t])))
               for t in all_t]
    ax4.bar(x-w, mae_hr, w, label='HR baseline', color='#4878D0')
    ax4.bar(x, mae_abl, w, label='Ablation', color='#EE854A')
    ax4.bar(x+w, mae_ful, w, label='Full (ours)', color='#6ACC65')
    ax4.set_xticks(x); ax4.set_xticklabels(all_t, rotation=35, ha='right', fontsize=9)
    ax4.set_ylabel('Atypia Score MAE (↓ lower is better)')
    ax4.set_title(f'Per-tissue Atypia MAE — ensemble N={N_RUNS}')
    ax4.legend(); ax4.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    bar_path = './logs/atypia_mae_per_tissue.png'
    plt.savefig(bar_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"MAE 条形图已保存: {bar_path}")
else:
    print("\n⚠️ 无可视化案例")