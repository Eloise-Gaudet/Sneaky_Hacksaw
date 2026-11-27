"""MedSAM unsupervised anomaly detection (UAD) of the Subject46 MS lesion on GluCEST, PatchCore-style.

MedSAM is a promptable segmenter, not a generative model, so it cannot "reconstruct" a healthy map. Instead its
frozen ViT image encoder is used as a feature extractor:

  1. Learn healthy: encode every healthy slab slice; keep each 3.75 x 3.75 mm token that lies in the brain ROI
     -> a memory bank of normal embeddings (healthy Scan1, v7 training subjects 0-8, 11, 12, 14-17).
  2. Score: anomaly(token) = mean distance to its k nearest healthy tokens; upsampled to voxels.
     Calibrated on held-out healthy subjects (v7 validation subjects 9, 10, 13): threshold = their p99.
  3. Segment: 3D connected component of above-threshold voxels with the highest peak -> a box per slice ->
     MedSAM's mask decoder draws the lesion.

Evaluated against the semi-automatic Subject46 lesion mask from the v7 project (drawn on the -3 ppm MT image, so the
'mt' input is partly circular). Inputs compared: 'glucest' (GluCEST map), 'mt' (Z at -3 ppm), 'combo' (both + Z +3 ppm).

  python scripts/medsam_uad_ms.py            # all inputs -> outputs/uad_<input>.npz + printed summary
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
GDL = Path('/home/jurgen/Desktop/GluCEST Deep Learning')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(GDL))
from medsam_glucest.model import MedSAM  # noqa: E402
from spatial_glucest_v7 import load_slices, make_lesion_mask  # noqa: E402

OUT = ROOT / 'outputs'
TRAIN = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 14, 15, 16, 17]
VAL = [9, 10, 13]
SLICES = range(1, 11)               # edge slices dropped, as in the paper and v7
I_M3, I_P3 = 4, 13                  # offsets -3.0 and +3.0 ppm
GRID = 64                           # MedSAM embedding grid (1024 / 16)


def load(npz):
    Z, roi, shp, _ = load_slices(str(GDL / npz), denoise=False)
    n, H, W, Zn, _ = (int(v) for v in shp)
    G = np.load(GDL / npz)['glucest'].reshape(n, H, W, Zn).astype(np.float32)
    return dict(Z=Z, roi=roi, G=np.nan_to_num(G))


def channels(d, s, z, mode):
    g, m3, p3 = d['G'][s, :, :, z], d['Z'][s, I_M3, :, :, z], d['Z'][s, I_P3, :, :, z]
    return {'glucest': [g, g, g], 'mt': [m3, m3, m3], 'combo': [g, m3, p3]}[mode]


def windows(d, mode):
    """Fixed per-channel window (healthy training ROI p1-p99), so the same value maps to the same input everywhere."""
    vals = [[] for _ in range(3)]
    for s in TRAIN:
        for z in SLICES:
            r = d['roi'][s, :, :, z]
            for c, im in enumerate(channels(d, s, z, mode)):
                vals[c].append(im[r][::7])
    return [np.percentile(np.concatenate(v), (1, 99)) for v in vals]


def to_input(chs, roi, win):
    x = np.stack([np.clip((c - lo) / max(hi - lo, 1e-6), 0, 1) for c, (lo, hi) in zip(chs, win)], -1)
    return x * roi[..., None]


def token_roi(roi):
    """fraction of each 64x64 token covered by the ROI (slice padded to square like MedSAM.set_rgb)."""
    H, W = roi.shape
    n = max(H, W)
    sq = np.zeros((n, n), np.float32)
    oy, ox = (n - H) // 2, (n - W) // 2
    sq[oy:oy + H, ox:ox + W] = roi
    return F.adaptive_avg_pool2d(torch.tensor(sq)[None, None], GRID)[0, 0].numpy()


def encode(model, d, s, z, mode, win):
    roi = d['roi'][s, :, :, z]
    model.set_rgb(to_input(channels(d, s, z, mode), roi, win), rescale=False)
    return model._emb[0], token_roi(roi)                       # (256, 64, 64), (64, 64)


def tokens_to_voxels(score64, shape):
    H, W = shape
    n = max(H, W)
    oy, ox = (n - H) // 2, (n - W) // 2
    up = F.interpolate(torch.tensor(score64)[None, None], size=(n, n), mode='bilinear', align_corners=False)[0, 0]
    return up.numpy()[oy:oy + H, ox:ox + W]


@torch.no_grad()
def knn_score(q, bank, k=3, chunk=4096):
    out = []
    for i in range(0, len(q), chunk):
        dist = torch.cdist(q[i:i + chunk].float(), bank.float())
        out.append(dist.topk(k, largest=False).values.mean(1))
    return torch.cat(out)


def score_volume(model, d, s, mode, win, bank, k):
    H, W = d['roi'].shape[1:3]
    vol = np.zeros((H, W, d['roi'].shape[3]), np.float32)
    embs = {}
    for z in SLICES:
        e, _ = encode(model, d, s, z, mode, win)
        embs[z] = model._emb
        sc = knn_score(e.reshape(256, -1).T, bank, k).reshape(GRID, GRID).cpu().numpy()
        vol[:, :, z] = tokens_to_voxels(sc, (H, W))
    return vol * d['roi'][s], embs


def run(mode, model, healthy, ms, k=3, min_roi=0.5):
    win = windows(healthy, mode)
    # 1. memory bank of healthy tokens
    bank = []
    for s in TRAIN:
        for z in SLICES:
            e, fr = encode(model, healthy, s, z, mode, win)
            bank.append(e.reshape(256, -1).T[torch.tensor(fr.ravel() >= min_roi, device=e.device)].half())
    bank = torch.cat(bank)
    # 2. calibrate on held-out healthy subjects
    val_scores = []
    for s in VAL:
        v, _ = score_volume(model, healthy, s, mode, win, bank, k)
        val_scores.append(v[healthy['roi'][s]])
    thr = float(np.percentile(np.concatenate(val_scores), 99))
    # 3. score the MS subject
    roi = ms['roi'][0]
    les = make_lesion_mask(ms['Z'][0].transpose(1, 2, 3, 0), roi)
    sc, embs = score_volume(model, ms, 0, mode, win, bank, k)
    zr = np.zeros(roi.shape[2], bool)
    zr[list(SLICES)] = True
    ev = roi & zr[None, None]
    auc = roc_auc_score(les[ev], sc[ev])
    hit = float((sc[les] > thr).mean())
    fp = float((sc[ev & ~les] > thr).mean())
    # 4. MedSAM segments the top component
    above = (sc > thr) & ev
    lab, n = ndi.label(above)
    seg = np.zeros_like(les)
    if n:
        peak = ndi.maximum(sc, lab, index=range(1, n + 1))
        comp = lab == 1 + int(np.argmax(peak))
        for z in np.nonzero(comp.any((0, 1)))[0]:
            r, c = np.nonzero(comp[:, :, z])
            if len(r) < 5:
                continue
            model._emb = embs[z]
            model._img = np.zeros(roi.shape[:2] + (3,))
            n_ = max(roi.shape[:2])
            model._off = ((n_ - roi.shape[0]) // 2, (n_ - roi.shape[1]) // 2)
            seg[:, :, z] = model.predict([c.min() - 1, r.min() - 1, c.max() + 1, r.max() + 1])[0] & roi[:, :, z]
        top_dice = 2 * (comp & les).sum() / max(comp.sum() + les.sum(), 1)
    else:
        comp, top_dice = np.zeros_like(les), 0.0
    dice_seg = 2 * (seg & les).sum() / max(seg.sum() + les.sum(), 1)
    res = dict(mode=mode, auc=auc, lesion_above_p99=hit, healthy_tissue_above_p99=fp, n_components=n,
               top_component_dice=top_dice, medsam_dice=dice_seg, lesion_vox=int(les.sum()),
               top_component_vox=int(comp.sum()), medsam_vox=int(seg.sum()))
    np.savez_compressed(OUT / f'uad_{mode}.npz', score=sc, thr=thr, lesion=les, top=comp, seg=seg, roi=roi,
                        glucest=ms['G'][0], mt=ms['Z'][0, I_M3], **{k_: v for k_, v in res.items() if k_ != 'mode'})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--modes', nargs='*', default=['glucest', 'mt', 'combo'])
    ap.add_argument('--k', type=int, default=3)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    healthy, ms = load('GluCEST_Scan1_B0corrected.npz'), load('GluCEST_Subject46_B0corrected.npz')
    model = MedSAM()
    rows = [run(m, model, healthy, ms, k=a.k) for m in a.modes]
    import pandas as pd
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == '__main__':
    main()
