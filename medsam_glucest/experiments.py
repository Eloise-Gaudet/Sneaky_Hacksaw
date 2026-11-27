"""Extension experiments on top of plain box-prompted MedSAM.

1. structure_eval   subcortical structures on the whole-brain MP2RAGE, box from FSL FIRST, Dice / HD95 vs FIRST
2. box_robustness   same, with increasingly sloppy boxes (each edge moved by up to x % of the box size)
3. propagate_3d     one box on one slice -> whole 3D structure: each next slice is prompted with the box of the
                    previous slice's MedSAM mask (no reference used after the first slice)
4. glucest_cc       GluCEST application: MedSAM refines the warped JHU corpus-callosum atlas on the GluCEST slab's
                    MP2RAGE; GluCEST is then read inside the MedSAM mask and compared with the atlas mask
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from . import data as DT
from .metrics import box_from_mask, dice, hd95, largest_component

DEFAULT_STRUCTS = [10, 49, 11, 50, 12, 51, 17, 53]        # L/R thalamus, caudate, putamen, hippocampus


def best_axial_slice(mask3d):
    return int(np.argmax(mask3d.sum(axis=(0, 1))))


def structure_eval(model, s, labels=DEFAULT_STRUCTS, offsets=(0,)):
    """One row per (structure, slice): the slice of largest area plus optional offsets (in slices)."""
    nv = DT.native(s)
    sp = (nv['spacing'][1], nv['spacing'][0])                  # display rows = A, cols = R
    rows = []
    for lab in labels:
        m3 = nv['first'] == lab
        if not m3.any():
            continue
        k0 = best_axial_slice(m3)
        for off in offsets:
            k = k0 + off
            ref = largest_component(DT.axial(m3, k))
            box = box_from_mask(ref, pad=2)
            if box is None:
                continue
            pred = model.segment(DT.axial(nv['uni'], k), box)[0]
            rows.append(dict(Subject=s, structure=DT.FIRST[lab], label=lab, slice=k, offset=off,
                             dice=dice(pred, ref), hd95_mm=hd95(pred, ref, sp),
                             area_ref_mm2=ref.sum() * sp[0] * sp[1], area_pred_mm2=pred.sum() * sp[0] * sp[1]))
    return pd.DataFrame(rows)


def box_robustness(model, s, label=11, jitters=(0, 0.05, 0.1, 0.2, 0.3), reps=5, seed=0):
    nv = DT.native(s)
    m3 = nv['first'] == label
    k = best_axial_slice(m3)
    ref = largest_component(DT.axial(m3, k))
    model.set_image(DT.axial(nv['uni'], k))                    # one encoder pass, many boxes
    rng = np.random.default_rng(seed)
    rows = []
    for j in jitters:
        for r in range(reps if j else 1):
            box = box_from_mask(ref, pad=2, jitter=j, rng=rng)
            pred = model.predict(box)[0]
            rows.append(dict(Subject=s, structure=DT.FIRST[label], jitter=j, rep=r, dice=dice(pred, ref)))
    return pd.DataFrame(rows)


def propagate_3d(model, s, label=11, pad=3, min_px=15):
    """Single-box 3D segmentation by slice-to-slice propagation; returns (pred3d, ref3d, summary dict)."""
    nv = DT.native(s)
    ref3 = nv['first'] == label
    k0 = best_axial_slice(ref3)
    pred3 = np.zeros_like(ref3)
    n_calls = 0

    def seg(k, box):
        nonlocal n_calls
        n_calls += 1
        m = largest_component(model.segment(DT.axial(nv['uni'], k), box)[0])
        return m if m.sum() >= min_px else None

    m0 = seg(k0, box_from_mask(largest_component(DT.axial(ref3, k0)), pad=2))   # the only reference-based box
    if m0 is not None:
        pred3[:, :, k0] = np.rot90(m0, -1)                    # undo DT.axial
        for step in (+1, -1):
            k, m = k0 + step, m0
            while 0 <= k < ref3.shape[2]:
                m = seg(k, box_from_mask(m, pad=pad))
                if m is None:
                    break
                pred3[:, :, k] = np.rot90(m, -1)
                k += step
    vx = float(np.prod(nv['spacing']))
    summ = dict(Subject=s, structure=DT.FIRST[label], dice3d=dice(pred3, ref3), vol_ref_ml=ref3.sum() * vx / 1000,
                vol_pred_ml=pred3.sum() * vx / 1000, slices_ref=int(ref3.any(axis=(0, 1)).sum()),
                slices_pred=int(pred3.any(axis=(0, 1)).sum()), encoder_calls=n_calls)
    return pred3, ref3, summ


def glucest_cc(model, s, min_px=20):
    """Corpus callosum on the slab MP2RAGE: atlas components -> boxes -> MedSAM, restricted to the slab ROI.
    Returns (per-slice masks dict, summary dict)."""
    sl = DT.slab(s)
    cc = DT.slab_cc(sl)
    valid = DT.valid_glucest(sl)
    med = np.zeros_like(cc)
    masks = {}
    for z in range(1, cc.shape[2] - 1):                        # paper convention: edge slices dropped
        lab, n = ndi.label(cc[:, :, z])
        boxes = [box_from_mask(lab == i, pad=2) for i in range(1, n + 1) if (lab == i).sum() >= min_px]
        if not boxes:
            continue
        pm = model.segment(sl['uni'][:, :, z], boxes)
        med[:, :, z] = pm.any(axis=0) & (sl['roi'][:, :, z] > 0)
        masks[z] = med[:, :, z]
    at = cc.copy()
    at[:, :, [0, -1]] = False
    g = sl['glucest']
    summ = dict(Subject=s, dice_medsam_vs_atlas=dice(med, at), n_vox_atlas=int((at & valid).sum()),
                n_vox_medsam=int((med & valid).sum()),
                glucest_atlas=float(g[at & valid].mean()) if (at & valid).any() else np.nan,
                glucest_medsam=float(g[med & valid].mean()) if (med & valid).any() else np.nan,
                wm_frac_atlas=float((sl['seg'][at] == 3).mean()) if at.any() else np.nan,
                wm_frac_medsam=float((sl['seg'][med] == 3).mean()) if med.any() else np.nan)
    return masks, summ
