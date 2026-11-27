"""Overlap / boundary metrics and box prompts."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def dice(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2 * (a & b).sum() / s


def hd95(a, b, spacing=(1.0, 1.0)):
    """95th-percentile symmetric surface distance (mm); nan if either mask is empty."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    if not a.any() or not b.any():
        return np.nan
    sa = a & ~ndi.binary_erosion(a)
    sb = b & ~ndi.binary_erosion(b)
    da = ndi.distance_transform_edt(~sb, sampling=spacing)[sa]
    db = ndi.distance_transform_edt(~sa, sampling=spacing)[sb]
    return float(np.percentile(np.concatenate([da, db]), 95))


def box_from_mask(mask, pad=2, jitter=0.0, rng=None):
    """Tight bounding box [x0, y0, x1, y1] around a 2D mask, padded by `pad` pixels.
    jitter > 0 moves each edge by a random fraction (up to `jitter`) of the box size: a sloppy user box."""
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return None
    x0, x1, y0, y1 = cols.min() - pad, cols.max() + pad, rows.min() - pad, rows.max() + pad
    if jitter:
        rng = rng or np.random.default_rng(0)
        w, h = x1 - x0, y1 - y0
        x0, x1 = x0 + rng.uniform(-jitter, jitter) * w, x1 + rng.uniform(-jitter, jitter) * w
        y0, y1 = y0 + rng.uniform(-jitter, jitter) * h, y1 + rng.uniform(-jitter, jitter) * h
    H, W = mask.shape
    x0, x1 = np.clip(sorted((x0, x1)), 0, W - 1)
    y0, y1 = np.clip(sorted((y0, y1)), 0, H - 1)
    return [float(x0), float(y0), float(x1), float(y1)]


def largest_component(m):
    lab, n = ndi.label(m)
    if n <= 1:
        return m
    return lab == (np.argmax(ndi.sum(m, lab, range(1, n + 1))) + 1)
