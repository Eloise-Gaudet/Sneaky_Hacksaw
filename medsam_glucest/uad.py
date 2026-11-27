"""Fine-tune MedSAM to find MS-like lesions on GluCEST without lesion labels (synthetic-anomaly UAD).

Healthy GluCEST slabs only. Synthetic lesions are painted into healthy white matter: inside a random smooth blob, the
three input channels are shifted by alpha x (this subject's GM mean - WM mean), alpha ~ U(0.5, 2.0). This mimics the
Subject46 lesion, which is spectrally GM-like (beyond GM: Z(-3 ppm) 0.57 vs GM 0.50 vs WM 0.38). MedSAM's mask decoder
is trained to outline the blobs given one box around the whole brain, and to output nothing on lesion-free slices.
The image encoder stays frozen (MedSAM's pretrained features).

Channels (from SVD-denoised, B0-corrected Z-spectra; same data as the v7 inpainting model):
  0  GluCEST contrast = MTRasym(3 ppm) = Z(-3) - Z(+3)
  1  Z(-3 ppm)  (MT-weighted)
  2  Z(+3 ppm)
mode 'glucest' feeds channel 0 only (copied to 3 channels).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi

GDL = Path('/home/jurgen/Desktop/GluCEST Deep Learning')
sys.path.insert(0, str(GDL))
from spatial_glucest_v7 import load_slices, make_lesion_mask  # noqa: E402

TRAIN = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 14, 15, 16, 17]     # v7 split (subject level)
VAL = [9, 10, 13]
SLICES = list(range(1, 11))                                       # edge slices dropped
I_M3, I_P3 = 4, 13
H, W, N = 240, 196, 240                                           # slab slice, padded square side
OX = (N - W) // 2


def load_subjects(npz):
    """-> dict(ch (n, 3, H, W, Z) float32, roi (n, H, W, Z) bool, seg (n, H, W, Z) int, Zraw)"""
    Z, roi, shp, _ = load_slices(str(GDL / npz), denoise=True)
    n = int(shp[0])
    seg = np.load(GDL / npz)['segmask'].reshape(tuple(int(v) for v in shp[:4])).astype(int)
    m3, p3 = Z[:, I_M3], Z[:, I_P3]
    ch = np.stack([m3 - p3, m3, p3], 1).astype(np.float32) * roi[:, None]
    return dict(ch=ch, roi=roi, seg=seg * roi, n=n, npz=npz)


def windows(d, subs):
    """fixed per-channel window (p1-p99 of healthy training ROI voxels)"""
    out = []
    for c in range(3):
        v = np.concatenate([d['ch'][s, c][:, :, SLICES][d['roi'][s][:, :, SLICES]][::5] for s in subs])
        out.append(np.percentile(v, (1, 99)))
    return np.array(out, np.float32)


def lesion_shift(d, s):
    """per-subject channel shift GM - WM (what a GM-like lesion in WM looks like)"""
    gm, wm = d['seg'][s] == 2, d['seg'][s] == 3
    return np.array([d['ch'][s, c][gm].mean() - d['ch'][s, c][wm].mean() for c in range(3)], np.float32)


def random_blob(rng, wm, roi):
    """smooth irregular blob centred in WM; returns weight map w in [0, 1]"""
    cand = np.argwhere(ndi.binary_erosion(wm, iterations=3))
    if len(cand) == 0:
        return None
    cy, cx = cand[rng.integers(len(cand))]
    ry, rx = rng.uniform(2.5, 9, 2)
    th = rng.uniform(0, np.pi)
    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    dy, dx = yy - cy, xx - cx
    u, v = dy * np.cos(th) + dx * np.sin(th), -dy * np.sin(th) + dx * np.cos(th)
    r = (u / ry) ** 2 + (v / rx) ** 2
    noise = ndi.gaussian_filter(rng.standard_normal((H, W)).astype(np.float32), 3) * 6
    w = ((r + noise * 0.15) < 1).astype(np.float32)
    w = ndi.gaussian_filter(w, rng.uniform(0.6, 1.5))
    return w * roi


class SynthSampler:
    def __init__(self, d, subs, win, mode='combo', p_neg=0.25, seed=0):
        self.d, self.subs, self.win, self.mode, self.p_neg = d, subs, win, mode, p_neg
        self.rng = np.random.default_rng(seed)
        self.shift = {s: lesion_shift(d, s) for s in subs}

    def sample(self):
        rng, d = self.rng, self.d
        s, z = self.subs[rng.integers(len(self.subs))], SLICES[rng.integers(len(SLICES))]
        ch, roi, wm = d['ch'][s, :, :, :, z].copy(), d['roi'][s, :, :, z], d['seg'][s, :, :, z] == 3
        target = np.zeros((H, W), bool)
        if rng.random() > self.p_neg:
            for _ in range(rng.integers(1, 3)):
                w = random_blob(rng, wm, roi)
                if w is None:
                    break
                ch += rng.uniform(0.5, 2.0) * w[None] * self.shift[s][:, None, None]
                target |= w > 0.5
        if rng.random() < 0.5:                                    # left-right flip
            ch, roi, target = ch[:, :, ::-1], roi[:, ::-1], target[:, ::-1]
        return to_input(ch, roi, self.win, self.mode), np.ascontiguousarray(target), np.ascontiguousarray(roi)


def to_input(ch, roi, win, mode):
    x = np.clip((ch - win[:, 0, None, None]) / (win[:, 1] - win[:, 0])[:, None, None], 0, 1) * roi[None]
    if mode == 'glucest':
        x = np.repeat(x[:1], 3, 0)
    return x.astype(np.float32)


def brain_box(roi):
    r, c = np.nonzero(roi)
    b = np.array([c.min() - 2 + OX, r.min() - 2, c.max() + 2 + OX, r.max() + 2], np.float32)
    return b / N * 1024


def pad_resize(x, size=1024):
    """(B, C, H, W) -> padded square (B, C, N, N) -> (B, C, size, size)"""
    x = F.pad(x, (OX, N - W - OX, 0, 0))
    return F.interpolate(x, size=(size, size), mode='bilinear', align_corners=False)


def logits_to_slice(logits):
    """(B, 256, 256) low-res logits -> (B, H, W) logits on the slab grid"""
    up = F.interpolate(logits[:, None], size=(N, N), mode='bilinear', align_corners=False)[:, 0]
    return up[:, :, OX:OX + W]


def forward(model, x, boxes, train_decoder=True):
    """x (B, 3, H, W) in [0, 1] on device; boxes (B, 4) in 1024 coords -> (B, H, W) logits"""
    model.vision_encoder.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):   # one image at a time: 8 GB GPU
        emb = torch.cat([model.get_image_embeddings(pad_resize(x[i:i + 1])) for i in range(len(x))])
    with torch.set_grad_enabled(train_decoder):
        out = model(image_embeddings=emb.float(), input_boxes=boxes[:, None], multimask_output=False)
    return logits_to_slice(out.pred_masks[:, 0, 0])


def loss_fn(logits, target, roi):
    t = target.float()
    bce = F.binary_cross_entropy_with_logits(logits, t, reduction='none')[roi].mean()
    p = torch.sigmoid(logits) * roi
    inter = (p * t).sum((1, 2))
    dice = 1 - (2 * inter + 1) / (p.sum((1, 2)) + t.sum((1, 2)) + 1)
    return bce + dice.mean()


@torch.no_grad()
def predict_volume(model, d, s, win, mode, device):
    """probability volume (H, W, Z) for subject s"""
    prob = np.zeros((H, W, d['roi'].shape[3]), np.float32)
    for z in SLICES:
        roi = d['roi'][s, :, :, z]
        x = torch.tensor(to_input(d['ch'][s, :, :, :, z], roi, win, mode))[None].to(device)
        b = torch.tensor(brain_box(roi))[None].to(device)
        prob[:, :, z] = torch.sigmoid(forward(model, x, b, train_decoder=False))[0].cpu().numpy() * roi
    return prob


def subject46_lesion(d46):
    Z, roi, _, _ = load_slices(str(GDL / 'GluCEST_Subject46_B0corrected.npz'), denoise=False)
    return make_lesion_mask(Z[0].transpose(1, 2, 3, 0), roi[0])
