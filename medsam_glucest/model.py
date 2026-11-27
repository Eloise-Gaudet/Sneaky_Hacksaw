"""MedSAM (Ma et al., Nat. Commun. 15, 654, 2024): SAM ViT-B fine-tuned on 1.57 M medical image-mask pairs.
Weights: the official checkpoint as ported to Hugging Face (wanglab/medsam-vit-base).

Promptable: give it a bounding box, it returns the object inside. The image encoder (the slow part) runs once per
slice; every extra box only runs the light mask decoder.

Pre-processing follows the official MedSAM inference code (bowang-lab/MedSAM, MedSAM_Inference.py), except that the
slice is zero-padded to a square first (a straight resize would distort non-square MRI slices): resize to 1024 x 1024, min-max scale to [0, 1], no ImageNet normalisation, no padding, box scaled to the 1024 grid, low-res
logits upsampled bilinearly to the slice size. The Hugging Face processor defaults (ImageNet mean/std + padding) are
kept as preprocess='hf' for comparison; they do noticeably worse on MRI (see notebook 01).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from skimage.transform import resize
from transformers import SamModel, SamProcessor

MODEL_ID = 'wanglab/medsam-vit-base'


def to_rgb(img, lo_hi=(0.5, 99.5)):
    """MRI slice -> 3-channel float in [0, 1], intensity window from brain-voxel percentiles."""
    img = np.asarray(img, float)
    vals = img[np.isfinite(img) & (img > 0)]
    lo, hi = np.percentile(vals, lo_hi) if vals.size else (0.0, 1.0)
    x = np.clip((np.nan_to_num(img) - lo) / max(hi - lo, 1e-6), 0, 1)
    return np.repeat(x[..., None], 3, axis=-1)


class MedSAM:
    def __init__(self, model_id=MODEL_ID, device=None, threads=None, preprocess='medsam'):
        if threads:
            torch.set_num_threads(threads)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SamModel.from_pretrained(model_id).to(self.device).eval()
        self.processor = SamProcessor.from_pretrained(model_id)
        self.preprocess = preprocess
        self._img = self._emb = None

    @torch.no_grad()
    def set_image(self, img2d):
        """Run the image encoder once for this slice (cached for all following boxes)."""
        return self.set_rgb(to_rgb(img2d))

    @torch.no_grad()
    def set_rgb(self, rgb, rescale=True):
        """Like set_image, for an already windowed (H, W, 3) image in [0, 1]. rescale=False keeps the intensity
        scale fixed across slices (needed when embeddings are compared between images, e.g. anomaly detection)."""
        self._img = np.asarray(rgb, float)
        if self.preprocess == 'medsam':
            # pad to a square first: a straight 1024 x 1024 resize of a 396 x 208 slice stretches anatomy ~2:1
            H, W = self._img.shape[:2]
            n = max(H, W)
            self._off = ((n - H) // 2, (n - W) // 2)
            sq = np.zeros((n, n, 3))
            sq[self._off[0]:self._off[0] + H, self._off[1]:self._off[1] + W] = self._img
            x = resize(sq, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True)
            x = (x - x.min()) / max(x.max() - x.min(), 1e-8) if rescale else np.clip(x, 0, 1)
            px = torch.tensor(x, dtype=torch.float32).permute(2, 0, 1)[None]
        else:
            px = self.processor((self._img * 255).astype(np.uint8), return_tensors='pt')['pixel_values']
        self._emb = self.model.get_image_embeddings(px.to(self.device))
        return self

    @torch.no_grad()
    def predict(self, boxes, threshold=0.5, return_prob=False):
        """boxes: [x0, y0, x1, y1] or a list of them, in pixel coordinates of the slice (x = column, y = row).
        Returns boolean masks (n_boxes, H, W), or probabilities if return_prob."""
        assert self._emb is not None, 'call set_image first'
        H, W = self._img.shape[:2]
        b = np.atleast_2d(np.asarray(boxes, float))
        if self.preprocess == 'medsam':
            n = max(H, W)
            oy, ox = self._off
            bs = (b + np.array([ox, oy, ox, oy])) / n * 1024
            b1024 = torch.tensor(bs, dtype=torch.float32)[None].to(self.device)
            out = self.model(image_embeddings=self._emb, input_boxes=b1024, multimask_output=False)
            logits = out.pred_masks[0, :, 0]                                     # (n, 256, 256)
            prob = torch.sigmoid(F.interpolate(logits[:, None], size=(n, n), mode='bilinear',
                                               align_corners=False))[:, 0].cpu().numpy()[:, oy:oy + H, ox:ox + W]
        else:
            inp = self.processor((self._img * 255).astype(np.uint8), input_boxes=[b.tolist()],
                                 return_tensors='pt').to(self.device)
            inp.pop('pixel_values')
            out = self.model(image_embeddings=self._emb, input_boxes=inp['input_boxes'], multimask_output=False)
            prob = self.processor.image_processor.post_process_masks(
                out.pred_masks.sigmoid().cpu(), inp['original_sizes'].cpu(), inp['reshaped_input_sizes'].cpu(),
                binarize=False)[0][:, 0].numpy()
        return prob if return_prob else prob > threshold

    def segment(self, img2d, boxes, **kw):
        return self.set_image(img2d).predict(boxes, **kw)
