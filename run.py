"""Train MedSAM to find MS-like lesions on GluCEST (healthy data + synthetic lesions), then test on the Subject46 lesion.

    conda activate medsam
    python run.py                      # both inputs, one after the other
    python run.py --mode glucest       # GluCEST map only (MTRasym 3 ppm, copied to 3 channels)
    python run.py --mode regular       # GluCEST + Z(-3 ppm) + Z(+3 ppm)
    python run.py --steps 500          # quick run

Outputs per mode in outputs/:  medsam_uad_<mode>.pth (mask-decoder weights, best synthetic-val Dice),
medsam_uad_<mode>.npz (Subject46 + held-out healthy probability maps, metrics), train_uad_<mode>.log.
See medsam_glucest/uad.py for the method.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import SamModel  # noqa: E402

from medsam_glucest import uad as U  # noqa: E402
from medsam_glucest.model import MODEL_ID  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
MODES = {'regular': 'combo', 'glucest': 'glucest'}          # user-facing name -> channel set in uad.py


def dice(a, b):
    return 2 * (a & b).sum() / max(a.sum() + b.sum(), 1)


def batch_from(sampler, n, device):
    xs, ts, rs = zip(*[sampler.sample() for _ in range(n)])
    x = torch.tensor(np.stack(xs)).to(device)
    t = torch.tensor(np.stack(ts)).to(device)
    r = torch.tensor(np.stack(rs)).to(device)
    b = torch.tensor(np.stack([U.brain_box(ri) for ri in rs])).to(device)
    return x, t, r, b


@torch.no_grad()
def validate(model, sampler, device, n=96, batch=8):
    """Dice on synthetic lesions in held-out healthy subjects; false-positive rate on their lesion-free slices."""
    dices, fp = [], []
    for _ in range(n // batch):
        x, t, r, b = batch_from(sampler, batch, device)
        p = (torch.sigmoid(U.forward(model, x, b, train_decoder=False)) > 0.5) & r
        for pi, ti, ri in zip(p.cpu().numpy(), t.cpu().numpy(), r.cpu().numpy()):
            (dices.append(dice(pi, ti)) if ti.any() else fp.append(pi.sum() / ri.sum()))
    return float(np.mean(dices)), float(np.mean(fp)) if fp else float('nan')


def train_and_test(name, data, a, device):
    mode = MODES[name]
    healthy, ms, lesion, win = data
    log = open(OUT / f'train_uad_{name}.log', 'w')

    def say(msg):
        tqdm.write(msg)
        log.write(msg + '\n'); log.flush()

    torch.manual_seed(a.seed)
    model = SamModel.from_pretrained(MODEL_ID).to(device)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.mask_decoder.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(model.mask_decoder.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    tr = U.SynthSampler(healthy, U.TRAIN, win, mode, seed=a.seed)
    va = U.SynthSampler(healthy, U.VAL, win, mode, seed=a.seed + 123)

    ck = OUT / f'medsam_uad_{name}.pth'
    best, hist, ema = -1.0, [], None
    say(f'\n=== {name}: train {len(U.TRAIN)} healthy subjects, validate {len(U.VAL)}, '
        f'{a.steps} steps x batch {a.batch} ===')
    model.mask_decoder.train()
    bar = tqdm(range(1, a.steps + 1), desc=f'{name:8s}', unit='step', dynamic_ncols=True)
    for step in bar:
        x, t, r, b = batch_from(tr, a.batch, device)
        loss = U.loss_fn(U.forward(model, x, b), t, r)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        ema = loss.item() if ema is None else 0.95 * ema + 0.05 * loss.item()
        bar.set_postfix(loss=f'{ema:.3f}', best_val_dice=f'{max(best, 0):.3f}', lr=f'{sched.get_last_lr()[0]:.1e}')
        if step % a.val_every == 0 or step == a.steps:
            model.mask_decoder.eval()
            vd, vfp = validate(model, va, device)
            model.mask_decoder.train()
            hist.append(dict(step=step, loss=ema, val_dice=vd, val_fp=vfp))
            star = ''
            if vd > best:
                best, star = vd, '  * saved'
                torch.save(model.mask_decoder.state_dict(), ck)
            say(f'step {step:5d} | loss {ema:.3f} | val synthetic Dice {vd:.3f} | val clean FP {100 * vfp:.3f} %{star}')
    bar.close()

    # ---- test on Subject46 and on clean held-out healthy subjects
    say('testing on Subject46 ...')
    model.mask_decoder.load_state_dict(torch.load(ck))
    model.eval()
    prob = U.predict_volume(model, ms, 0, win, mode, device)
    hp = [U.predict_volume(model, healthy, s, win, mode, device) for s in tqdm(U.VAL, desc='healthy val', leave=False)]
    roi = ms['roi'][0].copy()
    roi[:, :, [z for z in range(roi.shape[2]) if z not in U.SLICES]] = False
    thr99 = float(np.percentile(np.concatenate([h[healthy['roi'][s]] for h, s in zip(hp, U.VAL)]), 99))
    pred = (prob > 0.5) & roi
    res = dict(input=name, best_val_dice=best,
               auc=float(roc_auc_score(lesion[roi], prob[roi])),
               dice=float(dice(pred, lesion)),
               lesion_detected=float((pred & lesion).sum() / lesion.sum()),
               pred_vox=int(pred.sum()), false_pos_vox=int((pred & ~lesion).sum()),
               healthy_fp_vox_per_subject=float(np.mean([(h > 0.5).sum() for h in hp])),
               lesion_above_healthy_p99=float((prob[lesion] > thr99).mean()))
    say(json.dumps(res, indent=1))
    np.savez_compressed(OUT / f'medsam_uad_{name}.npz', prob=prob, lesion=lesion, roi=ms['roi'][0], ch=ms['ch'][0],
                        healthy_prob=np.stack(hp), healthy_roi=healthy['roi'][U.VAL], healthy_ch=healthy['ch'][U.VAL],
                        win=win, hist=json.dumps(hist), res=json.dumps(res))
    log.close()
    del model; torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', default='both', choices=['both', 'glucest', 'regular'])
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--batch', type=int, default=6)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--val-every', type=int, default=250)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    assert torch.cuda.is_available(), 'needs a GPU (the encoder is run every step)'
    OUT.mkdir(exist_ok=True)

    t0 = time.time()
    print('loading GluCEST data (healthy Scan1 + Subject46) ...', flush=True)
    healthy = U.load_subjects('GluCEST_Scan1_B0corrected.npz')
    ms = U.load_subjects('GluCEST_Subject46_B0corrected.npz')
    lesion = U.subject46_lesion(ms)
    win = U.windows(healthy, U.TRAIN)
    print(f'loaded in {time.time() - t0:.0f} s', flush=True)

    names = ['glucest', 'regular'] if a.mode == 'both' else [a.mode]
    rows = [train_and_test(n, (healthy, ms, lesion, win), a, 'cuda') for n in names]

    import pandas as pd
    print('\n=== Subject46 results (v7 inpainting reference: AUC 0.92) ===')
    print(pd.DataFrame(rows).set_index('input').round(3).T.to_string())


if __name__ == '__main__':
    main()
