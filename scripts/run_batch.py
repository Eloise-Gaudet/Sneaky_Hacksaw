"""Run the extension experiments on every participant; one CSV per experiment in outputs/, resumable.

  python scripts/run_batch.py                       # all 17 participants
  python scripts/run_batch.py --subjects JX KM      # a subset
  python scripts/run_batch.py --propagate 6         # 3D propagation for 6 participants spread over the age range
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from medsam_glucest import data as DT  # noqa: E402
from medsam_glucest import experiments as X  # noqa: E402
from medsam_glucest.model import MedSAM  # noqa: E402

OUT = ROOT / 'outputs'


def append(name, df):
    p = OUT / f'{name}.csv'
    df.to_csv(p, mode='a', header=not p.exists(), index=False)


def done(name, s):
    p = OUT / f'{name}.csv'
    return p.exists() and s in set(pd.read_csv(p).Subject)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subjects', nargs='*')
    ap.add_argument('--propagate', type=int, default=6)
    ap.add_argument('--threads', type=int, default=12)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    subs = [(s, age) for s, age in DT.subjects() if not a.subjects or s in a.subjects]
    prop = {subs[i][0] for i in np.unique(np.linspace(0, len(subs) - 1, a.propagate).round().astype(int))} \
        if a.propagate else set()
    model = MedSAM(threads=a.threads)
    for s, age in subs:
        t = time.time()
        if not done('structures', s):
            append('structures', X.structure_eval(model, s).assign(Age=age))
        if not done('box_robustness', s):
            append('box_robustness', pd.concat([X.box_robustness(model, s, lab) for lab in (11, 49)]).assign(Age=age))
        if not done('glucest_cc', s):
            append('glucest_cc', pd.DataFrame([X.glucest_cc(model, s)[1]]).assign(Age=age))
        if s in prop and not done('propagation', s):
            pred3, ref3, summ = X.propagate_3d(model, s, label=11)
            append('propagation', pd.DataFrame([summ]).assign(Age=age))
            np.savez_compressed(OUT / f'prop_{s}_Lcaudate.npz', pred=pred3, ref=ref3)
        DT.native.cache_clear()
        print(f'{s} ({age} y) done in {time.time() - t:.0f} s', flush=True)


if __name__ == '__main__':
    main()
