"""Fast checks of the non-model logic:  python tests/test_core.py   (or pytest)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medsam_glucest.metrics import box_from_mask, dice, hd95, largest_component  # noqa: E402


def test_dice():
    a = np.zeros((10, 10), bool); a[2:6, 2:6] = True
    b = np.zeros((10, 10), bool); b[4:8, 2:6] = True
    assert dice(a, a) == 1.0
    assert abs(dice(a, b) - 0.5) < 1e-9                      # 8 shared of 16 + 16
    assert dice(np.zeros(4, bool), np.zeros(4, bool)) == 1.0


def test_hd95():
    a = np.zeros((40, 40), bool); a[10:20, 10:20] = True
    b = np.roll(a, 3, axis=1)
    assert hd95(a, a) == 0.0
    assert abs(hd95(a, b) - 3.0) < 1e-9                      # shifted by 3 px
    assert abs(hd95(a, b, spacing=(1.0, 0.5)) - 1.5) < 1e-9  # anisotropic pixels
    assert np.isnan(hd95(a, np.zeros_like(a)))


def test_box():
    m = np.zeros((50, 60), bool); m[10:20, 30:41] = True
    assert box_from_mask(m, pad=0) == [30.0, 10.0, 40.0, 19.0]          # [x0, y0, x1, y1] = cols, rows
    assert box_from_mask(m, pad=100) == [0.0, 0.0, 59.0, 49.0]          # clipped to the image
    j = box_from_mask(m, pad=0, jitter=0.2, rng=np.random.default_rng(1))
    assert j != [30.0, 10.0, 40.0, 19.0] and j[0] < j[2] and j[1] < j[3]
    assert box_from_mask(np.zeros((5, 5), bool)) is None


def test_largest_component():
    m = np.zeros((20, 20), bool); m[1:3, 1:3] = True; m[10:18, 10:18] = True
    out = largest_component(m)
    assert out.sum() == 64 and not out[1, 1]


if __name__ == '__main__':
    for f in [test_dice, test_hd95, test_box, test_largest_component]:
        f()
    print('all tests passed')
