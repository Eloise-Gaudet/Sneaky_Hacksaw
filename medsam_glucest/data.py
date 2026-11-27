"""Data access for the 17 GluCEST participants (one excluded, as in the paper).

native(s): whole-brain 7T MP2RAGE (0.69 x 0.69 x 0.8 mm, HD-BET skull-stripped) with FSL FIRST subcortical labels,
           JHU white-matter labels and FAST tissue classes, all reoriented to RAS so slicing is unambiguous.
slab(s):   the 24 mm GluCEST slab (240 x 196 x 12, 1 x 1 x 2 mm): MP2RAGE resampled into it, JHU atlas, tissue mask,
           ROI mask and the GluCEST map, from the paper pipeline's cache (orientation-checked 2026-11-19).
"""
from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from pathlib import Path

import nibabel as nib
import numpy as np

# Paths. Override with environment variables on any machine:
#   GLUCEST_PROC  folder holding utils_main.py and nature_figures_agent3/_cache*
#   GLUCEST_BASE  .../GlucestData_withNifit/7T_GLUCEST_Repeatability (the data drive)
_PROC_CANDIDATES = [os.environ.get('GLUCEST_PROC'),
                    r'C:/Users/Abeer/OneDrive/Desktop/GluCEST_processing_code',
                    str(Path.home() / 'Desktop' / 'GluCEST_processing_code'),
                    str(Path.home() / 'Desktop' / 'GluCEST-WM-Ageing')]   # anonymized copy (P01-P18)
PROC = next((Path(p) for p in _PROC_CANDIDATES if p and (Path(p) / 'utils_main.py').exists()), None)
if PROC is None:
    raise FileNotFoundError('GluCEST_processing_code (with utils_main.py and nature_figures_agent3/_cache*) not found. '
                            'Copy it to ~/Desktop/GluCEST_processing_code or set GLUCEST_PROC=/path/to/it.')
CACHE = PROC / 'nature_figures_agent3' / '_cache'
CACHE_SEG = PROC / 'nature_figures_agent3' / '_cache_niftiseg'      # NIfTI (FAST) tissue masks
sys.path.insert(0, str(PROC))
import utils_main as U  # noqa: E402  (subject table + data-drive location)

if os.environ.get('GLUCEST_BASE'):
    U.BASE = Path(os.environ['GLUCEST_BASE'])
elif not U.BASE.exists():   # utils_main only searches Windows drive letters; try Linux mount points
    _rel = 'GlucestData_withNifit/7T_GLUCEST_Repeatability'
    U.BASE = next((m / _rel for root in (Path('/media') / os.environ.get('USER', ''), Path('/mnt'))
                   if root.is_dir() for m in root.iterdir() if (m / _rel).exists()), U.BASE)

# Participant initials are kept out of git. medsam_glucest/local_ids.json (gitignored) holds
#   {"initials": {"P01": "..", ...},  "folder_fix": {"P03_S00_rep1": ".."},  "exclude": [".."]}
# initials: P-code -> initials (the GluCEST .mat files are named by initials); folder_fix: the one folder that the
# anonymized utils_main renamed; exclude: the excluded participant's original initials.
_IDS_FILE = Path(__file__).with_name('local_ids.json')
_IDS = json.loads(_IDS_FILE.read_text()) if _IDS_FILE.exists() else {}
INITIALS = _IDS.get('initials', {})
EXCLUDE = {'P18'} | set(_IDS.get('exclude', []))    # excluded as in the paper
ALIAS = {ini: p for p, ini in INITIALS.items()}     # accept the original initials too
FOLDER_FIX = _IDS.get('folder_fix', {})
_ANON = 'P01 – Scan 1' in U.SUBJECT_MAP


def _key(s):
    return ALIAS.get(s, s) if _ANON else s
FIRST = {10: 'L thalamus', 11: 'L caudate', 12: 'L putamen', 13: 'L pallidum', 16: 'Brainstem', 17: 'L hippocampus',
         18: 'L amygdala', 26: 'L accumbens', 49: 'R thalamus', 50: 'R caudate', 51: 'R putamen', 52: 'R pallidum',
         53: 'R hippocampus', 54: 'R amygdala', 58: 'R accumbens'}
CC_LABELS = {'Genu of corpus callosum', 'Body of corpus callosum', 'Splenium of corpus callosum'}   # JHU names


def subjects():
    """[(initials, age)] youngest first."""
    out = [(k.split(' – ')[0], a) for k, (f, a) in U.SUBJECT_MAP.items()]
    return sorted([(s, a) for s, a in out if s not in EXCLUDE], key=lambda x: x[1])


def _native_dir(s):
    folder = U.SUBJECT_MAP[f'{_key(s)} – Scan 1'][0]
    folder = FOLDER_FIX.get(folder, folder) if not (U.BASE / folder).exists() else folder
    hits = sorted((U.BASE / folder).rglob('MP2RAGE_UNI_hdbet_first.nii.gz'))
    if not hits:
        raise FileNotFoundError(f'no FIRST segmentation for {s} under {U.BASE / folder} (is the data drive connected?)')
    # several export folders for some participants: take the one whose FIRST file is populated
    return max(hits, key=lambda h: (np.asarray(nib.load(str(h)).dataobj) > 0).sum()).parent


@lru_cache(maxsize=1)   # whole-brain volumes are ~300 MB per participant
def native(s):
    d = _native_dir(s)

    def ras(name, dtype):
        img = nib.as_closest_canonical(nib.load(str(d / name)))
        return np.asarray(img.dataobj, dtype), img

    uni, img = ras('MP2RAGE_UNI_hdbet.nii.gz', np.float32)
    out = dict(uni=uni, first=ras('MP2RAGE_UNI_hdbet_first.nii.gz', np.int16)[0],
               fast=ras('MP2RAGE_UNI_hdbet_fast3_seg.nii.gz', np.int16)[0], spacing=img.header.get_zooms()[:3],
               folder=str(d))
    if (d / 'MP2RAGE_UNI_hdbet_WMrois.nii.gz').exists():
        out['jhu'] = ras('MP2RAGE_UNI_hdbet_WMrois.nii.gz', np.int16)[0]
    return out


# RAS volume -> 2D display slice, superior/anterior at the top
def axial(v, k):
    return np.rot90(v[:, :, k])


def coronal(v, j):
    return np.rot90(v[:, j, :])


def sagittal(v, i):
    return np.rot90(v[i, :, :])


MAT_DIR = Path(os.environ.get('GLUCEST_MAT_DIR', U.BASE.parents[1] / 'GluCEST_CESTCORRECTEDMTR_data'))


def slab(s):
    if (CACHE / f'sub_{s}.npz').exists():
        sb = np.load(CACHE / f'sub_{s}.npz')
        vl = np.load(CACHE_SEG / f'vol_{s}.npz')
        lbl = {int(k): v for k, v in __import__('json').loads((CACHE / f'labels_{s}.json').read_text())['WM'].items()}
        return dict(uni=sb['uni'], wm_atlas=sb['wm_atlas'].astype(int), seg=vl['seg'], roi=vl['roi'],
                    glucest=vl['CESTmap'].astype(float), labels=lbl)
    return slab_from_source(s)


def slab_from_source(s):
    """Build the slab without the paper cache: CESTslab_* NIfTIs (MP2RAGE, JHU atlas, FAST seg) + the GluCEST .mat
    (CESTmap, disproimask ROI), all in the .mat frame (240 x 196 x 12). NIfTI -> .mat frame is transpose(1,0,2) plus
    the A/P flip in utils_main.atlas_to_mat_frame; checked on P03, whose cest_map.nii matches the .mat CESTmap (r = 1.0)."""
    folder = U.SUBJECT_MAP[f'{_key(s)} – Scan 1'][0]
    folder = FOLDER_FIX.get(folder, folder) if not (U.BASE / folder).exists() else folder
    hits = sorted((U.BASE / folder).rglob('CESTslab_WMrois.nii'))
    if not hits:
        raise FileNotFoundError(f'no CESTslab_WMrois.nii for {s} under {U.BASE / folder}')
    d = next((h.parent for h in hits if h.parent.name == 'nifti'), hits[0].parent)   # not XNIFTI/ (older export)

    def nii(name, dtype):
        # later scans (P07, P17) name the slab files CESTslabX*; the JHU atlas keeps the CESTslab_ name
        f = d / name if (d / name).exists() else d / name.replace('CESTslab', 'CESTslabX', 1)
        v = np.asarray(nib.load(str(f)).dataobj, dtype)
        if v.ndim != 3:
            raise ValueError(f'{f} has shape {v.shape}, expected a 3D slab')
        return np.transpose(v[:, ::-1], (1, 0, 2))

    labels = U._roi_labels(d, 'WM') or {int(a): b.strip() for a, b in (ln.split(' ', 1) for ln in
              (d / 'CESTslabX_WMrois.txt').read_text(errors='replace').splitlines() if ' ' in ln.strip())}

    ini = INITIALS.get(_key(s), s)
    out = U.load_scan(f'CEST3DNEW_prep_tfl_3D_glucest_GLUCEST_{ini}_01_SEGMENTED.mat', MAT_DIR)
    return dict(uni=nii('CESTslab.nii', np.float32), wm_atlas=nii('CESTslab_WMrois.nii', np.float64).round().astype(int),
                seg=nii('CESTslab_seg.nii', np.float64).round().astype(int),
                roi=np.asarray(out.disproimask, float), glucest=np.asarray(out.CESTmap, float),
                labels=labels)


def slab_cc(sl):
    """corpus callosum (genu + body + splenium) from the JHU atlas in the slab."""
    keys = [k for k, n in sl['labels'].items() if n in CC_LABELS]
    return np.isin(sl['wm_atlas'], keys)


def valid_glucest(sl):
    c = sl['glucest']
    return np.isfinite(c) & (sl['roi'] > 0) & (c > -10) & (c < 30) & (c != 0)
