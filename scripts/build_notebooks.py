"""Write notebooks/03_cohort_evaluation_GluCEST.ipynb (then execute with nbconvert). Notebook 01: scripts/build_demo_notebook.py; notebook 02: scripts/build_uad_notebook.py."""
from pathlib import Path

import nbformat as nbf

NB = Path(__file__).resolve().parents[1] / 'notebooks'
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

SETUP = '''import sys, time, os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
sys.path.insert(0, '..')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from medsam_glucest import data as DT
from medsam_glucest.model import MedSAM, to_rgb
from medsam_glucest.metrics import box_from_mask, dice, hd95, largest_component
from medsam_glucest.experiments import best_axial_slice
plt.rcParams.update({'figure.dpi': 110, 'font.size': 9, 'axes.titlesize': 9})

def show_box(ax, box, **kw):
    x0, y0, x1, y1 = box
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=kw.get('c', 'yellow'), lw=kw.get('lw', 1.2)))

def crop_to(m, pad=25):
    r, c = np.nonzero(m)
    return slice(max(r.min() - pad, 0), r.max() + pad), slice(max(c.min() - pad, 0), c.max() + pad)'''

nb2 = nbf.v4.new_notebook()
nb2.cells = [
    md('''# Extension: MedSAM at cohort scale, and for GluCEST

Built on the `medsam_glucest` package (see README). The heavy compute is done by `scripts/run_batch.py`
(~1–2 h on CPU for 17 participants); this notebook loads its results from `outputs/` and makes the figures.

1. **Cohort evaluation**: 8 subcortical structures × 17 participants vs FSL FIRST
2. **Robustness**: how sloppy can the user's box be?
3. **Single-click 3D**: one box on one slice → whole structure, by propagating MedSAM's own masks slice to slice
4. **GluCEST application**: MedSAM refines the warped JHU corpus-callosum atlas on the GluCEST slab; does the GluCEST
   value, and its age trend, change?'''),
    code(SETUP + '''
from scipy import stats
OUT = __import__('pathlib').Path('../outputs')
st = pd.read_csv(OUT / 'structures.csv'); rb = pd.read_csv(OUT / 'box_robustness.csv')
pr = pd.read_csv(OUT / 'propagation.csv')
gc = pd.read_csv(OUT / 'glucest_cc.csv') if (OUT / 'glucest_cc.csv').exists() else None   # needs the GluCEST slab cache
print('participants:', st.Subject.nunique(), '| structure evaluations:', len(st))'''),
    md('## 1. Cohort evaluation'),
    code('''st['side'] = st.structure.str[0]; st['struct'] = st.structure.str[2:]
summ = st.groupby('struct').agg(dice_mean=('dice', 'mean'), dice_sd=('dice', 'std'), hd95_mm=('hd95_mm', 'median'),
                                pred_over_ref_area=('area_pred_mm2', 'sum')).round(3)
summ['pred_over_ref_area'] = (st.groupby('struct').area_pred_mm2.sum() / st.groupby('struct').area_ref_mm2.sum()).round(2)
display(summ.sort_values('dice_mean', ascending=False))
order = summ.sort_values('dice_mean', ascending=False).index
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for i, s_ in enumerate(order):
    v = st[st.struct == s_].dice
    ax[0].scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(v)), v, s=10, alpha=0.6)
    ax[0].plot([i - 0.25, i + 0.25], [v.median()] * 2, 'k', lw=2)
ax[0].set_xticks(range(len(order)), order); ax[0].set_ylabel('Dice vs FSL FIRST'); ax[0].set_ylim(0, 1)
ax[0].set_title('Per structure (17 participants x 2 sides)')
pm = st.groupby(['Subject', 'Age']).dice.mean().reset_index()
r, p = stats.pearsonr(pm.Age, pm.dice)
ax[1].scatter(pm.Age, pm.dice); ax[1].set_xlabel('Age (years)'); ax[1].set_ylabel('mean Dice')
ax[1].set_title(f'Does performance depend on age? r = {r:.2f}, P = {p:.2f}')
plt.tight_layout(); plt.show()'''),
    md('## 2. Robustness to the box\nEach box edge is moved randomly by up to x % of the box size (5 draws per level), '
       'for the left caudate (high contrast) and right thalamus (low contrast).'),
    code('''g = rb.groupby(['structure', 'jitter']).dice.agg(['mean', 'sem']).reset_index()
fig, ax = plt.subplots(figsize=(5, 3.6))
for s_, d in g.groupby('structure'):
    ax.errorbar(100 * d.jitter, d['mean'], yerr=d['sem'], marker='o', capsize=2, label=s_)
ax.set_xlabel('box perturbation (% of box size)'); ax.set_ylabel('Dice vs FSL FIRST'); ax.legend(); ax.set_ylim(0, 1)
ax.set_title('How precise must the user be?'); plt.tight_layout(); plt.show()
display(g.pivot(index='jitter', columns='structure', values='mean').round(3))'''),
    md('## 3. Single-click 3D\nOne box on the slice where the left caudate is largest; every other slice is prompted '
       'with the box of MedSAM\'s own mask on the neighbouring slice. No reference is used after the first click.'),
    code('''display(pr[['Subject', 'Age', 'dice3d', 'vol_ref_ml', 'vol_pred_ml', 'slices_ref', 'slices_pred', 'encoder_calls']].round(3))
s0 = pr.sort_values('dice3d').iloc[len(pr) // 2].Subject              # median participant
z = np.load(OUT / f'prop_{s0}_Lcaudate.npz'); pred3, ref3 = z['pred'], z['ref']
nv = DT.native(s0)
jc = int(np.argmax(ref3.sum(axis=(0, 2)))); ic = int(np.argmax(ref3.sum(axis=(1, 2))))
fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))
for a, img, rf, pd_, t in [(ax[0], DT.coronal(nv['uni'], jc), DT.coronal(ref3, jc), DT.coronal(pred3, jc), 'coronal'),
                           (ax[1], DT.sagittal(nv['uni'], ic), DT.sagittal(ref3, ic), DT.sagittal(pred3, ic), 'sagittal')]:
    r, c = crop_to(rf | pd_, 25)
    a.imshow(img[r, c], cmap='gray'); a.contour(rf[r, c], [0.5], colors='lime'); a.contour(pd_[r, c], [0.5], colors='red')
    a.set_title(f'{s0}: left caudate, {t} view of the 3D result'); a.axis('off')
plt.tight_layout(); plt.show()'''),
    md('''## 4. GluCEST application: the corpus callosum
In the GluCEST study, white-matter regions come from the JHU atlas warped to each brain, which is coarse at the 1 mm
slab resolution. Here each connected piece of the atlas CC on each slab slice becomes a MedSAM box prompt on the slab's
MP2RAGE; GluCEST is then averaged inside MedSAM's mask instead of the atlas mask.'''),
    code('''display(gc[['Subject', 'Age', 'dice_medsam_vs_atlas', 'n_vox_atlas', 'n_vox_medsam', 'wm_frac_atlas', 'wm_frac_medsam',
           'glucest_atlas', 'glucest_medsam']].round(3))
t_ = stats.ttest_rel(gc.glucest_medsam, gc.glucest_atlas)
print(f"MedSAM vs atlas mask: Dice {gc.dice_medsam_vs_atlas.mean():.2f}; WM fraction {gc.wm_frac_atlas.mean():.2f} -> {gc.wm_frac_medsam.mean():.2f};",
      f"GluCEST {gc.glucest_atlas.mean():.2f} -> {gc.glucest_medsam.mean():.2f} % (paired P = {t_.pvalue:.3g})")
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].scatter(gc.glucest_atlas, gc.glucest_medsam); lim = [gc[['glucest_atlas', 'glucest_medsam']].min().min() - 0.3, gc[['glucest_atlas', 'glucest_medsam']].max().max() + 0.3]
ax[0].plot(lim, lim, 'k:'); ax[0].set_xlabel('GluCEST, atlas CC (%)'); ax[0].set_ylabel('GluCEST, MedSAM CC (%)'); ax[0].set_title('Per participant')
for col, lab in [('glucest_atlas', 'atlas'), ('glucest_medsam', 'MedSAM')]:
    r, p = stats.pearsonr(gc.Age, gc[col]); b = np.polyfit(gc.Age, gc[col], 1)
    ax[1].scatter(gc.Age, gc[col], s=14); ax[1].plot(gc.Age.sort_values(), np.polyval(b, gc.Age.sort_values()),
                                               label=f'{lab}: {10 * b[0]:+.2f} %/decade, r = {r:.2f}, P = {p:.2f}')
ax[1].set_xlabel('Age (years)'); ax[1].set_ylabel('CC GluCEST (%)'); ax[1].legend(fontsize=7); ax[1].set_title('Age trend')
plt.tight_layout(); plt.show()'''),
    code('''# what the refinement looks like on one slab slice
from medsam_glucest.experiments import glucest_cc
model = MedSAM(threads=8)
s1 = gc.sort_values('Age').iloc[len(gc) // 2].Subject
masks, _ = glucest_cc(model, s1)
sl = DT.slab(s1); cc = DT.slab_cc(sl)
z = max(masks, key=lambda q: masks[q].sum())
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].imshow(sl['uni'][:, :, z], cmap='gray'); ax[0].contour(cc[:, :, z], [0.5], colors='lime'); ax[0].set_title(f'{s1} slab slice {z}: JHU atlas CC')
ax[1].imshow(sl['uni'][:, :, z], cmap='gray'); ax[1].contour(masks[z], [0.5], colors='red'); ax[1].set_title('MedSAM CC')
g_ = np.where(DT.valid_glucest(sl)[:, :, z], sl['glucest'][:, :, z], np.nan)
ax[2].imshow(sl['uni'][:, :, z], cmap='gray'); ax[2].imshow(np.where(masks[z], g_, np.nan), cmap='magma', vmin=0, vmax=12)
ax[2].set_title('GluCEST inside the MedSAM CC')
for a in ax: a.axis('off')
plt.tight_layout(); plt.show()'''),
    md('''## Take-aways for the pitch
* **Scale**: one generic model, no training, segments 8 structures in 17 brains from boxes alone.
* **Robust**: performance holds up when the box is drawn sloppily (see section 2 for where it breaks).
* **Single click → 3D**: one box gives a whole structure in 3D.
* **Plugs into real research**: MedSAM can refine atlas-derived ROIs for quantitative imaging such as GluCEST.
* **Limits**: driven by visible contrast; low-contrast boundaries (thalamus / putamen on MP2RAGE) are under-segmented;
  2D model (3D only by propagation); CPU inference ~10 s per slice.'''),
]

if __name__ == '__main__':
    nbf.write(nb2, NB / '03_cohort_evaluation_GluCEST.ipynb')
    print('notebooks written')
