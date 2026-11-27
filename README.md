# MedSAM on 7T brain MRI: demo + GluCEST extension

Course project (IMgVultr Capital investor pitch). Tool: **MedSAM** (Ma *et al.*, *Nat. Commun.* 15, 654, 2024;
github.com/bowang-lab/MedSAM), the Segment Anything Model fine-tuned on 1.57 M medical image–mask pairs. Draw a box,
get the organ.

Data: our own 7T research scans: whole-brain MP2RAGE and 3D GluCEST from 17 healthy adults (23–73 y). Reference masks:
FSL FIRST (subcortical) and the JHU white-matter atlas warped to each brain.

## Contents

| | |
|---|---|
| `notebooks/01_MedSAM_full_demo.ipynb` | **Full MedSAM demo following the paper** (Ma *et al.* 2024), on public data: the 4 steps of the model (Fig. 2b), pre-processing, EDA of the 10 modalities, MedSAM vs SAM on all 10 (Fig. 1), liver cancer CT / brain cancer MRI / breast cancer US / polyp (Fig. 3), fetal head US, lungs X-ray, dermoscopy, pathology nuclei (Fig. 4), box robustness, box vs point, 3D tumor with sparse markers (Fig. 5b), vessel limitation. Set `N_IMAGES` / `N_SMALL` at the top for a quick or full run |
| `notebooks/01b_MedSAM_brain_MRI_demo.ipynb` | MedSAM on a public brain MRI (ICBM152 T1/T2/PD + Harvard-Oxford): EDA, MedSAM vs SAM, box shifts, 3D; where we found the `wanglab` weight problem |
| `notebooks/02_MedSAM_GluCEST_MS_lesion.ipynb` | **MS lesion UAD**: MedSAM fine-tuned on synthetic lesions in healthy GluCEST, tested on the Subject46 MS lesion (results of `run.py`) |
| `notebooks/03_cohort_evaluation_GluCEST.ipynb` | 17-participant evaluation, box robustness, single-click 3D, GluCEST corpus-callosum application |
| `run.py` | trains + tests the MS-lesion model from the terminal (`python run.py [--mode glucest\|regular]`, tqdm progress) |
| `medsam_glucest/uad.py` | synthetic-lesion training method |
| `medsam_glucest/model.py` | MedSAM wrapper (official pre-processing, encoder cached per slice, multi-box decoding) |
| `medsam_glucest/data.py` | loaders for the native MP2RAGE (RAS) and the GluCEST slab |
| `medsam_glucest/experiments.py` | the four extension experiments |
| `medsam_glucest/metrics.py` | Dice, HD95, box prompts (with jitter), largest component |
| `scripts/run_batch.py` | runs the experiments for all participants → `outputs/*.csv` (resumable) |
| `scripts/build_uad_notebook.py` | regenerates notebook 02 |
| `scripts/build_notebooks.py` | regenerates notebook 03 |
| `tests/test_core.py` | fast checks of the metric / prompt logic |

## Data for notebook 01

Public datasets go in `data/paper_examples/` (not in git); the notebook downloads what is missing. The authors' 10-modality
demo set (`test_demo/`) comes from the MedSAM Google Drive folder:
`gdown --folder https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN -O third_party/MedSAM/work_dir/MedSAM`

## Setup

```bash
python -m venv medsam && medsam\Scripts\activate          # Windows
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m ipykernel install --user --name medsam
python tests/test_core.py                                  # sanity check
python scripts/run_batch.py                                # ~1-2 h on CPU; then run notebook 03
```

The MedSAM weights (375 MB) download from Hugging Face (`wanglab/medsam-vit-base`) on first use. A GPU is not needed:
the image encoder takes ~10 s per slice on a 12-core CPU; each extra box prompt takes milliseconds.

The loaders expect the GluCEST data drive (`GlucestData_withNifit`) and the paper pipeline's cache
(`GluCEST_processing_code/nature_figures_agent3/_cache*`); paths are in `medsam_glucest/data.py`.

## Notes

* **Which MedSAM weights.** `wanglab/medsam-vit-base` on Hugging Face has the original SAM image encoder (every tensor
  identical); only its mask decoder is fine-tuned. `flaviagiammarino/medsam-vit-base` is a full conversion. Notebooks 01/01b
  use the full one (mean DSC 0.76 vs 0.42 with the wanglab upload). `medsam_glucest/model.py`, notebooks 02/03 and
  `run.py` still use the wanglab upload and should be rerun.
* **Participant IDs.** The P-code -> initials map lives in `medsam_glucest/local_ids.json`, which is not in git.

* **Pre-processing.** We follow MedSAM's own inference recipe (percentile window, [0, 1], resize to 1024², no
  ImageNet normalisation) but zero-pad each slice to a square first; the Hugging Face processor's defaults and the
  straight resize both lose accuracy on non-square MRI slices (our 7T slices, notebook 03).
* **Reference masks are not ground truth.** FSL FIRST is a shape model and the JHU atlas is a warped template; Dice
  against them measures agreement, not accuracy.
* **Data sharing.** The images are human research data. Check with the PI / IRB before uploading any of them (or
  derived images) to Canvas; the notebooks can be submitted with outputs only.
