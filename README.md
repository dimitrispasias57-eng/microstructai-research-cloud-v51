# MicroStructAI Research Cloud v5.1-rc1 (integration candidate)

Separate unified Research Use Only integration candidate. The working
MicroStructAI Native RBC Cloud v1.1 repository and deployment are not replaced.
The stable v5.0 application is also unchanged. This candidate is not yet certified
for hosted release; local synthetic passes are not hosted acceptance.

## Modules

- Native RBC morphology: unchanged v1.1 U-Net/refinement/QC; centred native
  96×96 grayscale DPC crops only; every mask requires human approval.
- RBC aggregation/spatial structure: preserved classical RC2 gap engine,
  QC, SASI, GapScore, PTI and AggIndex. Exploratory, not rheologically validated.
- Temporal analysis: image-sequence metric handoff and strict timestamp/CSV
  analysis. Event/baseline/timing qualifications remain explicit.
- Dynamic morphology: supplied reviewed same-cell masks or approved native
  observations. Identity must be documented. Not membrane stiffness.
- General structure: original grayscale/RGB images, explicit channel/target,
  optional ROI/illumination correction, experimental proposals and review;
  connected-region, topology, network, thickness, spectral descriptors and PCA.
- Reference comparison: supplied binary masks at identical pixel geometry.
- Session-local CSV, figures, HTML reports and ZIP provenance exports.
- Supplied-instance morphology, explicit separate-cell stacks and nonoverlapping
  field summaries; legacy perimeter conventions remain separately labelled.
- Clot phase proposals with context/review gates; no validated fibrin recognition.
- Structural detail, perturbation checks, supplied XML tracks, device-distance
  analysis and descriptive/reference-table comparison.
- Sealed historical classical proposals with explicit accept/reject/replacement.
- Standalone PCA, stored multi-sample comparisons and bounded upload-only batches.
- Historical Object Recording: four unchanged historical methods and their
  original controls, optical/POD results, review and exports; a separate
  measurement contract from the main General Structural Analysis route.
- Restored RBC mask/POD, optical/mask and automatic-start diagnostics, paired
  reference plots and independent-recording summary figures.
- A visible Advanced M2 evidence page; no live external-model calls.

Scientific source copies and weights are checked against VENDOR_MANIFEST.json.
Additional exact historical helpers are checked against HELPER_MANIFEST.json.
No training, model substitution or external inference is performed.

## Run

Python 3.13, `pip install -r requirements.txt`, `streamlit run app.py`.
Linux installs CPU-only PyTorch. No API credentials or paid services are needed.
Use a NEW Streamlit Community Cloud application and URL. Do not edit the native
v1.1 deployment. Hosted acceptance is recorded outside this public app package.

## Limits and privacy

Uploaded images and generated results stay in session memory, not a shared
application output directory. Only the read-only model is cached. Hosted upload
necessarily sends files to Streamlit. This is not a confidential clinical-data
platform. Use only authorised, non-identifiable research data. Export before
refresh, sleep, disconnect or clearing a session.
Do not upload protected health information (PHI), patient identifiers or
confidential datasets. This free research app is not a clinical-data service.

Classical uploads: maximum 200 files, 8 MiB each, 32 MiB total, 1 million spatial
pixels per image and 16 million channel-pixels per run. Native: maximum 50 crops,
1 MiB each and 16 MiB total. No silent resizing/thinning. Larger complete
recordings remain LOCAL/ADVANCED ONLY. Direct video-container upload remains an
explicit integration gap; study-specific extraction scripts are not a supported
generic uploader and are not claimed intrinsically cloud-incompatible.
Do not split one temporal recording merely to evade limits.

All numerical results are research descriptors. Connected components do not
establish cell identity; repeated observations are not independent cells.
No diagnosis, clinical accuracy, aggregation diagnosis, RheoScan equivalence,
yield stress, membrane stiffness or mechanical deformability is claimed.

## Attribution and licensing

Native training source: erythroSight, Shrestha et al. (2024), FRDR DOI
[10.20383/103.0916](https://doi.org/10.20383/103.0916), CC BY 4.0.
Reference-outline rasterisation, centred cropping, train-only augmentation and
from-scratch model training were performed. References were published Cellpose
outlines, not manual ground truth. No Cellpose weights or Astra outputs were used
for training. Dataset authors do not endorse the app. No raw biological images,
participant mappings, reviewer identities or credentials are distributed.

MicroStructAI code is the project owner's existing research software. This
candidate does not invent a new open-source licence grant for historical code.
Dependencies retain their own licences. PyTorch, NumPy, SciPy, scikit-image,
pandas, scikit-learn, matplotlib, Pillow, Streamlit and tifffile are installed
from their normal distributions, with upstream notices intact.
