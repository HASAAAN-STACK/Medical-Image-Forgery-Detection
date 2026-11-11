# Medical Image Forgery Detection and Localization

Research prototype following the supplied TransGAN + Swin Transformer + MedSAM concept.

## 1. D-drive paths

Set up the following folders:

```text
D:\Medical_Forgery_Project\
    datasets\
        CQ500\raw\
        MIMIC_CXR\raw\
        Forgery_Dataset\
    models\checkpoints\
    outputs\
```

Place extracted CQ500 DICOM files under `datasets\CQ500\raw`.
Place authorized MIMIC-CXR DICOM or image files under `datasets\MIMIC_CXR\raw`.

Do not redistribute MIMIC-CXR. It requires PhysioNet credentialing, required training, and a signed data-use agreement.

## 2. Create the Python environment

Open Anaconda Prompt or PowerShell:

```powershell
cd D:\Medical_Forgery_Project
py -3.12 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install a PyTorch build compatible with your NVIDIA driver from https://pytorch.org/get-started/locally/ if the default install does not provide CUDA.

Check GPU:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 3. Prepare synthetic forgery pairs

Copy `src\medical_forgery_pipeline.py` into this project and run:

```powershell
python src\medical_forgery_pipeline.py --prepare
```

The program creates authentic samples with zero masks and synthetic manipulated samples with binary masks. It creates a 70/15/15 split at the source-file level.

Important research limitation: these are controlled synthetic manipulations. They do not represent naturally occurring clinical image tampering, and they do not provide independent clinical forgery ground truth. The simple splice/inpainting-like operations in this starter code must be replaced or expanded for a publishable benchmark.

## 4. Train

```powershell
python src\medical_forgery_pipeline.py --train
```

The best Swin model is saved to:

`D:\Medical_Forgery_Project\models\checkpoints\best_swin_forgery.pth`

## 5. Predict a new image

```powershell
python src\medical_forgery_pipeline.py --predict "D:\Medical_Forgery_Project\sample.png"
```

The probability map and binary mask are saved under `outputs`.

## 6. MedSAM

The `refine_with_medsam()` function is an optional inference adapter. Clone the official MedSAM repository, install its requirements, obtain the appropriate pretrained checkpoint, and verify the checkpoint/model registry configuration for that release. MedSAM is prompted with a bounding box derived from the coarse Swin mask.

Example from Python:

```python
from pathlib import Path
from src.medical_forgery_pipeline import refine_with_medsam

refine_with_medsam(
    image_path=Path(r"D:\Medical_Forgery_Project\sample.png"),
    coarse_mask_path=Path(r"D:\Medical_Forgery_Project\outputs\predicted_probability_map.png"),
    medsam_checkpoint=Path(r"D:\Medical_Forgery_Project\models\pretrained\medsam_vit_b.pth"),
)
```

## 7. What is and is not implemented

Implemented:
- DICOM and common image loading.
- Synthetic copy-move, splice-proxy, and inpainting-like image creation.
- Paired binary masks and CSV manifests.
- Swin-Tiny hierarchical feature extraction.
- Image-level classification and pixel-level localization.
- BCE + Dice + semantic consistency losses.
- Mixed precision training, validation metrics, checkpointing.
- Transformer generator/discriminator building blocks.
- Optional MedSAM predictor-based inference refinement.

Research-development work still required:
- The transformer GAN blocks are architectural components, not a fully trained end-to-end TransGAN forgery-synthesis training loop.
- The main training path currently trains the Swin classifier/localizer on generated paired examples.
- The simple splice proxy currently uses a patch from the same image; use a second independent source image for genuine splicing experiments.
- Add subject-level splitting for MIMIC-CXR and study-level splitting for CQ500 to prevent leakage between slices/images from the same patient or study.
- Validate MedSAM on the modality and prompt strategy used in the study.
- Report classification metrics separately from pixel-level localization metrics.
- Do not claim clinical validity based only on synthetic manipulations.

## 8. Suggested methodology for the final paper

Authentic medical image -> controlled manipulation -> paired forged image and ground-truth mask -> Swin multi-scale feature extraction -> classification/localization heads -> hybrid loss -> optional MedSAM refinement -> evaluation.

Use a separate GAN experiment to study whether adversarially generated manipulations improve robustness. Document the GAN training objective and ablation results rather than describing an untrained generator as a completed TransGAN module.
