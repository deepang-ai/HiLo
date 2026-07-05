# HiLo: Spatial-Spectral Hybrid High-Low Frequency Activation for Heart and Brain Vessel Segmentation

HiLo is a 3D vessel segmentation framework for cardio-cerebrovascular imaging. It jointly models spatial and spectral high-/low-frequency cues to preserve fine tubular details while encoding large-scale anatomical structure.

[![Hugging Face weights](https://img.shields.io/badge/Weights-HuggingFace-yellow)](https://huggingface.co/deepang/HiLo)


> Published in [Expert Systems with Applications](https://www.sciencedirect.com/science/article/pii/S0957417426023626).

## Network Architecture

Cardio-cerebrovascular vessel segmentation: (a) spatial-domain-only methods focus on high-frequency local details; (b) spectral-domain-only methods process high- and low-frequency components; (c) HiLo jointly models high-/low-frequency cues in the spatial and spectral domains.

![Overview](./figures/fig1.png)

The HiLo architecture contains two core components. The HiLo block in the encoder captures spatial and spectral high-/low-frequency components to preserve fine tubular structures and encode global anatomy. The TAG module in skip connections modulates multi-domain features to enhance foreground vessels while suppressing background responses. In the inset, E denotes spectral entropy.

![Architecture](./figures/fig2.png)

## News

- Pretrained weights: [Hugging Face](https://huggingface.co/deepang/HiLo)
- Paper: [Expert Systems with Applications](https://www.sciencedirect.com/science/article/pii/S0957417426023626)

## Installation

We recommend using a clean conda environment. The following setup was used for the current codebase:

```bash
conda create -n hilo python=3.10 -y
conda activate hilo

conda install -c nvidia cudatoolkit=11.8 cuda-nvcc=11.8 -y
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118

pip install packaging
pip install causal-conv1d==1.1.1 --no-cache-dir --no-build-isolation
pip install mamba-ssm==1.1.1 --no-cache-dir --no-build-isolation

pip install -r requirements.txt
pip install -e ./nnUNet
```

Check that the bundled nnU-Net command line tools are available:

```bash
which nnUNetv2_train
which nnUNetv2_extract_fingerprint
python -c "import nnunetv2; print(nnunetv2.__file__)"
```

### Important Note About nnU-Net

This repository vendors a project-adapted `nnUNet/` package. HiLo depends on this local version for the custom trainer entry point and project-specific training behavior.

Please install the bundled package with `pip install -e ./nnUNet`. Do not replace it with the official PyPI `nnunetv2` package unless you manually port the HiLo trainer and local modifications. If this project is released publicly or used in a paper, please acknowledge both the original nnU-Net project and the third-party modified nnU-Net source that this codebase builds on. The exact upstream link/commit for the modified nnU-Net should be added here before final release.

## Data Description

### ImageCAS

ImageCAS is a large-scale coronary artery segmentation dataset containing 3D CTA images from 1,000 patients diagnosed with coronary artery disease. Each CT scan has a voxel size in the range `[512 x 512 x 206, 512 x 512 x 275]` and pixel spacing from 0.25 mm to 0.45 mm.

```bibtex
@article{zeng2023imagecas,
  title={ImageCAS: A large-scale dataset and benchmark for coronary artery segmentation based on computed tomography angiography images},
  author={Zeng, An and Wu, Chunbiao and Lin, Guisen and Xie, Wen and Hong, Jin and Huang, Meiping and Zhuang, Jian and Bi, Shanshan and Pan, Dan and Ullah, Najeeb and others},
  journal={Computerized Medical Imaging and Graphics},
  volume={109},
  pages={102287},
  year={2023},
  publisher={Elsevier}
}
```

### CAS2023

CAS2023 is a cerebral artery segmentation dataset for 3D time-of-flight magnetic resonance angiography (3D TOF-MRA). It contains 100 cerebrovascular 3D TOF-MRA scans from symptomatic patients diagnosed with intracranial arterial stenosis, with manual annotations. The volume sizes are distributed within `[208 x 320 x 96, 784 x 784 x 255]`.

```bibtex
@misc{cas2023,
  title        = {Cerebral artery segmentation challenge (cas) 2023},
  howpublished = {\url{https://codalab.lisn.upsaclay.fr/competitions/9804\#learn_the_details-overview}},
  note         = {Accessed: January 10, 2026},
  year         = {2023}
}
```

## Data Preparation

HiLo follows the nnU-Net v2 directory convention. Set the nnU-Net paths before training:

```bash
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results
export nnUNet_compile=False
export PYTHONPATH=$PWD:$PWD/nnUNet:$PYTHONPATH
```

The conversion scripts currently contain dataset-specific `base_dir` values. Update them to your local dataset locations before running.

For ImageCAS:

```bash
python scripts/imagecas/convert_imagecas_30.py
```

Expected raw layout:

```text
ImageCAS/
  data/
    000.img.nii.gz
  mask/
    000.label.nii.gz
```

For CAS2023:

```bash
python scripts/cas2023/convert_cas_40.py
```

Expected raw layout:

```text
CAS2023/
  data/
    000.nii.gz
  mask/
    000.nii.gz
```

The conversion scripts create `dataset.json`, `splits_final.json`, extract fingerprints, generate nnU-Net plans, set the `3d_fullres` patch size to `[96, 96, 96]`, and run preprocessing.

## Training

Train HiLo with the custom nnU-Net trainer:

```bash
nnUNetv2_train 40 3d_fullres 0 -tr HiLoTrainer -num_gpus 1
```

For ImageCAS, use dataset ID `30`:

```bash
nnUNetv2_train 30 3d_fullres 0 -tr HiLoTrainer -num_gpus 1
```

To continue training from the latest checkpoint:

```bash
nnUNetv2_train 40 3d_fullres 0 -tr HiLoTrainer -num_gpus 1 --c
```

The helper scripts under `scripts/cas2023/` and `scripts/imagecas/` can be used as templates. Before running them, update the hardcoded paths, GPU IDs, and training command for your local checkout.

## Evaluation and Inference

Run validation using the final checkpoint:

```bash
nnUNetv2_train 40 3d_fullres 0 -tr HiLoTrainer --val
```

Run inference on a folder of nnU-Net formatted images:

```bash
nnUNetv2_predict \
  -i /path/to/imagesTs \
  -o /path/to/predictions \
  -d 40 \
  -c 3d_fullres \
  -tr HiLoTrainer \
  -f 0
```

## Benchmark

![Benchmark](./figures/fig4.png)
![Benchmark](./figures/fig3.png)

## Visualization

Qualitative comparison of 3D segmentation results on ImageCAS and CAS2023. Red indicates true positives; green denotes false positives.

![Visualization](./figures/fig5.png)

## Project Structure

```text
HiLo/
  nnUNet/                         # bundled modified nnU-Net v2 package
  src/model/HiLo/                 # HiLo network implementation
  scripts/cas2023/                # CAS2023 conversion and training helpers
  scripts/imagecas/               # ImageCAS conversion and training helpers
  figures/                        # README figures
  requirements.txt
```

## Acknowledgement

This project builds on nnU-Net v2 and related open-source medical image segmentation tooling. The repository includes a modified nnU-Net package under `nnUNet/`; please keep the upstream license and citation requirements when using or redistributing this code.

README organization is inspired by common medical segmentation release pages such as UltraMamba.

## Citation

If you use HiLo, please cite:

```bibtex
@article{HUANG2026133453,
title = {HiLo: Spatial-Spectral Hybrid High-Low Frequency Activation for Heart and Brain Vessel Segmentation},
journal = {Expert Systems with Applications},
pages = {133453},
year = {2026},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2026.133453},
url = {https://www.sciencedirect.com/science/article/pii/S0957417426023626},
author = {Jiahui Huang and Xin Lei and Qiong Wang and Valentin Sinitsyn and Yun Zhu and Ying Hu and Hao Chen and Yan Pang}
}
```

Please also cite nnU-Net when using the bundled training framework:

```bibtex
@article{isensee2021nnunet,
  title={nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation},
  author={Isensee, Fabian and Jaeger, Paul F and Kohl, Simon AA and Petersen, Jens and Maier-Hein, Klaus H},
  journal={Nature Methods},
  volume={18},
  number={2},
  pages={203--211},
  year={2021},
  publisher={Nature Publishing Group}
}
```
