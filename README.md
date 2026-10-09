# :page_facing_up: MSSA
This is the official PyTorch implementation of our ECCV 2026 paper "[Memory-Supported Synergistic Adaptation
for Training-Free Test-Time Medical Image Segmentation](https://arxiv.org/pdf/2607.17693)".

<div align="center">
  <img width="100%" alt="MSSA Illustration" src="Overview.png">
</div>

## Environment
```
CUDA 10.1
Python 3.7.0
Pytorch 2.0.1
CuDNN 8.0.5
```
Our Anaconda environment is also available for download from [Google Drive](https://drive.google.com/file/d/1fmqEl5HDHjuQ3Ih4vD81B0xOHC5fLSE7/view?usp=drive_link).

Upon decompression, please move ```mssa``` to ```your_root/anaconda3/envs/```. Then the environment can be activated by ```conda activate mssa```.

## Data Preparation
The preprocessed fundus data can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1axgu3-65un-wA_1OH-tQIUIEHEDrnS_-?usp=drive_link).
The lung data can be downloaded from [Baidu Drive1](https://pan.baidu.com/s/1CTFUUzZIeI-yeWPajsmnZA?pwd=smsp)  [Baidu Drive1](https://pan.baidu.com/s/1H949elVhFmVBjDh-UjE-fg?pwd=smsp)  [Baidu Drive1](https://pan.baidu.com/s/1F25nICXacMk-dzX4m_SCzQ?pwd=smsp).

## Pre-trained Models
Download SAM vit_h Model from [Google Drive](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) and drag it into the checkpoint folder.
Download MedSAM_CLIP fine-tuned BiomedCLIP Model from [Google Drive](https://drive.google.com/file/d/1jjnZabUlc9_gpcP0d2nz_GNS-EGX0lq5/view?usp=sharing)  and drag it into the checkpoint folder.


* **OD Segmentation**
```
python ours.py \
    --input "${INPUT_DIR}" \
    --output "${OUTPUT_ROOT}" \
    --gt-dir "${GT_DIR}" \
    --sam-checkpoint "${SAM_CHECKPOINT}" \
    --sam-model-type vit_h \
    --clip-model-path "${CLIP_MODEL_PATH}" 
```
* **Lung Segmentation**
```
python ours_lung.py \
    --input "${INPUT_DIR}" \
    --output "${OUTPUT_ROOT}" \
    --gt-dir "${GT_DIR}" \
    --sam-checkpoint "${SAM_CHECKPOINT}" \
    --sam-model-type vit_h \
    --clip-model-path "${CLIP_MODEL_PATH}" 
```

## How to Run
Please first modify the root in ```MSSA_OD.sh``` and ```MSSA_Lung.sh```, and then run the following command.
```
# On the OD segmentation task (for different datasets, change the path in DATASET and OUTPUT_ROOT )
bash MSSA_OD.sh
# On the lung MC segmentation task
bash MSSA_Lung_MC.sh
# On the lung Shenzhen segmentation task
bash MSSA_Lung_sz.sh
# On the lung Xray (Covid) segmentation task
bash MSSA_Lung_xray.sh
```

## Citation ✏️
If this code is helpful for your research, please cite:
```
@inproceedings{li2026memory,
  title={Memory-Supported Synergistic Adaptation for Training-Free Test-Time Medical Image Segmentation},
  author={Li, Lingrui and Pu, Nan and Zhao, Dong and Li, Wenjing and French, Andrew P and Zhong, Zhun and Chen, Xin},
  booktitle={European Conference on Computer Vision},
  pages={411--429},
  year={2026},
  organization={Springer}
}
```
## Acknowledgement
Our code is adapted from the PyTorch implementations of [MedClipSAMv2](https://github.com/HealthX-Lab/MedCLIP-SAMv2).

## Contact
Lingrui Li (lingrui.li@nottingham.ac.uk)
