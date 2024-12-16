# ATLASS: An AnaTomicaLly-Aware Self-Supervised Learning Framework for Generalizable Retinal Disease Detection

This repository contains the **self-supervised pretraining** (`pretrain` folder) and **fine-tuning** (`finetune` folder) code for our paper:

**ATLASS: An AnaTomicaLly-Aware Self-Supervised Learning Framework for Generalizable Retinal Disease Detection**.

---

## Weights Download

Pretrained and fine-tuned model weights can be downloaded from [this OneDrive folder](https://stduestceducn-my.sharepoint.com/:f:/g/personal/201714060114_std_uestc_edu_cn/Ev2HO8d20ZtFrgk940KM5FIBH9w-ZDv94HAhab_cQIuGDQ?e=acOFLb).  
After downloading, place them inside the **`checkpoints_large/`** directory (or wherever your code expects them), for example:


---

## Required Structure

```
├── pretrain/
│   ├── pretrain.py
│   ├── vessel_best_model.pth  # Place downloaded model weights here
│   ├── models/
│   │   ├── vit_models.py
│   │   └── ...
│   ├── utils/
│   │   └── utils.py
│   ├── config/
│   │   └── config.py
│   └── data/
│       └── dataset.py
├── finetune/
│   ├── finetune.py
│   ├── models/
│   │   └── ...
│   ├── utils/
│   │   └── ...
│   ├── config/
│   │   └── ...
│   └── data/
│       └── ...
├── checkpoints_large/
│   ├── vit_weights.pth   
│   └── best_finetuned_model.pth #(if you want to test only)
```

- **`pretrain/`**: Self-supervised learning stage (ATLASS).
- **`finetune/`**: Fine-tuning code for downstream classification or segmentation tasks.
- **`checkpoints_large/`**: Contains pre-trained and fine-tuned model weights.

---

## Requirements

- Python 3.8+
- PyTorch (1.13+ / 2.0+ recommended)
- TorchVision
- [timm](https://github.com/huggingface/pytorch-image-models)
- Albumentations
- scikit-image
- scikit-learn
- Pillow
- NumPy
- tqdm
- matplotlib

```bash
pip install -r requirements.txt
```

---

## Dataset Preparation

For dataset orientation, refer to the **RETFound** repository:

- **Benchmark & Dataset Links**: [RETFound_MAE Benchmark](https://github.com/rmaphoh/RETFound_MAE/blob/main/BENCHMARK.md)

**Dataset Structure**:
```
datasets/
├── DatasetA/
│   ├── train/
│   ├── val/
│   └── test/
├── DatasetB/
│   ├── train/
│   ├── val/
│   └── test/
└── ...
```

Each dataset should have `train/`, `val/`, and `test/` subfolders, each containing class-labeled images.
To pretrain the model with self-supervised learning, download the dataset from [Kaggle - EyePACS, APTOS, Messidor](https://www.kaggle.com/datasets/ascanipek/eyepacs-aptos-messidor-diabetic-retinopathy).
Ensure the folder contains approximately **92,000 images** properly organized into `datasets/Combined/train/` subdirectory. Its better to use additional augmented images aswell. 

---

## Pretraining (Self-Supervised)

**ATLASS** is conducted in the `pretrain/` folder.

1. **Configure**:
   - Edit `pretrain/config/config.py` for hyperparameters (batch size, learning rate, etc.) and dataset paths.
2. **Run**:
   ```bash
   cd pretrain
   python pretrain.py
   ```
3. **Checkpoints & Logs**:
   - Output weights and logs appear in `checkpoints_large/`.
   - Intermediate checkpoint files may be saved periodically.

---

## Fine-Tuning

Use the pretrained model from `pretrain/` for supervised tasks:

1. **Configure**:
   - Edit `finetune/config/config.py` (paths, hyperparameters).
2. **Run**:
   ```bash
   cd finetune
   python finetune.py
   ```
3. **Results**:
   - Final fine-tuned weights saved in `checkpoints_large/` (e.g., `best_finetuned_model.pth`).

---

## Weights Files

**`vessel_best_model.pth`** and other pretrained weights:
- keep vessel_best_model.pth in pretrain folder (at same level or adjust paths)
- Download from [this OneDrive folder](https://stduestceducn-my.sharepoint.com/:f:/g/personal/201714060114_std_uestc_edu_cn/Ev2HO8d20ZtFrgk940KM5FIBH9w-ZDv94HAhab_cQIuGDQ?e=acOFLb)
- Place them under `checkpoints_large/`.

---

## Example Commands

```bash
# Pretraining
cd pretrain
python pretrain.py

# Fine-tuning
cd ../finetune
python finetune.py
```

---

## Troubleshooting

1. **ModuleNotFoundError**: Ensure each directory (`pretrain`, `finetune`, etc.) has an `__init__.py` if you’re using imports across folders.  
2. **OOM Errors**: Lower batch size, patch size, or use gradient accumulation.  
3. **Dataset Not Found**: Double-check your dataset path in `config.py`.

---

## Acknowledgements

- **Datasets**: Download as described in the [RETFound_MAE benchmarks](https://github.com/rmaphoh/RETFound_MAE/blob/main/BENCHMARK.md).
- **Weights**: Provided [OneDrive link](https://stduestceducn-my.sharepoint.com/:f:/g/personal/201714060114_std_uestc_edu_cn/Ev2HO8d20ZtFrgk940KM5FIBH9w-ZDv94HAhab_cQIuGDQ?e=acOFLb).
- For further details, see the ATLASS paper for anatomically-aware self-supervision in retinal imaging (Will add details later).

---

