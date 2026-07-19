# CDMNet

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)

---

## 📖 Introduction

1.  This is the code for our paper: A Cross-Domain Mamba Network with joint spatial-frequency learning for robust SAR oil spill detection

---

## 💻 System Requirements

This model is optimized for **accuracy-critical applications** in geosciences. Due to the sophisticated attention mechanisms and high-resolution inputs, it requires a robust hardware environment.

*   **OS:** Linux (Ubuntu 20.04/22.04 recommended).
*   **GPU:** NVIDIA GPU with **Compute Capability ≥ 8.0** (Ampere architecture or newer).
*   **VRAM:** 
    *   **Training:** ≥ 24GB (e.g., RTX 3090/4090/5090) is strongly recommended.
    *   **Inference:** ≥ 12GB.
*   **CUDA:** 12.4 (Strictly required for the provided installation steps).

---

## 🛠️ Installation

To ensure reproducibility, please follow these steps strictly to configure the Mamba environment.

### 1. Clone the repository
```bash
git clone https://github.com/FF7CA/CDMNet.git
cd CDMNet
```

### 2. Create Environment
```bash
conda create -n aerith python=3.12
conda activate aerith
```

### 3. Install PyTorch (CUDA 12.4)

Note: We use PyTorch 2.6.0 which is compatible with Mamba 2.2.4.
```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

### 4. Install Mamba-SSM & Dependencies
This step requires nvcc (CUDA compiler) to be available in your path.
```bash
# Install core libraries
pip install -r requirements.txt

# Install Mamba components
pip install causal-conv1d==1.6.0
pip install mamba-ssm==2.2.4
```

### 5. Install VMamba
This step requires nvcc (CUDA compiler) to be available in your path.
```bash
# Install VMamba libraries
git clone https://github.com/MzeroMiko/VMamba.git
cd VMamba
pip install -r requirements.txt
cd kernels/selective_scan && pip install .
```

## :rocket:Training

```
python train_supervision.py -c ./config/Palsar/CDMNet.py
```

```
python train_supervision.py -c ./config/Sentinel/CDMNet.py
```

```
python train_supervision.py -c ./config/M4D/CDMNet.py
```

## :100: Testing

```
python SOS_seg_test.py -c ./config/Palsar/CDMNet.py -o /root/results/Palsar/CDMNet --rgb -t 'lr'
```

```
python SOS_seg_test.py -c ./config/Sentinel/CDMNet.py -o /root/results/Sentinel/CDMNet --rgb -t 'lr'
```

```
python M4D_test.py -c ./config/M4D/CDMNet.py -o /root/results/M4D/CDMNet --rgb -t 'lr'
```

---

## 📂 Data Preparation

### Download the dataset

```
The datasets used and analyzed during the current study are available from the following sources: SOS dataset: https://grzy.cug.edu.cn/zhuqiqi/en/yjgk/32384/list/index.html; M4D dataset: https://m4d.iti.gr/oil-spill-detection-dataset/.
```

### Organize the data as follows:

```text
datasets/
├── SOS/
│   ├── test/
│   ├── train/
└── M4D/
│   ├── test/
│   ├── train/
```
## 🤝 Acknowledgement

Our training scripts comes from [GeoSeg](https://github.com/WangLibo1995/GeoSeg). Thanks for the author's open-sourcing code.

- [GeoSeg(UNetFormer)](https://github.com/WangLibo1995/GeoSeg)
- [pytorch lightning](https://www.pytorchlightning.ai/)
- [timm](https://github.com/rwightman/pytorch-image-models)
- [pytorch-toolbelt](https://github.com/BloodAxe/pytorch-toolbelt)
- [ttach](https://github.com/qubvel/ttach)
- [catalyst](https://github.com/catalyst-team/catalyst)
- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)

