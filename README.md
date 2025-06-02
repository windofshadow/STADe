# 🔬 Title of Your Research Paper
Official implementation of the paper:  
"STADe: Sensory Temporal Action Detection via Temporal-Spectral Representation Learning"
Authors: Bing Li, Haotian Duan, Yun Liu, Le Zhang, Wei Cui, Joey Tianyi Zhou
Published in *IEEE TPAMI*, 2025.  
[[Paper](B. Li, H. Duan, Y. Liu, L. Zhang, W. Cui and J. T. Zhou, "STADe: Sensory Temporal Action Detection via Temporal-Spectral Representation Learning," in IEEE Transactions on Pattern Analysis and Machine Intelligence, doi: 10.1109/TPAMI.2025.3574367)] 

---

## 🚀 Introduction
Temporal action detection (TAD) is a vital challenge in computer vision and the internet of things, aiming to detect and identify actions within temporal sequences. While TAD has primarily been associated with video data, its applications can also be extended to sensor data, opening up opportunities for various real-world applications. However, applying existing TAD models to sensory signals presents distinct challenges such as varying sampling rates, intricate pattern structures, and subtle, noise-prone patterns. In response to these challenges, we propose a Sensory Temporal Action Detection (STADe) model. STADe leverages Fourier kernels and adaptive frequency filtering to adaptively capture the nuanced interplay of temporal and frequency features underlying complex patterns. Moreover, STADe embraces adaptability by employing deep fusion at varying resolutions and scales, making it versatile enough to accommodate diverse data characteristics, such as the wide spectrum of sampling rates and action durations encountered in sensory signals. Unlike conventional models with unidirectional category-to-proposal dependencies, STADe adopts a cross-cascade predictor to introduce bidirectional and temporal dependencies within categories. To extensively evaluate STADe and promote future research in sensory TAD, we establish three diverse datasets using various sensors, featuring diverse sensor types, action categories, and sampling rates. Experiments across one public and our three new datasets demonstrate STADe's superior performance over state-of-the-art TAD models in sensory TAD tasks. 

---

## 📁 Repository Structure

- `src/`: Core implementation of the method.
- `scripts/`: Shell scripts for training/evaluation.
- `configs/`: YAML config files for different experimental settings.
- `checkpoints/`: Pretrained models.
- `results/`: Evaluation outputs.
- `docs/`: Additional documentation and visualizations.

---

### Environment
- Python 3.7
- PyTorch == 1.10.0 
- CUDA == 11.3
- python == 3.8
- ubuntu == 20.04
- GPU == RTX3090

---

## ⚙️ Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd STADe-DeepSeg/STADe-DeepSeg
pip install -r requirements.txt
python3 setup.py develop
```

### Data Preparation
- **DeepSeg data:**
1. Download the DeepSeg npy data.Baidu Netdisk:  https://pan.baidu.com/s/1b5EJTrzpDTgm-MsIzohZeQ?pwd=nzn7  (Password: nzn7) 
2. Unzip the DeepSeg npy data to `./Public_Behave_Data_npy/` (Align with the `video_data_path` parameter in thumos14.yaml)
- **Sebehave data:**
1. Download Sebehave npy data (13.7GB): Baidu Netdisk: https://pan.baidu.com/s/1ac3AUZRYYnQXkeaYc_so2w?pwd=kq6d (Password: kq6d)
2. Unzip the DeepSeg npy data to `./Sebehave_Data_npy/` (Align with the `video_data_path` parameter in thumos14.yaml)
- **Wikeystroke data:**
1. Download Wikeystroke npy data (13.7GB): Baidu Netdisk:  https://pan.baidu.com/s/11eKvRfICoOLOCCG1omWgbA?pwd=3hxg (Password: 3hxg)
2. Unzip the Wikeystroke npy data to `./Wikeystroke_Data_npy/` (Align with the `video_data_path` parameter in thumos14.yaml)
- **Wibehave data:**
1. Download Wibehave npy data (13.7GB): Baidu Netdisk: https://pan.baidu.com/s/1iyFOc6kiGgime2VSjTbpEg?pwd=drp7  (Password: drp7)
2. Unzip the Wibehave npy data to `./Wibehave_Data_npy/` (Align with the `video_data_path` parameter in thumos14.yaml)


## 🚀 Usage

To train the model:

```bash
1. cd .../STADe-DeepSeg
2. python STADe-DeepSeg/train.py
```
To test the model:
```bash
1. cd .../STADe-DeepSeg
2. python STADe-DeepSeg/test.py
```



## 🧠 Key Contributions
🔧 A new method for sensory temporal action detection using Temporal-Spectral Representation Learning

📈 Strong improvements over prior work on multiple datasets

📦 Pretrained models and reproducible evaluation pipeline

## 📝 Citation
If you find this work helpful, please cite:

@article{Li2025STADe,
  author       = {Bing Li and Haotian Duan and Yun Liu and Le Zhang and Wei Cui and Joey Tianyi Zhou},
  title        = {STADe: Sensory Temporal Action Detection via Temporal-Spectral Representation Learning},
  journal      = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year         = {2025},
  note         = {To appear},
  doi          = {10.1109/TPAMI.2025.3574367}, 
  publisher    = {IEEE}
}

## 📬 Contact
For questions or collaborations, feel free to reach out:

✉️ bing_li@uestc.edu.cn

## 📄 License
This project is licensed under the MIT License.
