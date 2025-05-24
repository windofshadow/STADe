# 🔬 Title of Your Research Paper
Official implementation of the paper:  
"STADe: Sensory Temporal Action Detection via Temporal-Spectral Representation Learning"
Authors: Bing Li, Haotian Duan, Yun Liu, Le Zhang, Wei Cui, Joey Tianyi Zhou
Published in *IEEE TPAMI*, 2025.  
[[Paper](https://arxiv.org/abs/XXXX.XXXXX)] 

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

## 🛠 Installation

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt


## ⚙️ Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
Or with conda:


🚀 Usage
Training
bash
复制
编辑
bash scripts/run_train.sh
Or with Python:

bash
复制
编辑
python src/train.py --config configs/default.yaml
Evaluation
bash
复制
编辑
bash scripts/run_eval.sh
You can download our pretrained checkpoints from this link and place them in the checkpoints/ directory.

📊 Results
Dataset	Metric	Baseline	Ours
Dataset A	mAP@0.5	81.2	86.7
Dataset B	F1	0.88	0.93

Additional qualitative results and visualizations can be found in the results/ folder.

🧠 Key Contributions
🔧 A new method for sensory temporal action detection using Temporal-Spectral Representation Learning

📈 Strong improvements over prior work on multiple datasets

📦 Pretrained models and reproducible evaluation pipeline

📝 Citation
If you find this work helpful, please cite:

@article{Li2025STADe,
  author       = {Bing Li and Haotian Duan and Yun Liu and Le Zhang and Wei Cui and Joey Tianyi Zhou},
  title        = {STADe: Sensory Temporal Action Detection via Temporal-Spectral Representation Learning},
  journal      = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year         = {2025},
  note         = {To appear},
  doi          = {10.1109/TPAMI.2025.xxxxx}, 
  publisher    = {IEEE}
}

📬 Contact
For questions or collaborations, feel free to reach out:

✉️ bing_li@uestc.edu.cn

📄 License
This project is licensed under the MIT License.
