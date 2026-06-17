# 🌴 Palm Cooking Oil Detection System

![YOLOv11](https://img.shields.io/badge/Model-YOLOv11-blue?style=for-the-badge&logo=ultralytics)
![Python](https://img.shields.io/badge/Python-3.8%2B-green?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange?style=for-the-badge&logo=pytorch)

A production-ready, end-to-end AI-powered computer vision system designed to detect **Palm Cooking Oil** using the state-of-the-art **YOLOv11** architecture. This repository includes everything needed to train the model, run real-world inference, and interact with the model via a premium Tkinter Dashboard.

---

## 🌟 Features

- **🚀 YOLOv11 Training Pipeline**: Fully automated and hyperparameter-tuned training script (`train_yolov11_palm_oil.py`) optimized for custom datasets.
- **🖥️ Interactive Dashboard**: A premium, dark-themed Tkinter GUI (`dashboard.py`) for live inference visualization, featuring adjustable Confidence and IoU sliders.
- **🔍 Real-World Inference**: Robust script (`realworld_infer.py`) to run detection on single images, directories, live webcams, or video files with ByteTrack tracking support.
- **📊 Dataset Checking Utility**: Included `check_dataset.py` to validate your Roboflow dataset format before training.

---

## 📂 Repository Structure

```text
oil-detection-system/
├── train_yolov11_palm_oil.py   # Training pipeline & hyperparameter tuning
├── dashboard.py                # Premium Tkinter UI for interactive testing
├── realworld_infer.py          # Production inference (Images/Video/Webcam)
├── check_dataset.py            # Dataset validation utility
├── README.md                   # Project documentation
├── inference_results.json      # Output logs from realworld_infer.py
└── runs/                       # Model training outputs and weights
```

---

## 🛠️ Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd oil-detection-system
   ```

2. **Install the dependencies:**
   Make sure you have Python 3.8+ installed, then run:
   ```bash
   pip install ultralytics torch torchvision torchaudio opencv-python roboflow pillow --upgrade
   ```

---

## 🚀 Usage

### 1. Training the Model
To start training the YOLOv11 model on the palm cooking oil dataset:
```bash
python train_yolov11_palm_oil.py
```
*Note: Make sure to set your `ROBOFLOW_API_KEY` in the script if you want to auto-download the dataset.*

### 2. Launching the Interactive Dashboard
To interactively test your trained weights with a beautiful UI:
```bash
python dashboard.py
```
- Click **Choose Image** to load an image.
- Adjust **Confidence** and **IoU** thresholds using the sliders for real-time results.

### 3. Real-World Inference (CLI)
Run batch predictions or test on live video streams:
```bash
# Test on a single image
python realworld_infer.py --source test_image.jpg --weights palm_oil_yolov11/train_v1/weights/best.pt

# Test on a directory of images
python realworld_infer.py --source /path/to/images/ --weights palm_oil_yolov11/train_v1/weights/best.pt

# Test on a video file with ByteTrack tracking
python realworld_infer.py --source factory_line.mp4 --track --show

# Live webcam inference
python realworld_infer.py --source 0 --show
```
