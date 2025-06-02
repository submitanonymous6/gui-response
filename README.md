# MobileGUIPerf: Measuring GUI Responsiveness from Mobile Screencasts

This repository contains the implementation, dataset, and example output used in our study on **GUI responsiveness in mobile applications**.

**MobileGUIPerf** is a black-box tool that analyzes screencast videos captured during automated GUI testing to measure **user-perceived GUI responsiveness**. It automatically detects user actions (e.g., taps or swipes) and computes two key metrics based on visual changes in frames:

- **Response Time**: Time from the user action to the first visible feedback
- **Finish Time**: Time until the visual feedback stabilizes

> 🔧 Note: To improve readability and usability, the code is still under active development and may be updated regularly.
---

## 🎯 Features

- **No source code required**: Operates solely on video (black-box analysis)
- **Automatic interaction detection**: Tap and swipe detection via Faster R-CNN
- **Visual similarity-based timing**: Uses SSIM + Isolation Forest to detect UI response and finish frames
- **Scalable**: Designed to process thousands of screencasts daily
- **Dataset released**: 2,458 annotated interactions from 64 Android apps

---

## 📂 Repository Structure

```text
GUI-Response/
├── dataset/        # Annotated benchmark of 2,458 user interactions
├── examples/       # Sample videos and output reports
├── models/         # Pretrained detection model (e.g., Faster R-CNN)
├── src/            # Core implementation
├── scripts/        # Utility scripts
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Install dependencies

We recommend Python 3.9+.

```bash
pip install -r requirements.txt
```

Dependencies include:
- PyTorch
- torchvision
- OpenCV
- scikit-learn
- ffmpeg-python

### 2. Run MobileGUIPerf

```bash
python src/main.py --video examples/demo.mp4 --output results/
```

The output includes:
- Response and finish times per interaction
- Optional: annotated clips or logs per user interaction

---

## 📊 Dataset

We provide a benchmark dataset of **2,458 manually annotated interactions** from **64 popular Android apps**, covering 32 app categories.

Each interaction is labeled with:
- Start, response, and finish frame indices
- Type of interaction (tap or swipe)
---

## 📌 Notes

- Screencasts must be recorded with Android's **Show taps** feature enabled
- Videos are encouraged to be recorded at 60 FPS (16.7ms per frame)
- See the `examples/` folder for demo inputs and sample outputs
- **GPU is required** for efficient video/image processing
---

