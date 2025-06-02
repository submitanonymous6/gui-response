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
- scikit-learn
- PyTorch
- OpenCV
- pytorch-msssim
- TensorFlow

### 2. Run MobileGUIPerf

```bash
python src/main.py --video examples/demo.mp4 --output results/
```

The output includes:
- Response and finish times per interaction
- Optional: annotated clips or logs per user interaction

<p style="color:red;"><strong>Warning:</strong> The file <code>demo.mp4</code> is from a previous study and is provided for demonstration purposes only. All content in the video is anonymized and has no connection to the author personally.</p>

---

## 📊 Dataset Overview

We release a benchmark dataset of **2,458 manually annotated user interactions**, serving as the ground truth for evaluating GUI responsiveness.

📁 Annotation file:
```text
dataset/annotated_user_interactions_groundtruth.csv
```

Each row in the CSV corresponds to a user interaction annotated with its timing boundaries and computed responsiveness metrics.

---

### 📄 File Format

The CSV file contains the following columns:

| Column Name                      | Description                                                                 |
|----------------------------------|-----------------------------------------------------------------------------|
| `video`                          | Name of the screencast video (excluding extension)                          |
| `index`                          | Index of the user interaction (1-based per video)                           |
| `index of user operation frame`  | Frame index where the user action (e.g., tap) is first observed             |
| `index of response frame`        | Frame index where the GUI first reacts with visible feedback                |
| `index of finish frame`          | Frame index where the GUI visual response stabilizes                        |
| `index of end frame`             | Frame index where this interaction ends (before the next user input starts) |
| `response time (ms)`             | Time from user operation to response frame, in milliseconds                 |
| `finish time (ms)`               | Time from user operation to finish frame, in milliseconds                   |

All frame indices are 0-based. The frames can be extracted from the videos using FFmpeg or OpenCV.


### 🔗 Screencast Videos

The raw screencast videos associated with these annotations are hosted by the previous study [video2sceneario](https://sites.google.com/view/video2sceneario/home).
<p style="color:red;"><strong>Warning:</strong> All screencast videos are from a previous study and have no connection to the author personally. All content is anonymized.</p>

---

### 📌 Notes

- All annotations were created manually by experienced testers
- Tap indicators in the screencasts were enabled via Android’s **"Show taps"** setting
- These annotations serve as the **ground truth** for evaluating the accuracy of interaction detection and GUI responsiveness measurement



