# 🔍 Compare - An Ensemble Learning Approach to Object Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Lines of Code](https://tokei.rs/b1/github/headertag/compare)](https://github.com/headertag/compare)

**Compare** is a sophisticated, real-time object detection system designed for high-accuracy monitoring. It leverages an ensemble of five distinct, state-of-the-art object detection models, running in parallel, to create a highly reliable and nuanced alert system. This project is the result of extensive research concluding that a multi-model ensemble is the most effective strategy to minimize false positives and create a robust detection signal, especially in challenging conditions.

The core philosophy is that by combining the outputs of diverse models—each with its own training data, biases, and architectural nuances—we can overcome the limitations of any single model and achieve a more holistic and trustworthy understanding of the visual data.

## ✨ Key Features

-   **Ensemble of Five Models**: Utilizes DETR, YOLOS, Faster R-CNN, RetinaNet, and YOLOv5 simultaneously to analyze a video stream.
-   **High-Accuracy Person Detection**: The ensemble approach significantly reduces false positives and negatives, providing reliable alerts.
-   **Real-time Telegram Alerts**: Receive instant image alerts in your Telegram chat when a person is detected with high confidence.
-   **Highly Configurable**: Easily adjust model confidence thresholds, alert sensitivity, camera settings, and more.
-   **Efficient & Modern Codebase**: A modular and testable architecture that avoids disk I/O for image processing.
-   **CPU & GPU Support**: Automatically detects and uses a CUDA-enabled GPU, with a seamless fallback to CPU if not available.
-   **Web Dashboard**: A simple web interface to view the live camera feed with bounding boxes.

## ⚡ Performance and Low Latency

This system is designed for low-latency performance. By running the five object detection models in parallel threads, we maximize CPU/GPU utilization and ensure that the video stream is processed as quickly as possible.

For the lowest possible latency and highest throughput, **a CUDA-enabled GPU is highly recommended**. The models will automatically run on the GPU if one is detected, significantly accelerating the inference process.

## 🔧 How It Works

1.  **Capture**: The system captures a frame from the video stream.
2.  **Parallel Inference**: The frame is passed *in-memory* to all five models, which run in parallel threads for maximum efficiency.
3.  **Ensemble Aggregation**: Each model returns a confidence score for the presence of a person. These scores are aggregated.
4.  **Thresholding**: If the combined score surpasses a user-defined sensitivity threshold, an alert is triggered.
5.  **Alerting**: An image of the event, with bounding boxes from the models, is sent to your specified Telegram chats.

## 📂 Project Structure

The project has been refactored into a modular and maintainable structure:

```
/
├── main.py             # Main application entry point
├── dashboard.py        # Web dashboard for live feed
├── config.py           # Configuration loading and management
├── camera.py           # Camera handling
├── model_loader.py     # Model loading and inference logic
├── alerts.py           # Telegram alerting functionality
├── tests/              # Unit and integration tests
├── config.yaml         # Your local configuration
├── config.yaml.example # Example configuration
├── requirements.txt    # Project dependencies
└── README.md           # This file
```

## 🚀 Setup

### 1. Clone the Repository

```bash
git clone https://github.com/headertag/compare.git
cd compare
```

### 2. Create a Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note on Dependencies**: This project requires several deep learning and computer vision libraries. The `requirements.txt` file includes specific versions and additional libraries like `timm`, `ultralytics`, `pandas`, and `seaborn` that were added to support all the models. We also constrain `numpy` to a version below `2.0` to avoid compatibility issues.

### 4. Configure the Application

Copy the example configuration file:

```bash
cp config.yaml.example config.yaml
```

Now, edit `config.yaml` with your settings:

-   **`telegram.token`**: Your Telegram bot token.
-   **`telegram.chat_ids`**: A list of chat IDs to send alerts to.
-   **`processing.device`**: Set to `"cuda"` if you have a compatible GPU, otherwise `"cpu"` (default).
-   Adjust other settings like `camera` and `alerting` thresholds as needed.

## 🏃 Usage

To start the main application (command-line only):

```bash
python main.py
```

To view the live feed with bounding boxes in your browser, run the web dashboard:

```bash
python dashboard.py
```
Then open your browser to `http://0.0.0.0:8080`.

## 🧪 Testing

A `tests/` directory has been set up with `pytest`. You can run the tests with:

```bash
python -m pytest
```

---
*This project demonstrates the power of ensemble learning in practical, real-world applications. By moving beyond single-model solutions, we unlock a new level of reliability and performance.*