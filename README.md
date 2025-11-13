# Compare - Object Detection Comparison Tool

This project uses multiple object detection models to identify the precense of people on a video stream and sends alerts via Telegram. This can be easily configured to give you something like your own hosted version of "Ring" smart motion activated alerts.

## Features

-   Compares results from 5 different object detection models running in parallel.
-   Models used: DETR, YOLOS, Faster R-CNN, RetinaNet, YOLOv5.
-   Sends image alerts to specified Telegram chats when a person is detected with sufficient confidence to avoid noise/false positive results.
-   Configurable alerting thresholds and camera settings.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd compare
    ```

2.  **Install dependencies:**
    This project requires Python and several libraries. You will need `pyyaml`, `torch`, `torchvision`, `transformers`, `opencv-python`, `imutils`, `numpy`, `scikit-image`, `telepot`, `imagehash`.
    It is recommended to use a virtual environment.

    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure the application:**
    -   Copy the example configuration file:
        ```bash
        cp config.yaml.example config.yaml
        ```
    -   Edit `config.yaml` to add your Telegram bot token and chat IDs.
    -   Adjust other settings like camera index and model confidence thresholds as needed.

## Usage

Run the main script:

```bash
python compare.py
```

The script will start the camera, process the video stream with the configured models, and send alerts if the conditions are met.
