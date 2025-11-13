# Compare - An Ensemble Learning Approach to Object Detection

**Compare** is a sophisticated, real-time object detection system designed for high-accuracy monitoring. It leverages an ensemble of five distinct, state-of-the-art object detection models, running in parallel, to create a highly reliable and nuanced alert system. This project is the result of extensive research concluding that a multi-model ensemble is the most effective strategy to minimize false positives and create a robust detection signal, especially in challenging conditions.

The core philosophy is that by combining the outputs of diverse models—each with its own training data, biases, and architectural nuances—we can overcome the limitations of any single model and achieve a more holistic and trustworthy understanding of the visual data.

## Key Features

-   **Ensemble of Five Models**: Utilizes DETR, YOLOS, Faster R-CNN, RetinaNet, and YOLOv5 simultaneously to analyze a video stream.
-   **High-Accuracy Person Detection**: The ensemble approach significantly reduces false positives and negatives, providing reliable alerts.
-   **Real-time Telegram Alerts**: Receive instant image alerts in your Telegram chat when a person is detected with high confidence.
-   **Highly Configurable**: Easily adjust model confidence thresholds, alert sensitivity, camera settings, and more.
-   **Efficient & Modern Codebase**: A modular and testable architecture that avoids disk I/O for image processing.
-   **CPU & GPU Support**: Automatically detects and uses a CUDA-enabled GPU, with a seamless fallback to CPU if not available.

## How It Works

1.  **Capture**: The system captures a frame from the video stream.
2.  **Parallel Inference**: The frame is passed *in-memory* to all five models, which run in parallel threads for maximum efficiency.
3.  **Ensemble Aggregation**: Each model returns a confidence score for the presence of a person. These scores are aggregated.
4.  **Thresholding**: If the combined score surpasses a user-defined sensitivity threshold, an alert is triggered.
5.  **Alerting**: An image of the event, with bounding boxes from the models, is sent to your specified Telegram chats.

## Project Structure

The project has been refactored into a modular and maintainable structure:

```
/
├── main.py             # Main application entry point
├── dashboard.py        # Example single image dashboard 
├── config.py           # Configuration loading and management
├── camera.py           # Camera handling
├── models.py           # Model loading and inference logic
├── alerts.py           # Telegram alerting functionality
├── tests/              # (Coming Soon) Unit and integration tests
├── config.yaml         # Your local configuration
├── config.yaml.example # Example configuration
├── requirements.txt    # Project dependencies
└── README.md           # This file
```

## Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
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

## Usage

To start the application, simply run:

```bash
python main.py
```

The system will initialize the camera and models and begin monitoring the video stream.

## Making it Your Own

### Adding a New Model

The modular design makes it easy to add new models to the ensemble.

1.  **Add a `run_new_model` function** to `models.py`. It should take an image and two lists (`results_list`, `box_list`) as input.
2.  **Load your model** in `models.py`.
3.  **Add a new thread** in `main.py` to run your new model.

### Testing (Coming Soon)

A `tests/` directory will be added with unit tests for each module. This will allow for robust testing of individual components and the system as a whole.

## Future Directions & Creative Enhancements

This project provides a strong foundation for a variety of enhancements:

-   **Containerization**: A `Dockerfile` will be added to allow for easy deployment as a portable service.
-   **Web Dashboard**: A simple web interface could be built with Flask or FastAPI to view the live feed and alert history.
-   **Low-Light Enhancement**: An image pre-processing step could be added to improve model performance in low-light or infrared conditions.
-   **Advanced Alerting**: Integrate with other services (e.g., IFTTT, Home Assistant) for more complex alerting and automation.

---
*This project demonstrates the power of ensemble learning in practical, real-world applications. By moving beyond single-model solutions, we unlock a new level of reliability and performance.*
