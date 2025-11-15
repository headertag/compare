import torch
import numpy as np
import pickle
import cv2
import threading
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection, YolosImageProcessor, YolosForObjectDetection
from torchvision.models import detection
from config import DEVICE, COCO_CLASSES_PATH, MODELS_CONFIG

def load_coco_classes(path):
    """Loads COCO class labels from a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)

CLASSES = load_coco_classes(COCO_CLASSES_PATH)
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Thread-safe lock for shared list access
results_lock = threading.Lock()

# --- Model Loading ---

def load_detr_model():
    """Loads the DETR model and processor."""
    processor = DetrImageProcessor.from_pretrained(MODELS_CONFIG["detr_resnet_50"]["name"])
    model = DetrForObjectDetection.from_pretrained(MODELS_CONFIG["detr_resnet_50"]["name"]).to(DEVICE)
    return model, processor

def load_yolos_model():
    """Loads the YOLOS model and processor."""
    processor = YolosImageProcessor.from_pretrained(MODELS_CONFIG["yolos_tiny"]["name"])
    model = YolosForObjectDetection.from_pretrained(MODELS_CONFIG["yolos_tiny"]["name"]).to(DEVICE)
    return model, processor

def load_torchvision_model(name):
    """Loads a torchvision model (Faster R-CNN or RetinaNet)."""
    model_class = {
        "frcnn-resnet": detection.fasterrcnn_resnet50_fpn,
        "retinanet": detection.retinanet_resnet50_fpn
    }[name]
    model = model_class(pretrained=True, progress=True, num_classes=len(CLASSES), pretrained_backbone=True).to(DEVICE)
    model.eval()
    return model

def load_yolov5_model():
    """Loads the YOLOv5 model."""
    model = torch.hub.load("ultralytics/yolov5", MODELS_CONFIG["yolov5s"]["name"])
    return model

# Load all models
detr_model, detr_processor = load_detr_model()
yolos_model, yolos_processor = load_yolos_model()
frcnn_model = load_torchvision_model(MODELS_CONFIG["frcnn_resnet"]["name"])
retinanet_model = load_torchvision_model(MODELS_CONFIG["retinanet"]["name"])
yolov5_model = load_yolov5_model()


# --- Model Inference ---

def enhance_low_light(img):
    """
    Placeholder for a low-light image enhancement model.
    A real implementation would use a dedicated model (e.g., a GAN)
    to improve the brightness and contrast of the image before detection.
    """
    # For now, just return the original image
    return img

def run_detr(img, results_list, box_list):
    """Runs inference using the DETR model."""
    with torch.no_grad():
        img = enhance_low_light(img)
        image = Image.fromarray(img)
        inputs = detr_processor(images=image, return_tensors="pt").to(DEVICE)
        outputs = detr_model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = detr_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=MODELS_CONFIG["detr_resnet_50"]["confidence_threshold"])[0]

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if detr_model.config.id2label[label.item()] == "person":
                with results_lock:
                    results_list.append(float(score.item() / 2))
                    box_list.append(box.tolist())
                break

def run_yolos(img, results_list, box_list):
    """Runs inference using the YOLOS model."""
    with torch.no_grad():
        image = Image.fromarray(img)
        inputs = yolos_processor(images=image, return_tensors="pt").to(DEVICE)
        outputs = yolos_model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = yolos_processor.post_process_object_detection(outputs, threshold=MODELS_CONFIG["yolos_tiny"]["confidence_threshold"], target_sizes=target_sizes)[0]

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if yolos_model.config.id2label[label.item()] == "person":
                with results_lock:
                    results_list.append(float(score.item() / 2))
                    box_list.append(box.tolist())
                break

def run_torchvision_model(model, img, results_list, box_list, confidence_threshold):
    """Runs inference using a torchvision model."""
    with torch.no_grad():
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame = frame.transpose((2, 0, 1))
        frame = np.expand_dims(frame, axis=0)
        frame = frame / 255.0
        frame = torch.FloatTensor(frame).to(DEVICE)
        detections = model(frame)[0]

        for i in range(len(detections["boxes"])):
            confidence = detections["scores"][i]
            if confidence > confidence_threshold:
                idx = int(detections["labels"][i])
                if "person" in CLASSES[idx]:
                    box = detections["boxes"][i].detach().cpu().numpy()
                    with results_lock:
                        results_list.append(float(confidence))
                        box_list.append(box)
                    break

def run_yolov5(img, results_list, box_list):
    """Runs inference using the YOLOv5 model."""
    with torch.no_grad():
        detections = yolov5_model(img)
        detections_ary = detections.pandas().xyxy[0].to_numpy()
        for i in detections_ary:
            if i[-1] == "person" and i[-3] > MODELS_CONFIG["yolov5s"]["confidence_threshold"]:
                with results_lock:
                    results_list.append(float(i[-3]))
                    # YOLOv5 doesn't provide a single box, so we don't append to box_list
                break
