import torch
import numpy as np
import pickle
import cv2
import threading
from PIL import Image
from transformers import (
    DetrImageProcessor,
    DetrForObjectDetection,
    YolosImageProcessor,
    YolosForObjectDetection,
    AutoImageProcessor,
    AutoModelForObjectDetection,
)
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

def enhance_low_light(img):
    """
    Placeholder for low-light image enhancement.
    Returns the original image.
    """
    return img

# --- Base Detector & Specific Detectors ---

class BaseDetector:
    """Base detector interface for any object detection model."""
    def __init__(self, key, config_dict, device):
        self.key = key
        self.name = config_dict.get("name", key)
        self.confidence_threshold = float(config_dict.get("confidence_threshold", 0.5))
        self.weight = float(config_dict.get("weight", 1.0))
        self.color = tuple(config_dict.get("color", (0, 255, 255)))
        self.device = device
        self.model = None

    def run(self, img: np.ndarray, results_list: list, box_list: list) -> None:
        raise NotImplementedError

class DetrDetector(BaseDetector):
    """DETR (DEtection TRansformer) model handler."""
    def __init__(self, key, config_dict, device):
        super().__init__(key, config_dict, device)
        if "weight" not in config_dict:
            self.weight = 0.5
        if "color" not in config_dict:
            self.color = (0, 0, 255)  # Red
        self.processor = DetrImageProcessor.from_pretrained(self.name)
        self.model = DetrForObjectDetection.from_pretrained(self.name).to(self.device)
        self.model.eval()

    def run(self, img: np.ndarray, results_list: list, box_list: list) -> None:
        with torch.no_grad():
            img_enh = enhance_low_light(img)
            image = Image.fromarray(img_enh)
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            target_sizes = torch.tensor([image.size[::-1]])
            results = self.processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=self.confidence_threshold
            )[0]

            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                if self.model.config.id2label[label.item()] == "person":
                    with results_lock:
                        results_list.append(float(score.item() * self.weight))
                        box_list.append((box.tolist(), self.key))
                    break

class RfDetrDetector(BaseDetector):
    """RF-DETR (Roboflow) model handler."""
    def __init__(self, key, config_dict, device):
        super().__init__(key, config_dict, device)
        if "weight" not in config_dict:
            self.weight = 0.5
        if "color" not in config_dict:
            self.color = (180, 105, 255)  # Hot Pink
        self.processor = AutoImageProcessor.from_pretrained(self.name)
        self.model = AutoModelForObjectDetection.from_pretrained(self.name).to(self.device)
        self.model.eval()

    def run(self, img: np.ndarray, results_list: list, box_list: list) -> None:
        with torch.no_grad():
            img_enh = enhance_low_light(img)
            image = Image.fromarray(img_enh)
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            target_sizes = torch.tensor([image.size[::-1]])
            results = self.processor.post_process_object_detection(
                outputs, threshold=self.confidence_threshold, target_sizes=target_sizes
            )[0]

            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                label_name = self.model.config.id2label.get(label.item(), "")
                if label_name == "person" or "person" in label_name.lower():
                    with results_lock:
                        results_list.append(float(score.item() * self.weight))
                        box_list.append((box.tolist(), self.key))
                    break

class YolosDetector(BaseDetector):
    """YOLOS (You Only Look at One Sequence) model handler."""
    def __init__(self, key, config_dict, device):
        super().__init__(key, config_dict, device)
        if "weight" not in config_dict:
            self.weight = 0.5
        if "color" not in config_dict:
            self.color = (255, 100, 0)  # Cyan/Blue
        self.processor = YolosImageProcessor.from_pretrained(self.name)
        self.model = YolosForObjectDetection.from_pretrained(self.name).to(self.device)
        self.model.eval()

    def run(self, img: np.ndarray, results_list: list, box_list: list) -> None:
        with torch.no_grad():
            image = Image.fromarray(img)
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            target_sizes = torch.tensor([image.size[::-1]])
            results = self.processor.post_process_object_detection(
                outputs, threshold=self.confidence_threshold, target_sizes=target_sizes
            )[0]

            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                if self.model.config.id2label[label.item()] == "person":
                    with results_lock:
                        results_list.append(float(score.item() * self.weight))
                        box_list.append((box.tolist(), self.key))
                    break

class TorchvisionDetector(BaseDetector):
    """Torchvision model handler (Faster R-CNN, RetinaNet)."""
    def __init__(self, key, config_dict, device):
        super().__init__(key, config_dict, device)
        if "color" not in config_dict:
            if "frcnn" in self.name.lower() or "frcnn" in key.lower():
                self.color = (0, 255, 255)  # Yellow
            else:
                self.color = (255, 0, 255)  # Magenta

        name_lower = self.name.lower()
        if "frcnn" in name_lower or "faster" in name_lower:
            model_fn = detection.fasterrcnn_resnet50_fpn
        elif "retina" in name_lower:
            model_fn = detection.retinanet_resnet50_fpn
        else:
            model_fn = detection.fasterrcnn_resnet50_fpn

        self.model = model_fn(pretrained=True, progress=True, num_classes=len(CLASSES), pretrained_backbone=True).to(self.device)
        self.model.eval()

    def run(self, img: np.ndarray, results_list: list, box_list: list) -> None:
        with torch.no_grad():
            frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to(self.device, non_blocking=True).float().div_(255.0)
            detections = self.model(tensor)[0]

            boxes = detections["boxes"]
            scores = detections["scores"]
            labels = detections["labels"]

            for i in range(len(boxes)):
                confidence = scores[i].item()
                if confidence > self.confidence_threshold:
                    idx = int(labels[i].item())
                    if idx < len(CLASSES) and "person" in CLASSES[idx]:
                        box = boxes[i].detach().cpu().numpy()
                        with results_lock:
                            results_list.append(float(confidence * self.weight))
                            box_list.append((box, self.key))
                        break

class YOLOv5Detector(BaseDetector):
    """YOLOv5 model handler."""
    def __init__(self, key, config_dict, device):
        super().__init__(key, config_dict, device)
        if "color" not in config_dict:
            self.color = (0, 165, 255)  # Orange
        self.model = torch.hub.load("ultralytics/yolov5", self.name)

    def run(self, img: np.ndarray, results_list: list, box_list: list) -> None:
        with torch.no_grad():
            detections = self.model(img)
            detections_ary = detections.pandas().xyxy[0].to_numpy()
            for i in detections_ary:
                if i[-1] == "person" and i[-3] > self.confidence_threshold:
                    with results_lock:
                        results_list.append(float(i[-3] * self.weight))
                        box_list.append((i[:4], self.key))
                    break

class UltralyticsDetector(BaseDetector):
    """Ultralytics modern YOLO (YOLO11, YOLOv8, RT-DETR) handler."""
    def __init__(self, key, config_dict, device):
        super().__init__(key, config_dict, device)
        if "color" not in config_dict:
            self.color = (50, 205, 50)  # Bright Lime Green
        from ultralytics import YOLO
        self.device_str = "cuda" if device.type == "cuda" else "cpu"
        self.model = YOLO(self.name)

    def run(self, img: np.ndarray, results_list: list, box_list: list) -> None:
        with torch.no_grad():
            res = self.model(img, device=self.device_str, verbose=False)[0]
            for box in res.boxes:
                conf = float(box.conf[0].item())
                if conf > self.confidence_threshold:
                    cls_id = int(box.cls[0].item())
                    label_name = res.names.get(cls_id, "")
                    if label_name == "person" or "person" in str(label_name).lower():
                        xyxy = box.xyxy[0].cpu().numpy()
                        with results_lock:
                            results_list.append(float(conf * self.weight))
                            box_list.append((xyxy.tolist(), self.key))
                        break

# --- Model Factory & Dynamic Pipeline ---

def create_detector(key: str, config_dict: dict, device: torch.device) -> BaseDetector:
    """Factory function creating appropriate detector instance from configuration."""
    model_type = config_dict.get("type", "").lower()
    name_str = config_dict.get("name", key).lower()
    key_str = key.lower()

    if model_type in ("ultralytics", "yolo11", "yolov8", "rtdetr") or "yolo11" in key_str or "yolov8" in key_str or "yolo11" in name_str or "yolov8" in name_str:
        return UltralyticsDetector(key, config_dict, device)
    elif model_type == "rf_detr" or "rf_detr" in key_str or "rf-detr" in key_str or "rf_detr" in name_str or "rf-detr" in name_str:
        return RfDetrDetector(key, config_dict, device)
    elif model_type == "detr" or ("detr" in key_str and "rf" not in key_str and "rf" not in model_type):
        return DetrDetector(key, config_dict, device)
    elif model_type == "yolos" or "yolos" in key_str:
        return YolosDetector(key, config_dict, device)
    elif model_type == "torchvision" or "frcnn" in key_str or "retina" in key_str:
        return TorchvisionDetector(key, config_dict, device)
    elif model_type in ("yolov5", "yolo") or "yolov5" in key_str:
        return YOLOv5Detector(key, config_dict, device)
    else:
        return DetrDetector(key, config_dict, device)

class ModelPipeline:
    """
    Dynamic object detection pipeline managing N arbitrary models.
    Supports dynamic loading, enable/disable toggling, and configurable sequential/parallel execution.
    """
    def __init__(self, models_config=None, device=None):
        self.device = device if device is not None else DEVICE
        self.models_config = models_config if models_config is not None else MODELS_CONFIG
        self.detectors = []
        self._load_models()

    def _load_models(self):
        """Loads all enabled models defined in models_config."""
        self.detectors = []
        for key, conf in self.models_config.items():
            if not isinstance(conf, dict):
                continue
            if not conf.get("enabled", True):
                print(f"[MODEL_PIPELINE] ⏩ Skipping disabled model: '{key}'")
                continue
            print(f"[MODEL_PIPELINE] 📦 Loading model: '{key}' ({conf.get('name', key)})...")
            try:
                detector = create_detector(key, conf, self.device)
                self.detectors.append(detector)
            except Exception as e:
                print(f"[MODEL_PIPELINE] ⚠️ Failed to load model '{key}': {e}")
        print(f"[MODEL_PIPELINE] ✅ Loaded {len(self.detectors)} active model(s).")

    def run_inference(self, img: np.ndarray, execution_mode: str = "sequential"):
        """Runs inference across all active models in parallel or sequentially."""
        results = []
        multi_box = []

        if not self.detectors:
            return results, multi_box

        if execution_mode == "parallel":
            threads = [
                threading.Thread(target=detector.run, args=(img, results, multi_box))
                for detector in self.detectors
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        else:
            for detector in self.detectors:
                detector.run(img, results, multi_box)

        return results, multi_box

    def get_model_colors(self) -> dict[str, tuple]:
        """Returns mapping of model keys to their display BGR colors."""
        return {detector.key: detector.color for detector in self.detectors}

# Singleton pipeline instance
_pipeline = None
_pipeline_lock = threading.Lock()

def get_model_pipeline(models_config=None, device=None, reload=False) -> ModelPipeline:
    """Returns singleton ModelPipeline instance."""
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None or reload:
            _pipeline = ModelPipeline(models_config=models_config, device=device)
        return _pipeline

# --- Legacy Backward Compatibility Exports ---

def load_detr_model():
    p = get_model_pipeline()
    for d in p.detectors:
        if isinstance(d, DetrDetector):
            return d.model, d.processor
    d = DetrDetector("detr_resnet_50", MODELS_CONFIG.get("detr_resnet_50", {"name": "facebook/detr-resnet-50"}), DEVICE)
    return d.model, d.processor

def load_yolos_model():
    p = get_model_pipeline()
    for d in p.detectors:
        if isinstance(d, YolosDetector):
            return d.model, d.processor
    d = YolosDetector("yolos_tiny", MODELS_CONFIG.get("yolos_tiny", {"name": "hustvl/yolos-tiny"}), DEVICE)
    return d.model, d.processor

def load_torchvision_model(name):
    d = TorchvisionDetector(name, {"name": name}, DEVICE)
    return d.model

def load_yolov5_model():
    d = YOLOv5Detector("yolov5s", MODELS_CONFIG.get("yolov5s", {"name": "yolov5s"}), DEVICE)
    return d.model

def run_detr(img, results_list, box_list):
    p = get_model_pipeline()
    for d in p.detectors:
        if isinstance(d, DetrDetector):
            d.run(img, results_list, box_list)
            return

def run_yolos(img, results_list, box_list):
    p = get_model_pipeline()
    for d in p.detectors:
        if isinstance(d, YolosDetector):
            d.run(img, results_list, box_list)
            return

def run_torchvision_model(model, img, results_list, box_list, confidence_threshold, model_name):
    p = get_model_pipeline()
    for d in p.detectors:
        if isinstance(d, TorchvisionDetector) and (d.key == model_name or d.name == model_name):
            d.run(img, results_list, box_list)
            return
    # Fallback direct execution
    d = TorchvisionDetector(model_name, {"name": model_name, "confidence_threshold": confidence_threshold}, DEVICE)
    d.model = model
    d.run(img, results_list, box_list)

def run_yolov5(img, results_list, box_list):
    p = get_model_pipeline()
    for d in p.detectors:
        if isinstance(d, YOLOv5Detector):
            d.run(img, results_list, box_list)
            return

# Expose model instances for backward compatibility
detr_model, detr_processor = None, None
yolos_model, yolos_processor = None, None
frcnn_model = None
retinanet_model = None
yolov5_model = None

def _init_legacy_globals():
    global detr_model, detr_processor, yolos_model, yolos_processor, frcnn_model, retinanet_model, yolov5_model
    p = get_model_pipeline()
    for d in p.detectors:
        if isinstance(d, DetrDetector):
            detr_model, detr_processor = d.model, d.processor
        elif isinstance(d, YolosDetector):
            yolos_model, yolos_processor = d.model, d.processor
        elif isinstance(d, TorchvisionDetector):
            if "frcnn" in d.key.lower():
                frcnn_model = d.model
            elif "retina" in d.key.lower():
                retinanet_model = d.model
        elif isinstance(d, YOLOv5Detector):
            yolov5_model = d.model

_init_legacy_globals()
