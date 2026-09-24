import torch
import numpy as np
import pickle
import cv2
import threading
import time
import math
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor
from trajectory import TrackingConfig, ByteTracker, VisualMotion, qualify_track, pane_bounds
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
from config import DEVICE, COCO_CLASSES_PATH, MODELS_CONFIG, TRACKING_CONFIG

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
        self.candidate_threshold = self.confidence_threshold
        self.collect_all = False
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
            image = Image.fromarray(cv2.cvtColor(img_enh, cv2.COLOR_BGR2RGB))
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            target_sizes = torch.tensor([image.size[::-1]])
            results = self.processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=self.candidate_threshold
            )[0]

            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                if self.model.config.id2label[label.item()] == "person":
                    with results_lock:
                        results_list.append(float(score.item() * self.weight))
                        box_list.append((box.tolist(), self.key))
                    if not self.collect_all:
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
            image = Image.fromarray(cv2.cvtColor(img_enh, cv2.COLOR_BGR2RGB))
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            target_sizes = torch.tensor([image.size[::-1]])
            results = self.processor.post_process_object_detection(
                outputs, threshold=self.candidate_threshold, target_sizes=target_sizes
            )[0]

            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                label_name = self.model.config.id2label.get(label.item(), "")
                if label_name == "person" or "person" in label_name.lower():
                    with results_lock:
                        results_list.append(float(score.item() * self.weight))
                        box_list.append((box.tolist(), self.key))
                    if not self.collect_all:
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
            image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            target_sizes = torch.tensor([image.size[::-1]])
            results = self.processor.post_process_object_detection(
                outputs, threshold=self.candidate_threshold, target_sizes=target_sizes
            )[0]

            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                if self.model.config.id2label[label.item()] == "person":
                    with results_lock:
                        results_list.append(float(score.item() * self.weight))
                        box_list.append((box.tolist(), self.key))
                    if not self.collect_all:
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
            if hasattr(self.model, "roi_heads"):
                self.model.roi_heads.score_thresh = self.candidate_threshold
            elif hasattr(self.model, "score_thresh"):
                self.model.score_thresh = self.candidate_threshold
            detections = self.model(tensor)[0]

            boxes = detections["boxes"]
            scores = detections["scores"]
            labels = detections["labels"]

            for i in range(len(boxes)):
                confidence = scores[i].item()
                if confidence > self.candidate_threshold:
                    idx = int(labels[i].item())
                    if idx < len(CLASSES) and "person" in CLASSES[idx]:
                        box = boxes[i].detach().cpu().numpy()
                        with results_lock:
                            results_list.append(float(confidence * self.weight))
                            box_list.append((box, self.key))
                        if not self.collect_all:
                            break

class YOLOv5Detector(BaseDetector):
    """YOLOv5 model handler."""
    def __init__(self, key, config_dict, device):
        super().__init__(key, config_dict, device)
        if "color" not in config_dict:
            self.color = (0, 165, 255)  # Orange
        self.model = torch.hub.load("ultralytics/yolov5", self.name).to(self.device)

    def run(self, img: np.ndarray, results_list: list, box_list: list) -> None:
        with torch.no_grad():
            self.model.conf = self.candidate_threshold
            detections = self.model(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            detections_ary = detections.pandas().xyxy[0].to_numpy()
            for i in detections_ary:
                if i[-1] == "person" and i[-3] > self.candidate_threshold:
                    with results_lock:
                        results_list.append(float(i[-3] * self.weight))
                        box_list.append((i[:4], self.key))
                    if not self.collect_all:
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
            res = self.model(img, device=self.device_str, conf=self.candidate_threshold, verbose=False)[0]
            for box in res.boxes:
                conf = float(box.conf[0].item())
                if conf > self.candidate_threshold:
                    cls_id = int(box.cls[0].item())
                    label_name = res.names.get(cls_id, "")
                    if label_name == "person" or "person" in str(label_name).lower():
                        xyxy = box.xyxy[0].cpu().numpy()
                        with results_lock:
                            results_list.append(float(conf * self.weight))
                            box_list.append((xyxy.tolist(), self.key))
                        if not self.collect_all:
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
    def __init__(self, models_config=None, device=None, tracking_config=None):
        self.device = device if device is not None else DEVICE
        self.models_config = models_config if models_config is not None else MODELS_CONFIG
        self.tracking_config = TrackingConfig(**(TRACKING_CONFIG if tracking_config is None else tracking_config))
        self.trackers = {}
        self.preview_boxes = []
        self.stream_generation = 0
        self.visual_motion = VisualMotion(self.tracking_config)
        self.pane_scores = {}
        self._frame_shape = None
        self._last_frame_time = None
        self._inference_lock = threading.Lock()
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

    def reset_tracking(self):
        """Call when changing input source or seeking a video."""
        self.trackers.clear()
        self.preview_boxes = []
        self.stream_generation += 1
        self.visual_motion = VisualMotion(self.tracking_config)
        self.pane_scores = {}
        self._frame_shape = None
        self._last_frame_time = None

    def run_inference(self, img: np.ndarray, execution_mode: str = "sequential"):
        # Models and trajectory state are shared by the legacy singleton callers.
        with self._inference_lock:
            return self._run_frame(img, execution_mode)

    def _run_frame(self, img, execution_mode):
        cfg = self.tracking_config
        now = time.monotonic()
        if (self._frame_shape != img.shape[:2] or
                (self._last_frame_time is not None and now - self._last_frame_time > cfg.reset_gap_seconds)):
            self.reset_tracking()
        self._frame_shape = img.shape[:2]
        self._last_frame_time = now
        self.pane_scores = {}
        panes = list(pane_bounds(img.shape, cfg)) if cfg.enabled else [(0, 0, 0, img.shape[1], img.shape[0])]
        all_boxes = []
        scores_by_pane = {}

        def run_detector(detector):
            detector.collect_all = cfg.enabled
            detector.candidate_threshold = detector.confidence_threshold
            scores, boxes = [], []
            detector.run(img, scores, boxes)
            return detector, scores, boxes

        # Exactly one full-frame inference per model. Panes only partition detections.
        inference_start = time.perf_counter()
        if execution_mode == "parallel":
            with ThreadPoolExecutor(max_workers=max(1, len(self.detectors))) as executor:
                outputs = list(executor.map(run_detector, self.detectors))
        else:
            outputs = [run_detector(d) for d in self.detectors]
        self.preview_boxes = [
            (list(box), model) for detector, scores, boxes in outputs
            for score, (box, model) in zip(scores, boxes)
            if detector.weight > 0 and score / detector.weight > detector.confidence_threshold
        ]
        tracking_start = time.perf_counter()
        if cfg.enabled and cfg.require_visual_motion:
            self.visual_motion.update(img, panes)
        routed = {}
        if cfg.enabled:
            height, width = img.shape[:2]
            x_edges = [col * width // cfg.columns for col in range(cfg.columns + 1)]
            y_edges = [row * height // cfg.rows for row in range(cfg.rows + 1)]
            for detector, scores, boxes in outputs:
                grouped = {pane: [] for pane, *_ in panes}
                routed[detector.key] = grouped
                for score, (box, _) in zip(scores, boxes):
                    box = np.asarray(box, dtype=float)
                    if (box.shape != (4,) or detector.weight <= 0 or
                            not math.isfinite(score) or not all(math.isfinite(v) for v in box)):
                        continue
                    # Original model cutoff is a hard gate, before history or boosts.
                    raw_confidence = score / detector.weight
                    if raw_confidence <= detector.confidence_threshold:
                        continue
                    left, top, right, bottom = map(float, box)
                    left, right = max(0., min(left, width)), max(0., min(right, width))
                    top, bottom = max(0., min(top, height)), max(0., min(bottom, height))
                    if right <= left or bottom <= top:
                        continue
                    col = bisect_right(x_edges, (left + right) / 2) - 1
                    row = bisect_right(y_edges, (top + bottom) / 2) - 1
                    pane = row * cfg.columns + col
                    _, x1, y1, x2, y2 = panes[pane]
                    # A seam-straddling box belongs only to its center's pane.
                    local = np.array([max(left, x1) - x1, max(top, y1) - y1,
                                      min(right, x2) - x1, min(bottom, y2) - y1])
                    grouped[pane].append((local, raw_confidence))
        for pane, x1, y1, x2, y2 in panes:
            pane_results = []
            for detector, scores, boxes in outputs:
                if not cfg.enabled:
                    pane_results.extend(scores)
                    all_boxes.extend(boxes)
                    continue
                tracker = self.trackers.setdefault((pane, detector.key), ByteTracker(cfg))
                qualified_scores = []
                for track in tracker.update(routed[detector.key][pane]):
                    fraction = (self.visual_motion.fraction(pane, track.last_box, track.previous_box)
                                if cfg.require_visual_motion else 0.0)
                    if qualify_track(track, cfg, detector.confidence_threshold, fraction):
                        qualified_scores.append(track.score * detector.weight * cfg.score_multiplier)
                        global_box = track.last_box + np.array([x1, y1, x1, y1])
                        all_boxes.append((global_box.tolist(), detector.key))
                # Crowd size cannot inflate a model's ensemble vote.
                if qualified_scores:
                    pane_results.append(max(qualified_scores))
            scores_by_pane[pane] = pane_results
            self.pane_scores[pane] = sum(pane_results)
        self.last_timings = {
            "inference_seconds": tracking_start - inference_start,
            "tracking_seconds": time.perf_counter() - tracking_start,
        }
        # Unrelated cameras cannot manufacture ensemble agreement.
        best = max(scores_by_pane, key=lambda p: sum(scores_by_pane[p]))
        self._last_frame_time = time.monotonic()
        return scores_by_pane[best], all_boxes

    def draw_trajectories(self, img):
        """Draw grid and bounded observed histories without modifying tracker state."""
        cfg = self.tracking_config
        if not cfg.enabled or not cfg.draw_history:
            return
        for pane, x1, y1, x2, y2 in pane_bounds(img.shape, cfg):
            cv2.rectangle(img, (x1, y1), (x2 - 1, y2 - 1), (180, 180, 180), 1)
            cv2.putText(img, f"Pane {pane + 1}: {self.pane_scores.get(pane, 0):.2f}",
                        (x1 + 6, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
            # Drawing on the crop clips all trails/labels to their own camera pane.
            crop = img[y1:y2, x1:x2]
            for (track_pane, model), tracker in self.trackers.items():
                if track_pane != pane:
                    continue
                for track in tracker.tracks:
                    if track.missed:
                        continue
                    ready = track.score_eligible
                    color = (0, 220, 0) if ready else (0, 180, 255)
                    points = []
                    for _, box in track.history:
                        a, b, c, d = map(int, box)
                        cv2.rectangle(crop, (a, b), (c, d), color, 1)
                        points.append(((a + c) // 2, (b + d) // 2))
                    if len(points) > 1:
                        cv2.polylines(crop, [np.array(points, np.int32)], False, color, 1)
                    x, y = map(int, track.last_box[:2])
                    label = 'QUALIFIED' if ready else 'candidate'
                    cv2.putText(crop, f"{model} #{track.id} {label}",
                                (x, max(30, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, .35, color, 1)

    def get_model_colors(self) -> dict[str, tuple]:
        """Returns mapping of model keys to their display BGR colors."""
        return {detector.key: detector.color for detector in self.detectors}

# Singleton pipeline instance
_pipeline = None
_pipeline_lock = threading.Lock()

def get_model_pipeline(models_config=None, device=None, reload=False, tracking_config=None) -> ModelPipeline:
    """Returns singleton ModelPipeline instance."""
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None or reload:
            _pipeline = ModelPipeline(models_config=models_config, device=device, tracking_config=tracking_config)
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

# Resolve legacy model attributes lazily, retaining their import/access behavior.
_LEGACY_MODEL_NAMES = {
    "detr_model", "detr_processor", "yolos_model", "yolos_processor",
    "frcnn_model", "retinanet_model", "yolov5_model",
}


def __getattr__(name):
    if name in _LEGACY_MODEL_NAMES:
        _init_legacy_globals()
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _init_legacy_globals():
    global detr_model, detr_processor, yolos_model, yolos_processor, frcnn_model, retinanet_model, yolov5_model
    p = get_model_pipeline()
    for name in _LEGACY_MODEL_NAMES:
        globals()[name] = None
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
