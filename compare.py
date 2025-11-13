import random
import cv2
from transformers import DetrImageProcessor, DetrForObjectDetection
from transformers import YolosImageProcessor, YolosForObjectDetection
import torch
from PIL import Image
from torchvision.models import detection
import numpy as np
import pickle
import time
from datetime import datetime
import telepot
import threading
import yaml

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# -- Configurable Parameters --
# Telegram
TELEGRAM_TOKEN = config["telegram"]["token"]
TELEGRAM_CHAT_IDS = config["telegram"]["chat_ids"]

# Alerting
ALERT_COOLDOWN = config["alerting"]["cooldown_seconds"]
ALERT_COOLDOWN_THRESHOLD = config["alerting"]["cooldown_threshold_multiplier"]
MIN_ALERT_INTERVAL = config["alerting"]["min_interval_seconds"]
ALERT_SENSITIVITY_THRESHOLD = config["alerting"]["sensitivity_threshold"]

# Camera
CAM_INDEX = config["camera"]["index"]
CAM_WIDTH = config["camera"]["width"]
CAM_HEIGHT = config["camera"]["height"]

# Processing
DEVICE = torch.device(config["processing"]["device"])
COCO_CLASSES_PATH = config["processing"]["coco_classes_path"]

# Models
MODELS_CONFIG = config["models"]

# -- End of Configurable Parameters --

TelegramBot = telepot.Bot(TELEGRAM_TOKEN)

CLASSES = pickle.loads(open(COCO_CLASSES_PATH, "rb").read())
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
MODELS = {
		"frcnn-resnet": detection.fasterrcnn_resnet50_fpn,
		"frcnn-mobilenet": detection.fasterrcnn_mobilenet_v3_large_320_fpn,
		"retinanet": detection.retinanet_resnet50_fpn
}

cam = cv2.VideoCapture(CAM_INDEX)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

processor = DetrImageProcessor.from_pretrained(MODELS_CONFIG["detr_resnet_50"]["name"])
model = DetrForObjectDetection.from_pretrained(MODELS_CONFIG["detr_resnet_50"]["name"]).to(DEVICE)

processor2 = YolosImageProcessor.from_pretrained(MODELS_CONFIG["yolos_tiny"]["name"])
model2 = YolosForObjectDetection.from_pretrained(MODELS_CONFIG["yolos_tiny"]["name"]).to(DEVICE)

model3 = MODELS[MODELS_CONFIG["frcnn_resnet"]["name"]](pretrained=True, progress=True,
        num_classes=len(CLASSES), pretrained_backbone=True).to(DEVICE)
model3.eval()

model4 = MODELS[MODELS_CONFIG["retinanet"]["name"]](pretrained=True, progress=True,
        num_classes=len(CLASSES), pretrained_backbone=True).to(DEVICE)
model4.eval()

model5 = torch.hub.load("ultralytics/yolov5", MODELS_CONFIG["yolov5s"]["name"])

def run_model(model_id, img, res, multi_box):
	ret_value = 0
	if model_id == 1:
		cv2.imwrite('model1.jpg',img)
		image = Image.open('model1.jpg')
		inputs = processor(images=image, return_tensors="pt")
		inputs = inputs.to(DEVICE)
		outputs = model(**inputs)

		target_sizes = torch.tensor([image.size[::-1]])
		results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=MODELS_CONFIG["detr_resnet_50"]["confidence_threshold"])[0]

		for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
			box = [round(i, 2) for i in box.tolist()]
			if model.config.id2label[label.item()] == "person":
				print("model",model_id,float(score.item()/2))
				ret_value = float(score.item()/2)
				multi_box.append( box )
				break

	if model_id == 2:
		cv2.imwrite('model2.jpg',img)
		image = Image.open('model2.jpg')
		inputs = processor2(images=image, return_tensors="pt")
		inputs = inputs.to(DEVICE)
		outputs = model2(**inputs)
		target_sizes = torch.tensor([image.size[::-1]])
		results = processor2.post_process_object_detection(outputs, threshold=MODELS_CONFIG["yolos_tiny"]["confidence_threshold"], target_sizes=target_sizes)[0]
		for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
			box = [round(i, 2) for i in box.tolist()]
			if model2.config.id2label[label.item()] == "person":
				print("model",model_id,float(score.item()/2))
				ret_value = float(score.item()/2)
				multi_box.append( box )
				break

	if model_id == 3:
		frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		frame = frame.transpose((2, 0, 1))
		frame = np.expand_dims(frame, axis=0)
		frame = frame / 255.0
		frame = torch.FloatTensor(frame)
		frame = frame.to(DEVICE)
		detections = model3(frame)[0]
		for i in range(0, len(detections["boxes"])):
			confidence = detections["scores"][i]
			if confidence > MODELS_CONFIG["frcnn_resnet"]["confidence_threshold"]:
				idx = int(detections["labels"][i])
				box = detections["boxes"][i].detach().cpu().numpy()
				(startX, startY, endX, endY) = box.astype("int")
				if "person" in CLASSES[idx]:
					print("model",model_id,float(confidence))
					ret_value = float(confidence)
					multi_box.append( box )
					break

	if model_id == 4:
		frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		frame = frame.transpose((2, 0, 1))
		frame = np.expand_dims(frame, axis=0)
		frame = frame / 255.0
		frame = torch.FloatTensor(frame)
		frame = frame.to(DEVICE)
		detections = model4(frame)[0]
		for i in range(0, len(detections["boxes"])):
			confidence = detections["scores"][i]
			if confidence > MODELS_CONFIG["retinanet"]["confidence_threshold"]:
				idx = int(detections["labels"][i])
				box = detections["boxes"][i].detach().cpu().numpy()
				(startX, startY, endX, endY) = box.astype("int")
				if "person" in CLASSES[idx]:
					print("model",model_id,float(confidence))
					ret_value = float(confidence)
					multi_box.append( box )
					break

	if model_id == 5:
		detections = model5(img)
		detections_ary = detections.pandas().xyxy[0].to_numpy()
		for i in detections_ary:
			if i[-1] == "person":
				if i[-3] > MODELS_CONFIG["yolov5s"]["confidence_threshold"]:
					print("model",model_id,float(i[-3]))
					ret_value = float(i[-3])
					break

	res.append( ret_value )

def send_alert():
    with open('ALERT.jpg', 'rb') as alert_image:
        for chat_id in TELEGRAM_CHAT_IDS:
            try:
                TelegramBot.sendPhoto(chat_id, alert_image)
            except Exception as e:
                print(f"Failed to send photo to {chat_id}: {e}")

if __name__ == "__main__":
	last_alert = 0

	while True:
		try:
			res = []
			multi_box = []
			s, img = cam.read()
			if not s:
				print("Failed to read frame from camera. Retrying...")
				time.sleep(1)
				continue

			x = threading.Thread(target=run_model, args=(1,img,res,multi_box))
			y = threading.Thread(target=run_model, args=(2,img,res,multi_box))
			z = threading.Thread(target=run_model, args=(3,img,res,multi_box))
			a = threading.Thread(target=run_model, args=(4,img,res,multi_box))
			b = threading.Thread(target=run_model, args=(5,img,res,multi_box))
			
			x.start()
			y.start()
			z.start()
			a.start()
			b.start()
			
			x.join()
			y.join()
			z.join()
			a.join()
			b.join()
			
			if sum(res) >= ALERT_SENSITIVITY_THRESHOLD:
				current_epoch = datetime.now().timestamp()
				time_delta = current_epoch - last_alert
				if current_epoch - last_alert > MIN_ALERT_INTERVAL:
					if time_delta/MIN_ALERT_INTERVAL < ALERT_COOLDOWN_THRESHOLD:
						time.sleep(ALERT_COOLDOWN)
					else:
						print("Alert", current_epoch,time_delta/MIN_ALERT_INTERVAL)
						last_alert = current_epoch
						if multi_box:
							for i in range(min(len(multi_box), 3)):
								box = multi_box[i]
								startX = int(box[0])
								startY = int(box[1])
								endX = int(box[2])
								endY = int(box[3])
								cv2.rectangle(img, (startX, startY), (endX, endY),(25*i,255,25*i), 2)
						cv2.imwrite('ALERT.jpg',img)
						alert = threading.Thread(target=send_alert, args=())
						alert.start()

						# after an alert use a new seed - useful to help scramble a bad prediction from becoming sequential
						torch.manual_seed( random.randint(1,3000000) )
		except Exception as e:
			print("ERROR",e)
