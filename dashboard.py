import time
from flask import Flask, Response
import cv2
import threading
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from transformers import AutoModelForImageSegmentation
from dashboard_main import main as detection_thread
from config import REMOVE_BACKGROUND

app = Flask(__name__)

# Global variable to hold the latest frame
latest_frame = None
frame_lock = threading.Lock()

# Background removal model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
bg_model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4",trust_remote_code=True).to(device)

def preprocess_image(im: np.ndarray, model_input_size: list) -> torch.Tensor:
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]
    im_tensor = torch.tensor(im, dtype=torch.float32).permute(2,0,1)
    im_tensor = F.interpolate(torch.unsqueeze(im_tensor,0), size=model_input_size, mode='bilinear')
    image = torch.divide(im_tensor,255.0)
    image = normalize(image,[0.5,0.5,0.5],[1.0,1.0,1.0])
    return image

def postprocess_image(result: torch.Tensor, im_size: list)-> np.ndarray:
    result = torch.squeeze(F.interpolate(result, size=im_size, mode='bilinear') ,0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result-mi)/(ma-mi)
    im_array = (result*255).permute(1,2,0).cpu().data.numpy().astype(np.uint8)
    im_array = np.squeeze(im_array)
    return im_array

def update_frame(frame):
    """Callback function to update the latest frame from the detection thread."""
    global latest_frame
    with frame_lock:
        latest_frame = frame.copy()

def generate_frames():
    """Generator function to yield frames for the web feed."""
    global latest_frame
    while True:
        frame_to_encode = None
        
        with frame_lock:
            if latest_frame is not None:
                frame_to_encode = latest_frame.copy()

        if frame_to_encode is not None:
            if REMOVE_BACKGROUND:
                orig_im = cv2.cvtColor(frame_to_encode, cv2.COLOR_BGR2RGB)
                orig_im_size = orig_im.shape[0:2]
                model_input_size = [1024, 1024]
                image = preprocess_image(orig_im, model_input_size).to(device)

                with torch.no_grad():
                    result = bg_model(image)

                result_image = postprocess_image(result[0][0], orig_im_size)
                
                pil_mask_im = Image.fromarray(result_image)
                orig_image = Image.fromarray(orig_im)
                no_bg_image = orig_image.copy()
                no_bg_image.putalpha(pil_mask_im)

                frame_to_encode = cv2.cvtColor(np.array(no_bg_image), cv2.COLOR_RGBA2BGRA)
                (flag, encodedImage) = cv2.imencode(".png", frame_to_encode)
                content_type = "image/png"
            else:
                (flag, encodedImage) = cv2.imencode(".jpg", frame_to_encode)
                content_type = "image/jpeg"

            if not flag:
                continue
            
            yield (
                b"--frame\r\n"
                b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + bytearray(encodedImage) + b"\r\n"
            )
        
        time.sleep(0.016)

@app.route("/")
def index():
    """Video streaming home page."""
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    # Start the object detection thread and pass the update_frame callback
    det_thread = threading.Thread(target=detection_thread, args=(update_frame,))
    det_thread.daemon = True
    det_thread.start()
    print("Detection thread started")

    # Start the Flask app
    # use_reloader=False prevents Flask from spawning duplicate processes
    # which would create multiple camera instances
    app.run(debug=True, host="0.0.0.0", port=8080, use_reloader=False)
