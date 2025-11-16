from flask import Flask, Response
import cv2
import threading
from dashboard_main import main as detection_thread

app = Flask(__name__)

# Global variable to hold the latest frame
latest_frame = None
frame_lock = threading.Lock()

def update_frame(frame):
    """Callback function to update the latest frame from the detection thread."""
    global latest_frame
    with frame_lock:
        latest_frame = frame.copy()

def generate_frames():
    """Generator function to yield frames for the web feed."""
    global latest_frame
    while True:
        with frame_lock:
            if latest_frame is not None:
                (flag, encodedImage) = cv2.imencode(".jpg", latest_frame)
                if not flag:
                    continue
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + bytearray(encodedImage) + b"\r\n"
                )

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