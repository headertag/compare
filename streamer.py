import threading
import time
import os
import cv2
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

class PreviewBroadcaster:
    """
    Thread-safe broadcaster for the latest processed frame.
    Stores frame in memory and exports to /dev/shm/preview.jpg.
    Uses Condition variables for zero-waste, change-driven streaming.
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.latest_jpeg = None
        self.latest_frame = None
        self.last_update_time = 0
        self.frame_id = 0
        self.shm_path = "/dev/shm/preview.jpg"
        self.local_path = "preview.jpg"

    def update_frame(self, frame, results=None, threshold=2.3, multi_box=None, status_text="Monitoring", target_width=1920, model_colors=None):
        """Update current frame, draw candidate bounding boxes, and encode lightweight preview JPEG."""
        orig_h, orig_w = frame.shape[:2]

        # Decouple preview resolution from 2K inference resolution for fast, low-latency streaming
        if orig_w > target_width:
            scale = target_width / float(orig_w)
            target_h = int(orig_h * scale)
            preview_img = cv2.resize(frame, (target_width, target_h), interpolation=cv2.INTER_LINEAR)
        else:
            scale = 1.0
            preview_img = frame.copy()

        h, w = preview_img.shape[:2]

        # Color palette
        color_green = (0, 230, 0)
        color_red = (0, 0, 255)
        color_cyan = (255, 255, 0)
        color_white = (255, 255, 255)

        score = sum(results) if results else 0.0
        is_alert = score >= threshold

        # Default fallback color mapping
        default_model_colors = {
            "detr": (0, 0, 255),       # Red
            "yolos": (255, 100, 0),    # Cyan/Blue
            "frcnn": (0, 255, 255),    # Yellow
            "retinanet": (255, 0, 255), # Magenta
            "yolov5": (0, 165, 255),   # Orange
        }
        active_model_colors = {**default_model_colors, **(model_colors or {})}

        # Draw all candidate bounding boxes discovered across active models
        if multi_box:
            for i, item in enumerate(multi_box):
                if len(item) == 2:
                    box_coords, model_name = item
                else:
                    box_coords, model_name = item[0], "detect"

                # Scale coordinates to preview dimensions
                startX = int(box_coords[0] * scale)
                startY = int(box_coords[1] * scale)
                endX = int(box_coords[2] * scale)
                endY = int(box_coords[3] * scale)

                box_color = color_green if is_alert else active_model_colors.get(model_name, color_cyan)
                cv2.rectangle(preview_img, (startX, startY), (endX, endY), box_color, 2)

                label = f"{model_name}"
                cv2.putText(preview_img, label, (startX, max(startY - 6, 16)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

        # Draw status overlay banner at top
        banner_h = 36
        overlay = preview_img.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (18, 18, 18), -1)
        cv2.addWeighted(overlay, 0.75, preview_img, 0.25, 0, preview_img)

        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
        status_color = color_red if is_alert else color_green
        alert_status = "⚠️ ALERT TRIGGERED" if is_alert else "LIVE MONITOR"

        # Status text: score, threshold, timestamp, and native resolution
        left_text = f"[{alert_status}] Score: {score:.2f} / {threshold:.2f}  |  {timestamp_str}  |  {orig_w}x{orig_h}"
        cv2.putText(preview_img, left_text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_white, 1)
        cv2.circle(preview_img, (w - 20, 18), 7, status_color, -1)

        # Encode to lightweight JPEG (quality 75, ~25KB per frame)
        ret, jpeg = cv2.imencode('.jpg', preview_img, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if ret:
            jpeg_bytes = jpeg.tobytes()
            with self.condition:
                self.latest_jpeg = jpeg_bytes
                self.latest_frame = preview_img
                self.last_update_time = time.time()
                self.frame_id += 1
                self.condition.notify_all()

            # Atomic save to /dev/shm (RAM buffer) and local preview.jpg
            try:
                tmp_shm = self.shm_path + ".tmp"
                with open(tmp_shm, "wb") as f:
                    f.write(jpeg_bytes)
                os.replace(tmp_shm, self.shm_path)
            except Exception:
                pass

    def get_jpeg(self):
        with self.lock:
            return self.latest_jpeg

    def get_jpeg_wait(self, last_seen_id=None, timeout=2.0):
        """Wait for a new frame or timeout. Returns (jpeg_bytes, current_frame_id)."""
        with self.condition:
            if last_seen_id is None or self.frame_id != last_seen_id:
                return self.latest_jpeg, self.frame_id
            self.condition.wait(timeout=timeout)
            return self.latest_jpeg, self.frame_id

_broadcaster = PreviewBroadcaster()

def get_broadcaster():
    return _broadcaster

class StreamingHandler(BaseHTTPRequestHandler):
    """HTTP handler serving MJPEG streams and snapshot JPEGs."""
    timeout = 10

    def log_message(self, format, *args):
        # Suppress standard HTTP request logging
        return

    def do_HEAD(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()

    def do_GET(self):
        broadcaster = get_broadcaster()

        if self.path in ('/', '/index.html'):
            html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Camera Monitor Preview</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background-color: #0f1117;
            color: #e6edf3;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
            padding: 16px;
        }
        header {
            width: 100%;
            max-width: 1400px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 20px;
            background: #161b22;
            border-radius: 8px;
            margin-bottom: 16px;
            border: 1px solid #30363d;
        }
        h1 { font-size: 1.25rem; font-weight: 600; display: flex; align-items: center; gap: 8px; }
        .live-dot { width: 10px; height: 10px; border-radius: 50%; background: #238636; box-shadow: 0 0 8px #238636; display: inline-block; }
        .container {
            width: 100%;
            max-width: 1400px;
            background: #000;
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid #30363d;
            box-shadow: 0 8px 24px rgba(0,0,0,0.5);
            display: flex;
            justify-content: center;
            align-items: center;
        }
        img.stream {
            width: 100%;
            height: auto;
            max-height: 82vh;
            object-fit: contain;
            display: block;
        }
        footer {
            margin-top: 12px;
            font-size: 0.85rem;
            color: #8b949e;
        }
    </style>
</head>
<body>
    <header>
        <h1><span class="live-dot"></span> Camera Object Detection Monitor</h1>
        <div>Source: <code>/dev/video0</code> &bull; Feed: <code>/stream</code></div>
    </header>
    <div class="container">
        <img class="stream" src="/stream" alt="Live Camera Preview">
    </div>
    <footer>
        Direct snapshot available at <a href="/preview.jpg" target="_blank" style="color: #58a6ff;">/preview.jpg</a> | Jetson Orin Nano
    </footer>
</body>
</html>"""
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(html.encode('utf-8'))))
            self.end_headers()
            self.wfile.write(html.encode('utf-8'))

        elif self.path in ('/stream', '/video_feed'):
            self.send_response(200)
            self.send_header('Age', '0')
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()

            current_id = None
            try:
                while True:
                    frame, current_id = broadcaster.get_jpeg_wait(last_seen_id=current_id, timeout=2.0)
                    if frame is not None:
                        self.wfile.write(b'--FRAME\r\n')
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', str(len(frame)))
                        self.end_headers()
                        self.wfile.write(frame)
                        self.wfile.write(b'\r\n')
            except (OSError, ConnectionError, Exception):
                pass

        elif self.path in ('/preview.jpg', '/latest.jpg', '/snapshot.jpg'):
            frame = broadcaster.get_jpeg()
            if frame is not None:
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Length', str(len(frame)))
                self.send_header('Cache-Control', 'no-cache')
                self.end_headers()
                self.wfile.write(frame)
            else:
                self.send_error(404, "No frame available yet")

        elif self.path == '/alert.jpg':
            if os.path.exists("ALERT.jpg"):
                with open("ALERT.jpg", "rb") as f:
                    data = f.read()
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_error(404, "No alert triggered yet")
        else:
            self.send_error(404, "Not Found")

def start_preview_server(host="0.0.0.0", port=8080):
    """Starts the background threaded HTTP preview server."""
    try:
        server = ThreadedHTTPServer((host, port), StreamingHandler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        print(f"[STREAMER] 🌐 Live monitor preview server running at http://{host}:{port}/")
        return server
    except Exception as e:
        print(f"[STREAMER] ⚠️ Could not start preview server on {host}:{port}: {e}")
        return None

if __name__ == "__main__":
    from camera import get_camera_manager
    print("[STREAMER] Starting standalone camera live preview server...")
    cam = get_camera_manager()
    cam.start()
    
    server = start_preview_server(host="0.0.0.0", port=8080)
    broadcaster = get_broadcaster()
    
    print("[STREAMER] Live camera feed available at: http://0.0.0.0:8080/")
    print("[STREAMER] Press Ctrl+C to stop.")
    
    try:
        while True:
            frame = cam.get_frame()
            if frame is not None:
                broadcaster.update_frame(frame, status_text="Live Camera Preview")
            time.sleep(0.03)  # ~30 FPS
    except KeyboardInterrupt:
        print("\n[STREAMER] Stopping stream server...")
    finally:
        cam.stop()
