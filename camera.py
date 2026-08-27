import cv2
import threading
import queue
import time
import glob
from config import CAM_INDEX, CAM_WIDTH, CAM_HEIGHT

class CameraManager:
    """
    Singleton camera manager to ensure only one camera capture thread exists.
    Multiple consumers can get frames from the same camera source.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self.cam = None
        self.frame_queue = queue.Queue(maxsize=1)
        self.capture_thread = None
        self.running = False
        self.consumers = 0
        self.consumers_lock = threading.Lock()

    def _open_camera(self):
        """Helper to open camera and configure FourCC / resolution."""
        if isinstance(CAM_INDEX, int):
            self.cam = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
        else:
            self.cam = cv2.VideoCapture(CAM_INDEX)

        if not self.cam.isOpened():
            available = sorted(glob.glob('/dev/video*'))
            print(f"[CAMERA] ❌ Error: Failed to open camera device '{CAM_INDEX}'.")
            print(f"[CAMERA] Available video nodes in /dev/: {available if available else 'None found'}")
            print(f"[CAMERA] Run 'python test_camera.py' or 'sudo modprobe uvcvideo' to troubleshoot.")
            return False

        # Set FourCC to MJPG for 4K / 1080p HDMI-to-USB capture cards
        self.cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        return True

    def start(self):
        """Start the camera capture thread if not already running."""
        import traceback

        with self.consumers_lock:
            self.consumers += 1

            # DIAGNOSTIC: Show who called start()
            stack = traceback.extract_stack()
            caller_info = []
            for frame in stack[-4:-1]:  # Show last 3 frames before this one
                caller_info.append(f"{frame.filename}:{frame.lineno} in {frame.name}")

            print(f"\n[CAMERA] start() called by:")
            for info in caller_info:
                print(f"  {info}")
            print(f"[CAMERA] Consumer count: {self.consumers}")

            if self.running:
                print(f"[CAMERA] Camera already running. Total consumers: {self.consumers}\n")
                return

            print(f"[CAMERA] Initializing camera with index/source: {CAM_INDEX}...")
            self._open_camera()

            self.running = True
            self.capture_thread = threading.Thread(target=self._reader_thread)
            self.capture_thread.daemon = True
            self.capture_thread.start()
            print(f"[CAMERA] Camera reader thread started. Consumers: {self.consumers}\n")

    def stop(self):
        """Stop the camera capture thread when no more consumers."""
        import traceback

        with self.consumers_lock:
            self.consumers -= 1

            # DIAGNOSTIC: Show who called stop()
            stack = traceback.extract_stack()
            caller_info = []
            for frame in stack[-4:-1]:
                caller_info.append(f"{frame.filename}:{frame.lineno} in {frame.name}")

            print(f"\n[CAMERA] stop() called by:")
            for info in caller_info:
                print(f"  {info}")
            print(f"[CAMERA] Remaining consumers: {self.consumers}")

            if self.consumers > 0:
                print(f"[CAMERA] Camera kept alive for remaining consumers\n")
                return

            if not self.running:
                print(f"[CAMERA] Camera already stopped\n")
                return

            print("[CAMERA] Stopping camera reader thread...")
            self.running = False

            if self.capture_thread and self.capture_thread.is_alive():
                self.capture_thread.join(timeout=2)

            if self.cam:
                self.cam.release()
                self.cam = None

            print("[CAMERA] Camera released.\n")

    def get_frame(self):
        """Get the latest frame from the queue without blocking."""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None

    def _reader_thread(self):
        """Internal camera reader thread with auto-retry and reconnection."""
        consecutive_failures = 0
        while self.running:
            if self.cam is None or not self.cam.isOpened():
                time.sleep(1)
                if self.running:
                    self._open_camera()
                continue

            ret, frame = self.cam.read()
            if not ret or frame is None:
                consecutive_failures += 1
                if consecutive_failures % 10 == 1:
                    print(f"[CAMERA] ⚠️ Frame grab failed (consecutive retry #{consecutive_failures}). Waiting for video signal...")
                time.sleep(0.1)

                if consecutive_failures >= 30:
                    print("[CAMERA] 🔄 Signal lost or timed out. Re-opening camera device...")
                    if self.cam:
                        self.cam.release()
                        self.cam = None
                    time.sleep(1)
                    if self.running:
                        self._open_camera()
                    consecutive_failures = 0
                continue

            consecutive_failures = 0
            # Ensure frame size does not exceed 2K (or configured CAM_WIDTH/CAM_HEIGHT)
            if CAM_WIDTH and CAM_HEIGHT and (frame.shape[1] > CAM_WIDTH or frame.shape[0] > CAM_HEIGHT):
                frame = cv2.resize(frame, (CAM_WIDTH, CAM_HEIGHT), interpolation=cv2.INTER_AREA)
            elif frame.shape[1] > 2560 or frame.shape[0] > 1440:
                frame = cv2.resize(frame, (2560, 1440), interpolation=cv2.INTER_AREA)

            # Always keep only the latest frame in the queue
            if not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()  # Discard old frame
                except queue.Empty:
                    pass
            self.frame_queue.put(frame)

        print("[CAMERA] Camera reader thread stopped.")


# Singleton instance
_camera_manager = None

def get_camera_manager():
    """Get the singleton camera manager instance."""
    import traceback
    global _camera_manager
    if _camera_manager is None:
        print("\n[CAMERA] Creating NEW CameraManager singleton")
        stack = traceback.extract_stack()
        for frame in stack[-4:-1]:
            print(f"  {frame.filename}:{frame.lineno} in {frame.name}")
        _camera_manager = CameraManager()
    return _camera_manager


# Legacy compatibility functions
def initialize_camera():
    """Legacy function - now returns camera manager."""
    return get_camera_manager()

def camera_reader_thread(camera_object, frame_queue, running_flag):
    """Legacy function - deprecated, use CameraManager instead."""
    raise DeprecationWarning("Use CameraManager instead of direct camera_reader_thread")
