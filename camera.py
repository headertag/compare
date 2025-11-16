import cv2
import threading
import queue
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

    def start(self):
        """Start the camera capture thread if not already running."""
        import traceback
        import sys

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

            print("[CAMERA] Initializing camera...")
            self.cam = cv2.VideoCapture(CAM_INDEX)
            self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
            self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

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
        """Internal camera reader thread."""
        while self.running:
            if self.cam is None:
                break

            ret, frame = self.cam.read()
            if not ret:
                print("Failed to grab frame from camera.")
                break

            # Always keep only the latest frame in the queue
            if not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()  # Discard old frame
                except queue.Empty:
                    pass
            self.frame_queue.put(frame)

        print("Camera reader thread stopped.")


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
