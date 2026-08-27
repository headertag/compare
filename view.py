#!/usr/bin/env python3
"""
Lightweight Local Screen Monitor Viewer
Reads directly from shared RAM buffer (/dev/shm/preview.jpg) or HTTP stream.
Ultra-low memory footprint (~20MB RAM) and near-zero latency.
"""

import os
import sys
import time
import argparse
import cv2

def run_viewer(source="/dev/shm/preview.jpg", window_name="2K Camera Monitor Preview", fps=30, fullscreen=False):
    # Check for DISPLAY environment variable
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        print("⚠️ No graphical display environment found in current shell session.")
        print("\n💡 Options to view the live monitor:")
        print("  1. If viewing from the Jetson's physically connected HDMI monitor:")
        print("     DISPLAY=:0 /home/march/compare/venv/bin/python view.py")
        print("  2. If using feh (ultra-lightweight X11 image viewer):")
        print("     DISPLAY=:0 feh --reload 0.1 --geometry 1280x720 /dev/shm/preview.jpg")
        print("  3. If viewing from a laptop/phone/remote machine on the same network:")
        print("     Open http://192.168.2.148:8080/ in your browser")
        return

    print(f"🎬 Starting lightweight monitor viewer from: {source}")
    print("⌨️  Controls: Press 'q' or ESC to quit, 'f' to toggle fullscreen")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(window_name, 1280, 720)

    delay = max(1, int(1000.0 / fps))
    is_fullscreen = fullscreen
    last_mtime = 0

    try:
        while True:
            # If reading from shared memory file
            if source.startswith("/dev/shm/") or os.path.exists(source):
                try:
                    current_mtime = os.path.getmtime(source)
                    if current_mtime != last_mtime:
                        frame = cv2.imread(source)
                        if frame is not None:
                            cv2.imshow(window_name, frame)
                            last_mtime = current_mtime
                except Exception:
                    pass
            elif source.startswith("http"):
                # If reading from HTTP stream URL
                cap = cv2.VideoCapture(source)
                while cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        cv2.imshow(window_name, frame)
                    key = cv2.waitKey(delay) & 0xFF
                    if key in (27, ord('q'), ord('Q')):
                        cap.release()
                        return
                cap.release()

            key = cv2.waitKey(delay) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break
            elif key in (ord('f'), ord('F')):
                is_fullscreen = not is_fullscreen
                prop = cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, prop)
                if not is_fullscreen:
                    cv2.resizeWindow(window_name, 1280, 720)

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        print("Viewer closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lightweight local camera monitor preview viewer")
    parser.add_argument("--source", default="/dev/shm/preview.jpg", help="Path to shared RAM preview image or stream URL")
    parser.add_argument("--fps", type=int, default=30, help="Target display refresh rate (default: 30)")
    parser.add_argument("--fullscreen", action="store_true", help="Launch in fullscreen mode")
    args = parser.parse_args()

    run_viewer(source=args.source, fps=args.fps, fullscreen=args.fullscreen)
