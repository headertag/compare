# 📷 Camera & HDMI-to-USB Capture Card Troubleshooting Guide

This guide covers troubleshooting steps for USB cameras and HDMI-to-USB capture cards (including 4K capture dongles such as Cam Link 4K, ShadowCast, and MS2109/MS2130/MS2131-based devices) on Linux and NVIDIA Jetson Orin.

---

## 🔍 Quick Diagnosis

We have included a diagnostic script in the repository to automatically detect and test all connected video feeds:

```bash
/home/march/compare/venv/bin/python test_camera.py
```

---

## 🛠️ Step-by-Step Troubleshooting

### 1. Check Hardware Detection & USB Ports
4K capture cards and high-bandwidth HDMI-to-USB dongles draw significant power and require high-speed USB 3.0 lines:
- **Use Direct USB 3.0 Ports:** Plug directly into a blue/USB 3.2 port on the Jetson Orin Nano, not through an unpowered USB hub.
- **Check Detection:** Verify that the system sees the USB device:
  ```bash
  lsusb
  ```
  Look for devices labeled with USB Video, Cam Link, MacroSilicon, or generic video capture devices.

### 2. Load the UVC Kernel Driver
On Linux/Jetson, USB cameras require the `uvcvideo` kernel module. If not loaded automatically, load it with:
```bash
sudo modprobe uvcvideo
```

To ensure `uvcvideo` is automatically loaded on boot:
```bash
echo "uvcvideo" | sudo tee -a /etc/modules
```

### 3. Identify the Correct Video Device Index
Once the camera is plugged in and the driver is loaded, check which `/dev/video*` devices are available:
```bash
ls -l /dev/video*
```
*Note: HDMI capture cards often expose multiple nodes (e.g., `/dev/video0` for the video stream and `/dev/video1` for metadata/controls).*

You can also run:
```bash
python test_camera.py
```
This script will test each index and save a snapshot `test_capture_video<N>.jpg` for each working device.

### 4. Update `config.yaml`
Edit `config.yaml` to specify the correct camera index:

```yaml
camera:
  # Set to the working index discovered from test_camera.py (e.g. 0, 1, or 2)
  index: 0
  width: 1920
  height: 1080
```

### 5. Codec & Resolution Handling (MJPG)
Uncompressed 4K or 1080p video over USB 2/3 (YUYV format) can cause frame drops, bandwidth saturation, or failure to negotiate stream parameters. The codebase in `camera.py` automatically requests `MJPG` format for high resolution and low latency.

### 6. Restart the Service
If running the background service, restart it after applying your configuration:
```bash
sudo systemctl restart camera-alert
```

View live logs to confirm frames are being processed:
```bash
journalctl -u camera-alert -f
```
