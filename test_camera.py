import cv2
import glob
import sys
import os

def check_devices():
    devices = sorted(glob.glob('/dev/video*'))
    print("=" * 50)
    print("VIDEO DEVICE DIAGNOSTICS")
    print("=" * 50)
    
    if not devices:
        print("❌ No /dev/video* devices found!")
        print("\nTroubleshooting steps:")
        print("1. Run: sudo modprobe uvcvideo")
        print("2. Unplug and replug the USB HDMI capture card.")
        print("3. Check `lsusb` to see if the USB hardware is detected.")
        return False
    
    print(f"Found video nodes: {devices}")
    
    working_indices = []
    for dev in devices:
        try:
            idx = int(dev.replace('/dev/video', ''))
        except ValueError:
            continue
            
        print(f"\nTesting {dev} (Index {idx})...")
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            print(f"  ❌ Could not open {dev}")
            cap.release()
            continue
            
        # Test common resolutions
        resolutions_to_test = [
            ("2.5K / 1440p", 2560, 1440),
            ("2K DCI", 2048, 1080),
            ("1080p", 1920, 1080),
            ("4K (UHD)", 3840, 2160),
            ("720p", 1280, 720)
        ]
        
        cap_works = False
        for label, req_w, req_h in resolutions_to_test:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, req_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req_h)
            
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            ret, frame = cap.read()
            if ret and frame is not None:
                actual_h, actual_w = frame.shape[:2]
                print(f"  ✅ [{label}] Requested {req_w}x{req_h} -> Negotiated: {w}x{h} (Actual Frame: {actual_w}x{actual_h}) @ {fps:.1f} FPS")
                if not cap_works:
                    cv2.imwrite(f"test_capture_video{idx}.jpg", frame)
                    print(f"     Saved sample frame to test_capture_video{idx}.jpg")
                    cap_works = True
            else:
                print(f"  ⚠️ [{label}] {req_w}x{req_h} failed to capture.")

        if cap_works:
            working_indices.append(idx)
        else:
            print(f"  ⚠️ Opened {dev}, but failed to read frames across test resolutions.")
            
        cap.release()
        
    print("\n" + "=" * 50)
    if working_indices:
        print(f"🎉 Working camera index found: {working_indices[0]}")
        print(f"Set 'index: {working_indices[0]}' in config.yaml")
        return True
    else:
        print("❌ No working camera feeds detected.")
        return False

if __name__ == '__main__':
    check_devices()
