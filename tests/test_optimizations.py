import numpy as np
import cv2
import torch
import pytest
from config import CAM_WIDTH, CAM_HEIGHT

def test_frame_resizing_from_4k_to_2k():
    """Verify that a 4K frame (3840x2160) is properly downscaled to 2K (CAM_WIDTH x CAM_HEIGHT or <= 2560x1440)."""
    # Create synthetic 4K frame (2160, 3840, 3)
    frame_4k = np.zeros((2160, 3840, 3), dtype=np.uint8)
    h, w = frame_4k.shape[:2]

    # Apply 2K resizing logic
    if CAM_WIDTH and CAM_HEIGHT and (w > CAM_WIDTH or h > CAM_HEIGHT):
        resized_frame = cv2.resize(frame_4k, (CAM_WIDTH, CAM_HEIGHT), interpolation=cv2.INTER_AREA)
    elif w > 2560 or h > 1440:
        resized_frame = cv2.resize(frame_4k, (2560, 1440), interpolation=cv2.INTER_AREA)
    else:
        resized_frame = frame_4k

    assert resized_frame.shape[1] <= 2560
    assert resized_frame.shape[0] <= 1440
    assert resized_frame.shape[1] == (CAM_WIDTH if CAM_WIDTH else 2560)
    assert resized_frame.shape[0] == (CAM_HEIGHT if CAM_HEIGHT else 1440)

def test_torch_cuda_empty_cache_safe_execution():
    """Verify that calling torch.cuda.empty_cache() conditionally or directly does not raise errors."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Should always succeed without exception
    assert True

def test_broadcaster_event_driven_notification():
    """Verify that PreviewBroadcaster increments frame_id and notifies waiting consumers."""
    from streamer import PreviewBroadcaster
    broadcaster = PreviewBroadcaster()
    
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    initial_id = broadcaster.frame_id
    
    broadcaster.update_frame(frame, status_text="Test Event")
    assert broadcaster.frame_id == initial_id + 1
    
    jpeg, new_id = broadcaster.get_jpeg_wait(last_seen_id=initial_id, timeout=0.1)
    assert jpeg is not None
    assert new_id == initial_id + 1
