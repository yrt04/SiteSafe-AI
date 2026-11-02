"""Standalone OpenCV webcam viewer for local YOLOv11 PPE detection debugging (Q: quit, S: snapshot)."""

import argparse
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

DEFAULT_MODEL = (
    "runs/detect/train/weights/best.pt"
    if Path("runs/detect/train/weights/best.pt").exists()
    else "yolo11n.pt"
)
CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.35
WINDOW_NAME = "SiteSafe AI - YOLOv11 Detection"
TARGET_FPS = 30
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

PALETTE = [
    (255, 56, 56), (56, 255, 56), (56, 56, 255), (255, 157, 56),
    (255, 56, 157), (56, 255, 157), (157, 56, 255), (157, 255, 56),
    (56, 157, 255), (255, 255, 56), (255, 56, 255), (56, 255, 255),
    (128, 128, 0), (128, 0, 128), (0, 128, 128), (128, 64, 0),
    (64, 0, 128), (0, 128, 64), (255, 200, 100), (100, 200, 255),
]


class ThreadedCamera:
    """Decouples OpenCV camera reads from the inference loop to avoid DirectShow/V4L stalls."""

    def __init__(
        self,
        camera_index: int = CAMERA_INDEX,
        width: int = FRAME_WIDTH,
        height: int = FRAME_HEIGHT,
        fps: int = TARGET_FPS,
    ):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            return

        # Force MJPG + single-frame driver buffer so read() always returns the latest frame.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.ret = False
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._capture_worker, daemon=True)
        self.thread.start()

        for _ in range(50):
            if self.frame is not None:
                break
            time.sleep(0.02)

    def isOpened(self) -> bool:
        return self.cap is not None and self.cap.isOpened() and self.running

    def _capture_worker(self):
        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
                    self.ret = ret
            else:
                time.sleep(0.002)

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def get(self, prop_id):
        if self.cap:
            return self.cap.get(prop_id)
        return 0

    def release(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=0.5)
        if self.cap:
            self.cap.release()
            self.cap = None
        self.frame = None


def get_color(class_id: int) -> tuple:
    return PALETTE[class_id % len(PALETTE)]


def draw_detections(frame, results):
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = result.names[cls_id]
            color = get_color(cls_id)

            label = f"{cls_name} {conf:.0%}"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(frame, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                frame,
                label,
                (x1 + 2, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return frame


def overlay_stats(frame, fps: float, device_label: str):
    stats = f"FPS: {fps:.1f} | Device: {device_label}"
    cv2.putText(
        frame, stats, (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2, cv2.LINE_AA,
    )
    return frame


def main():
    parser = argparse.ArgumentParser(description="SiteSafe AI Real-time Webcam Detection")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"YOLO model file (default: {DEFAULT_MODEL})")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX, help=f"Camera index (default: {CAMERA_INDEX})")
    parser.add_argument("--fps", type=int, default=TARGET_FPS, help=f"Target frame rate (default: {TARGET_FPS})")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
        device_label = torch.cuda.get_device_name(0)
    else:
        device = "cpu"
        device_label = "CPU"

    print(f"[INFO] Compute device: {device_label}")
    print(f"[INFO] Loading weights: {args.model}")
    model = YOLO(args.model)
    model.to(device)
    print(f"[INFO] Model ready ({len(model.names)} classes)")

    # Pre-warm CUDA kernels before opening the display window.
    dummy = torch.zeros((1, 3, 640, 640), device=device)
    try:
        model(dummy, verbose=False)
    except Exception:
        pass

    cap = ThreadedCamera(camera_index=args.camera, width=FRAME_WIDTH, height=FRAME_HEIGHT, fps=args.fps)
    if not cap.isOpened():
        print("[ERROR] Failed to open webcam. Verify camera index.", file=sys.stderr)
        sys.exit(1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera initialized ({actual_w}x{actual_h} @ {args.fps} FPS target)")
    print("[INFO] Controls: [Q] quit | [S] snapshot")

    fps = 0.0
    screenshots_dir = Path("screenshots")
    target_interval = 1.0 / args.fps
    next_frame_time = time.perf_counter()

    try:
        while True:
            t_start = time.perf_counter()

            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.002)
                continue

            results = model.predict(
                source=frame,
                conf=CONFIDENCE_THRESHOLD,
                verbose=False,
                device=device,
            )

            annotated = draw_detections(frame, results)
            annotated = overlay_stats(annotated, fps, device_label)
            cv2.imshow(WINDOW_NAME, annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("s"), ord("S")):
                screenshots_dir.mkdir(exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = screenshots_dir / f"detection_{ts}.jpg"
                cv2.imwrite(str(path), annotated)
                print(f"[SNAP] Saved -> {path}")

            # Hybrid sleep + spin-wait compensates for coarse (~15.6ms) Windows sleep granularity.
            next_frame_time += target_interval
            delay = next_frame_time - time.perf_counter()
            if delay > 0.012:
                time.sleep(delay - 0.003)
            while time.perf_counter() < next_frame_time:
                pass

            dt = time.perf_counter() - t_start
            fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 1e-6))

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
