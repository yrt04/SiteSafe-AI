"""WebSocket inference server for real-time PPE compliance monitoring."""

import argparse
import asyncio
import base64
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO
import websockets
from websockets.asyncio.server import serve

DEFAULT_PORT = 8765
DEFAULT_CAMERA = 0
DEFAULT_MODEL = (
    "runs/detect/train/weights/best.pt"
    if Path("runs/detect/train/weights/best.pt").exists()
    else "yolo11n.pt"
)
CONFIDENCE_THRESHOLD = 0.35
JPEG_QUALITY = 75
MAX_FPS = 30
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

RECORDINGS_DIR = Path("recordings")
SCREENSHOTS_DIR = Path("screenshots")

PALETTE = [
    (255, 56, 56), (56, 255, 56), (56, 56, 255), (255, 157, 56),
    (255, 56, 157), (56, 255, 157), (157, 56, 255), (157, 255, 56),
    (56, 157, 255), (255, 255, 56), (255, 56, 255), (56, 255, 255),
    (128, 128, 0), (128, 0, 128), (0, 128, 128), (128, 64, 0),
    (64, 0, 128), (0, 128, 64), (255, 200, 100), (100, 200, 255),
]

class ThreadedCamera:
    """Decouples OpenCV frame capture from the inference loop to avoid camera I/O stalls."""

    def __init__(
        self,
        camera_index: int,
        width: int = FRAME_WIDTH,
        height: int = FRAME_HEIGHT,
        fps: int = MAX_FPS,
    ):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.frame = None
        self.ret = False

    def start(self) -> bool:
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            return False

        # Force MJPG + single-frame driver buffer so read() never serves stale queued frames.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

        for _ in range(50):
            if self.frame is not None:
                break
            time.sleep(0.02)
        return True

    def isOpened(self) -> bool:
        return self.cap is not None and self.cap.isOpened() and self.running

    def _worker(self):
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


class VideoStreamer:
    """Coordinates camera lifecycle, YOLO inference, violation cooldowns, and WebSocket fan-out."""

    def __init__(self, camera_index: int, model_name: str):
        self.camera_index = camera_index
        self.model_name = model_name
        self.clients: set[websockets.ServerConnection] = set()

        self.capture_task = None
        self.streaming = False
        self.fps = 0.0
        self.model = None
        self.cam = None
        self.device = "cpu"

        self.is_recording = False
        self.video_writer = None
        self.recording_filename = ""
        self.recording_start_time = None
        self.frames_recorded = 0

        self.latest_annotated_frame = None
        self.total_detections = 0
        self.session_start = datetime.now()

    def _init_model(self):
        if self.model is not None:
            return

        if torch.cuda.is_available():
            self.device = "cuda"
            device_label = torch.cuda.get_device_name(0)
        else:
            self.device = "cpu"
            device_label = "CPU"

        print(f"[INFO] Compute device: {device_label}")
        print(f"[INFO] Loading weights: {self.model_name}")
        self.model = YOLO(self.model_name)
        self.model.to(self.device)
        print(f"[INFO] Model ready ({len(self.model.names)} classes)")

        # Pre-warm CUDA kernels with a dummy 640x640 tensor to avoid a latency spike on frame 1.
        dummy = torch.zeros((1, 3, 640, 640), device=self.device)
        try:
            self.model(dummy, verbose=False)
        except Exception:
            pass

    def _init_camera(self):
        if self.cam is not None and self.cam.isOpened():
            return

        self.cam = ThreadedCamera(self.camera_index, FRAME_WIDTH, FRAME_HEIGHT, MAX_FPS)
        if not self.cam.start():
            print("[ERROR] Failed to open webcam.", file=sys.stderr)
            return

        w = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[INFO] Camera initialized ({w}x{h} @ {MAX_FPS} FPS)")

    def _get_color(self, class_id: int) -> tuple:
        return PALETTE[class_id % len(PALETTE)]

    def _annotate_frame(self, frame, results) -> list[dict]:
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                if conf < CONFIDENCE_THRESHOLD:
                    continue

                cls_name = result.names[cls_id]
                color = self._get_color(cls_id)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label = f"{cls_name} {conf:.0%}"
                (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(frame, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, -1)
                cv2.putText(
                    frame,
                    label,
                    (x1 + 2, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                detections.append({
                    "class": cls_name,
                    "confidence": round(conf, 3),
                    "bbox": [x1, y1, x2, y2],
                })

        return detections

    def _encode_frame(self, frame) -> str:
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return base64.b64encode(buffer).decode("utf-8")

    def start_recording(self) -> dict:
        if self.is_recording:
            return {
                "type": "recording_status",
                "recording": True,
                "filename": self.recording_filename,
                "message": "Already recording",
            }

        if not self.streaming or self.cam is None or not self.cam.isOpened():
            return {
                "type": "recording_status",
                "recording": False,
                "filename": "",
                "message": "Cannot record, feed is not active.",
            }

        RECORDINGS_DIR.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recording_filename = f"recording_{ts}.avi"
        filepath = RECORDINGS_DIR / self.recording_filename

        w = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.video_writer = cv2.VideoWriter(str(filepath), fourcc, 30.0, (w, h))

        self.is_recording = True
        self.recording_start_time = datetime.now()
        self.frames_recorded = 0

        print(f"[REC] Started -> {filepath}")
        return {
            "type": "recording_status",
            "recording": True,
            "filename": self.recording_filename,
            "message": "Recording started",
        }

    def stop_recording(self) -> dict:
        if not self.is_recording:
            return {
                "type": "recording_status",
                "recording": False,
                "filename": "",
                "message": "Not recording",
            }

        self.is_recording = False
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

        duration = (
            (datetime.now() - self.recording_start_time).total_seconds()
            if self.recording_start_time
            else 0
        )
        print(f"[REC] Stopped {self.recording_filename} ({self.frames_recorded} frames, {duration:.1f}s)")

        return {
            "type": "recording_status",
            "recording": False,
            "filename": self.recording_filename,
            "frames": self.frames_recorded,
            "duration": round(duration, 1),
            "message": "Recording saved",
        }

    def take_snapshot(self) -> dict:
        if self.latest_annotated_frame is None:
            return {"type": "snapshot_saved", "success": False, "message": "No frame available"}

        SCREENSHOTS_DIR.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"snapshot_{ts}.jpg"
        filepath = SCREENSHOTS_DIR / filename

        cv2.imwrite(str(filepath), self.latest_annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"[SNAP] Saved -> {filepath}")

        _, buffer = cv2.imencode(".jpg", self.latest_annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        b64 = base64.b64encode(buffer).decode("utf-8")

        return {
            "type": "snapshot_saved",
            "success": True,
            "filename": filename,
            "data": b64,
            "message": f"Saved {filename}",
        }

    async def _broadcast(self, message: str):
        if not self.clients:
            return
        # Snapshot client set before awaiting I/O so concurrent disconnects cannot mutate iteration.
        disconnected = set()
        for client in self.clients.copy():
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
        self.clients -= disconnected

    async def _send_to(self, websocket, data: dict):
        try:
            await websocket.send(json.dumps(data))
        except websockets.exceptions.ConnectionClosed:
            pass

    async def handle_client(self, websocket: websockets.ServerConnection):
        self.clients.add(websocket)
        remote = websocket.remote_address
        print(f"[CONN] Client connected: {remote} ({len(self.clients)} active)")

        await self._send_to(websocket, {"type": "feed_status", "streaming": self.streaming})

        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    cmd = data.get("type", "")

                    if cmd == "ping":
                        await self._send_to(websocket, {"type": "pong"})
                    elif cmd == "start_feed":
                        if not self.streaming:
                            self.capture_task = asyncio.create_task(self.capture_loop())
                            await self._broadcast(
                                json.dumps({"type": "feed_status", "streaming": True, "message": "Feed started"})
                            )
                    elif cmd == "stop_feed":
                        if self.streaming:
                            self.stop()
                            await self._broadcast(
                                json.dumps({"type": "feed_status", "streaming": False, "message": "Feed stopped"})
                            )
                    elif cmd == "start_recording":
                        await self._broadcast(json.dumps(self.start_recording()))
                    elif cmd == "stop_recording":
                        await self._broadcast(json.dumps(self.stop_recording()))
                    elif cmd == "snapshot":
                        await self._send_to(websocket, self.take_snapshot())

                except json.JSONDecodeError:
                    pass
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)
            print(f"[DISC] Client disconnected: {remote} ({len(self.clients)} active)")

            # Release the hardware camera lock once the last subscriber disconnects.
            if not self.clients and self.streaming:
                print("[INFO] Last client disconnected; suspending capture loop.")
                self.stop()

    async def capture_loop(self):
        self._init_model()
        self._init_camera()

        if self.cam is None or not self.cam.isOpened():
            await self._broadcast(
                json.dumps({"type": "feed_status", "streaming": False, "message": "Failed to open camera"})
            )
            return

        self.streaming = True
        self.session_start = datetime.now()

        min_interval = 1.0 / MAX_FPS
        fps_smooth = 0.0
        next_frame_time = time.perf_counter()

        print(f"[INFO] Stream active (target {MAX_FPS} FPS)")

        try:
            while self.streaming:
                t_start = time.perf_counter()

                ret, frame = self.cam.read()
                if not ret or frame is None:
                    await asyncio.sleep(0.002)
                    continue

                results = self.model.predict(
                    source=frame,
                    conf=CONFIDENCE_THRESHOLD,
                    verbose=False,
                    device=self.device,
                )
                detections = self._annotate_frame(frame, results)
                self.latest_annotated_frame = frame.copy()

                if self.is_recording and self.video_writer:
                    self.video_writer.write(frame)
                    self.frames_recorded += 1

                self.total_detections += len(detections)

                frame_b64 = self._encode_frame(frame)
                await self._broadcast(json.dumps({
                    "type": "frame",
                    "camera_id": "cam-1",
                    "data": frame_b64,
                    "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                    "detections": detections,
                    "fps": round(fps_smooth, 1),
                    "detection_count": len(detections),
                    "total_detections": self.total_detections,
                    "is_recording": self.is_recording,
                    "uptime_seconds": round((datetime.now() - self.session_start).total_seconds()),
                }))

                # Hybrid sleep + spin-wait compensates for ~15.6ms Windows timer granularity before yielding to asyncio.
                next_frame_time += min_interval
                delay = next_frame_time - time.perf_counter()
                if delay > 0.012:
                    time.sleep(delay - 0.003)
                while time.perf_counter() < next_frame_time:
                    pass
                await asyncio.sleep(0)

                dt = time.perf_counter() - t_start
                fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / max(dt, 1e-6))
                self.fps = fps_smooth

        except asyncio.CancelledError:
            print("[INFO] Capture loop cancelled.")
        finally:
            if self.is_recording:
                self.stop_recording()
            if self.cam:
                self.cam.release()
                self.cam = None
            self.streaming = False
            self.latest_annotated_frame = None
            print("[INFO] Camera released.")

    def stop(self):
        self.streaming = False
        if self.capture_task:
            self.capture_task.cancel()
            self.capture_task = None


async def main(port: int, camera: int, model: str):
    streamer = VideoStreamer(camera_index=camera, model_name=model)

    async with serve(
        streamer.handle_client,
        "0.0.0.0",
        port,
        ping_interval=20,
        ping_timeout=20,
        max_size=10 * 1024 * 1024,
    ):
        print(f"[INFO] WebSocket server listening on ws://0.0.0.0:{port}")
        print(f"[INFO] Local endpoint: ws://localhost:{port} (Ctrl+C to exit)\n")

        try:
            await asyncio.Future()
        except KeyboardInterrupt:
            streamer.stop()

    print("[INFO] Server shut down.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SiteSafe AI WebSocket Video Server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"WebSocket port (default: {DEFAULT_PORT})")
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA, help=f"Camera index (default: {DEFAULT_CAMERA})")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"YOLO model file (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    try:
        asyncio.run(main(port=args.port, camera=args.camera, model=args.model))
    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")
