"""Fine-tune YOLOv11-Nano on the Ultralytics Construction-PPE dataset."""

from ultralytics import YOLO


def main() -> None:
    model = YOLO("yolo11n.pt")
    results = model.train(data="construction-ppe.yaml", epochs=50, imgsz=640)
    print(f"Training complete. Best weights saved under: {results.save_dir}")


if __name__ == "__main__":
    # Required on Windows so spawn-based PyTorch DataLoader workers do not re-execute module top-level.
    main()
