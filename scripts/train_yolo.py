"""Train the YOLO11n plate detector (needs dataset/data.yaml, a Roboflow export; not in git).
The result used by the app is runs/detect/runs/vehicle_plate/weights/best.pt."""
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":

    # YOLO11n architecture - training from scratch
    model = YOLO("yolo11n.yaml")

    model.train(
        data=str(ROOT / "dataset" / "data.yaml"),
        epochs=100,
        imgsz=640,
        batch=4,
        device=0,
        workers=2,
        patience=20,
        pretrained=False,
        project=str(ROOT / "runs" / "detect" / "runs"),
        name="vehicle_plate",
        plots=True
    )

    print("Training completed!")
    print("Best model: runs/detect/runs/vehicle_plate/weights/best.pt")
