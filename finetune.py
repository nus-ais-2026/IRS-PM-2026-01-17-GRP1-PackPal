# finetune.py
from ultralytics import YOLO

def main():
    model = YOLO("yolov8m-seg.pt")

    model.train(
        data="deepfashion2-m-10k-2/data.yaml",  # folder name after download
        epochs=50,
        imgsz=640,
        batch=8,
        lr0=0.001,
        patience=10,
        pretrained=True,
        project="packpal_runs",
        name="fashion_seg",
        device="mps",        # Apple Silicon
        workers=4,
    )

if __name__ == "__main__":
    main()