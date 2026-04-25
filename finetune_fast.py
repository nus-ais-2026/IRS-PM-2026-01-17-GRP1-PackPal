# finetune_fast.py — START NOW, RESULTS IN ~30 MIN
"""
Quick training for immediate results.
Trains YOLOv8 Nano on DeepFashion2 for clothing detection.
"""

from ultralytics import YOLO
import os
import time

os.environ["OMP_NUM_THREADS"] = "1"

def main():
    start = time.time()
    
    print("🚀 Starting quick training (YOLOv8 Nano)...")
    print("   Expected: ~30 minutes")
    
    model = YOLO("yolov8n-seg.pt")
    
    model.train(
        data="deepfashion2-m-10k-2/data.yaml",
        epochs=10,
        imgsz=320,
        batch=16,
        lr0=0.001,
        patience=5,
        pretrained=True,
        project="packpal_runs",
        name="fashion_seg_nano",     # ← Different name to not conflict
        device="mps",
        workers=0,
        cache=False,
        amp=False,
        exist_ok=True,
    )
    
    elapsed = (time.time() - start) / 60
    print(f"\n✅ Training complete in {elapsed:.0f} minutes")
    print(f"   Model saved to: packpal_runs/fashion_seg_nano/weights/best.pt")

if __name__ == "__main__":
    main()