from ultralytics import YOLO
import os

# 데이터셋 경로 및 모델 저장 경로
DATASET_PATH = os.getenv('DATASET_PATH', '../datasets/gonggu-v1')
MODEL_DIR = os.getenv('MODEL_DIR', '../models')
MODEL_NAME = 'gonggu-yolov8.pt'

# YOLO 모델 학습
model = YOLO('yolov8n.pt')
results = model.train(data=DATASET_PATH, epochs=50, imgsz=640)

# 모델 저장
model.save(f'{MODEL_DIR}/{MODEL_NAME}')
print(f"Model saved to {MODEL_DIR}/{MODEL_NAME}")
