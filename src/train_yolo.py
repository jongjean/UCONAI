
from ultralytics import YOLO
import os

# 데이터셋 경로 및 모델 저장 경로
DATASET_PATH = os.getenv('DATASET_PATH', '../datasets/gonggu-v1')
MODEL_DIR = os.getenv('MODEL_DIR', '../models')
MODEL_NAME = 'gonggu-yolov8.pt'

# 데이터셋 구조 예시
# datasets/gonggu-v1/
#   images/  (학습 이미지)
#   labels/  (YOLO txt 라벨)
#   tool_info.json (공구 정보)

# YOLO 모델 학습
model = YOLO('yolov8n.pt')
results = model.train(
	data={
		'train': f'{DATASET_PATH}/images',
		'val': f'{DATASET_PATH}/images',
		'names': [
			'오실로스코프', '예초기', '전동분무기', '전동가위', '인두기',
			'용접기', '청소기', '원형톱', '그라인더', '전동드릴'
		]
	},
	epochs=50,
	imgsz=640
)

# 모델 저장
model.save(f'{MODEL_DIR}/{MODEL_NAME}')
print(f"Model saved to {MODEL_DIR}/{MODEL_NAME}")
