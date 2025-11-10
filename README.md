# 🧠 UCONAI — 공구 형상 인식 AI 엔진

> **UCONAI**는 공구(도구) 이미지를 AI가 자동 분석하여  
> **종류, 제조사, 모델명을 인식하고 DB화하는 시스템**입니다.  
> YOLO 기반의 딥러닝 모델과 FastAPI 기반 추론 서버로 구성되며,  
> `공구반장(Gonggu Manager)` 웹앱과 직접 연동됩니다.  

---

## 🚀 프로젝트 개요

| 항목 | 내용 |
|------|------|
| **프로젝트명** | UCONAI |
| **목표** | 공구 형상 인식·분석 AI + REST API 서버 |
| **AI 모델** | YOLOv8 (Ultralytics) |
| **API 프레임워크** | FastAPI + Uvicorn |
| **개발 환경** | Ubuntu 22.04 + Docker + GPU (CUDA 12.x) |
| **프런트 연동** | `/gonggu/api/analyze` (Caddy HTTPS 라우팅) |
| **데이터베이스(확장)** | MongoDB / PostgreSQL / MinIO |

---

## 🧩 프로젝트 구조

/home/ucon/gonggu-ai/

├── src/ 

│   ├── train_yolo.py        # YOLO 학습 스크립트 

│   ├── detect_yolo.py       # 추론 테스트 

│   ├── api_server.py        # FastAPI 추론 서버

│   ├── data_prep.py         # 데이터셋 전처리/DB 등록

│   └── utils.py             # 공통 유틸 함수

├── datasets/                # 학습용 이미지 및 라벨

├── models/                  # 학습된 모델 가중치(.pt / .onnx)

├── notebooks/               # 분석/시각화용 Jupyter 노트북

├── docker-compose.yml       # GPU 컨테이너 실행 설정

├── requirements.txt         # Python 의존성 목록

└── README.md

---

## ⚙️ 설치 및 실행 가이드

### 1️⃣ 환경 준비

Ubuntu에서 Docker 및 NVIDIA GPU 지원 설치:

```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose
nvidia-smi

2️⃣ 프로젝트 클론

git clone https://github.com/jongjean/UCONAI.git
cd UCONAI

3️⃣ 컨테이너 실행
sudo docker compose up -d
sudo docker exec -it gonggu-ai bash

4️⃣ Python 환경 준비

컨테이너 내부에서:

pip install -r requirements.txt

🧠 YOLO 모델 학습

src/train_yolo.py 예시:

from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="/datasets/gonggu-v1/data.yaml",
    epochs=50,
    imgsz=640,
    project="/models",
    name="gonggu_v1"
)


실행:

python src/train_yolo.py


모델 결과:

/models/gonggu_v1/weights/best.pt

🌐 FastAPI 추론 서버 실행

src/api_server.py:

from fastapi import FastAPI, UploadFile
from ultralytics import YOLO
import cv2, numpy as np

app = FastAPI()
model = YOLO("/models/gonggu_v1/weights/best.pt")

@app.post("/analyze")
async def analyze(file: UploadFile):
    image = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)
    results = model(img)
    detections = []
    for box, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):
        detections.append({
            "bbox": box.tolist(),
            "class": model.names[int(cls)]
        })
    return {"detections": detections}


실행:

uvicorn src.api_server:app --host 0.0.0.0 --port 4000

📡 API 테스트
curl -X POST -F "file=@sample_tool.jpg" http://127.0.0.1:4000/analyze


응답 예시:

{
  "detections": [
    {
      "bbox": [52.1, 47.5, 210.3, 198.2],
      "class": "spanner"
    }
  ]
}

🗄️ 데이터셋 구조 (예시)
datasets/gonggu-v1/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml


data.yaml:

train: /datasets/gonggu-v1/images/train
val: /datasets/gonggu-v1/images/val
nc: 5
names: ['drill', 'spanner', 'hammer', 'screwdriver', 'pliers']

🚀 향후 확장 계획
단계	목표	세부 내용
🔹 1단계	YOLO 학습 자동화	공구 이미지 수집·전처리 파이프라인 구축
🔹 2단계	OCR 결합	제조사/모델명 자동 인식 (PaddleOCR)
🔹 3단계	FastAPI 확장	/analyze, /feedback, /train API 추가
🔹 4단계	DB 연동	MongoDB + MinIO로 학습 데이터 버전 관리
🔹 5단계	프런트 연동	/gonggu/api/analyze → React 웹앱과 통신
🔹 6단계	MLOps 구축	DVC + GitHub Actions + Caddy 자동배포
👥 프로젝트 정보
항목	내용
저장소	https://github.com/jongjean/UCONAI

작성자	Dr. Kang Jong-Jean (UCONCREATIVE)
이메일	uconcreative@gmail.com

프로젝트명	UCONAI – Tool Recognition & Management AI
📜 라이선스

MIT License © 2025 UCONCREATIVE

✅ 한 줄 요약

“UCONAI는 공구 사진 한 장으로 종류·제조사·모델명을 자동 인식하고 DB화하는
산업용 AI 엔진입니다. YOLO + FastAPI 기반으로 개발되어,
‘공구반장’ 웹 플랫폼과 실시간 연동됩니다.”








