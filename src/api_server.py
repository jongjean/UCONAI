from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import shutil
import os

app = FastAPI()
MODEL_PATH = os.getenv('MODEL_PATH', '../models/gonggu-yolov8.pt')
model = YOLO(MODEL_PATH)

@app.post('/analyze')
async def analyze_image(file: UploadFile = File(...)):
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    results = model(temp_path)
    # 결과 예시: 바운딩 박스, 클래스, 확률 등
    output = results[0].tojson()
    os.remove(temp_path)
    return {"result": output}

@app.get('/')
def root():
    return {"message": "Gonggu AI API Server"}
