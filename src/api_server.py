

from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from ultralytics import YOLO
import shutil
import os
import json

app = FastAPI()
app.mount("/public", StaticFiles(directory="public"), name="public")
MODEL_PATH = os.getenv('MODEL_PATH', '../models/gonggu-yolov8.pt')
TOOL_INFO_PATH = os.getenv('TOOL_INFO_PATH', '../datasets/gonggu-v1/tool_info.json')
model = YOLO(MODEL_PATH)

def get_tool_info(tool_name):
    try:
        with open(TOOL_INFO_PATH, 'r', encoding='utf-8') as f:
            tool_db = json.load(f)
        for tool in tool_db:
            if tool['name'] == tool_name:
                return tool
    except Exception as e:
        return {"error": str(e)}
    return None

@app.post('/analyze')
async def analyze_image(file: UploadFile = File(...)):
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    results = model(temp_path)
    # YOLO 결과에서 가장 높은 확률의 클래스 추출
    try:
        pred = results[0].probs.top1
        class_name = results[0].names[pred]
    except Exception:
        class_name = None
    tool_info = get_tool_info(class_name) if class_name else None
    os.remove(temp_path)
    return {
        "detected_tool": class_name,
        "tool_info": tool_info
    }


# 기본 진입점에서 index.html 반환
@app.get("/")
def root():
    return FileResponse("public/index.html")
