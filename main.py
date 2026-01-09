from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()

# CORS設定（通信許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# モデルの読み込み
model = YOLO("best.pt")

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 推論 (confの値で感度を調整)
    results = model(img, conf=0.25)

    # 検出された場所に「×」を描く
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = (0, 0, 255) # 赤色
            thickness = 5
            # ×印を描画
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            cv2.line(img, (x2, y1), (x1, y2), color, thickness)

    # JPEG形式で圧縮して返す（通信を速くするため）
    _, encoded_img = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")

@app.get("/")
def read_root():
    return {"status": "API is running"}
