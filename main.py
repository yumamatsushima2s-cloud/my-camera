from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware  # 追加
from ultralytics import YOLO
import cv2
import numpy as np
import io

app = FastAPI()

# ★★★ セキュリティ制限（CORS）を解除する設定を追加 ★★★
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # すべてのサイトからのアクセスを許可
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

    # 推論
    results = model(img)

    # 判定結果を描画
    res_plotted = results[0].plot()

    _, encoded_img = cv2.imencode(".png", res_plotted)
    return Response(content=encoded_img.tobytes(), media_type="image/png")

@app.get("/")
def read_root():
    return {"status": "API is running"}
