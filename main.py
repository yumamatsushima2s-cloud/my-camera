from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from ultralytics import YOLO
import cv2
import numpy as np
import io

app = FastAPI()

# モデルの読み込み (best.pt または best_v2.pt)
model = YOLO("best.pt")

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    # 1. 送られてきた画像データを読み込む
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 2. YOLOで推論（判定）
    results = model(img)

    # 3. 判定結果を画像に描き込む（枠やラベル）
    res_plotted = results[0].plot()

    # 4. 画像をバイトデータに変換してレスポンスとして返す
    _, encoded_img = cv2.imencode(".png", res_plotted)
    return Response(content=encoded_img.tobytes(), media_type="image/png")

@app.get("/")
def read_root():
    return {"status": "API is running"}
