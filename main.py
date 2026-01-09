from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# モデルの読み込み (パスの書き方を修正しました)
model = YOLO("best.pt")

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 推論 (conf=0.25 は必要に応じて調整してください)
    results = model(img, conf=0.25)

    # 判定結果を1つずつ取り出して「×」を描く
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # 座標を取得 (x1, y1, x2, y2)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # --- ここで「×」を描画 ---
            color = (0, 0, 255) # 赤色 (BGR形式)
            thickness = 5       # 線の太さ
            
            # 左上から右下への線
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            # 右上から左下への線
            cv2.line(img, (x2, y1), (x1, y2), color, thickness)
            
            # オプション：枠も描きたい場合は以下を追加
            # cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # 画像をエンコードして返す
    _, encoded_img = cv2.imencode(".png", img)
    return Response(content=encoded_img.tobytes(), media_type="image/png")

@app.get("/")
def read_root():
    return {"status": "API is running"}
