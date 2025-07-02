# main.py
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import torch
from ultralytics import YOLO
from PIL import Image
import io
import json # <-- Thêm thư viện json

# === KHỞI TẠO ỨNG DỤNG VÀ TẢI MÔ HÌNH ===
app = FastAPI(title="Skin Diagnosis AI API")

# 1. Tải mô hình YOLOv8
try:
    yolo_model = YOLO("./models/best.pt")
    print("✅ YOLOv8 model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading YOLOv8 model: {e}")
    yolo_model = None

# 2. Tải mô hình Phân loại (Pipeline) và Label Encoder
try:
    classification_pipeline = joblib.load("./models/acne_diagnosis_pipeline.joblib")
    label_encoder = joblib.load("./models/label_encoder.joblib")
    print("✅ Classification pipeline and encoder loaded successfully.")
except Exception as e:
    print(f"❌ Error loading classification model: {e}")
    classification_pipeline = None
    label_encoder = None

# 3. (MỚI) Tải dữ liệu bệnh học từ file JSON
# Sử dụng encoding="utf-8" để đọc tiếng Việt
try:
    with open("./diseases.json", "r", encoding="utf-8") as f:
        diseases_data = json.load(f)
    print("✅ Diseases data loaded successfully.")
except Exception as e:
    print(f"❌ Error loading diseases.json: {e}")
    diseases_data = {}

# === ĐỊNH NGHĨA CẤU TRÚC DỮ LIỆU ĐẦU VÀO ===
class ClinicalData(BaseModel):
    Tuoi: int
    GioiTinh: int
    ViTri: int
    DacDiem: int
    MucDoDau: int
    ThoiGian: int
    LanRong: int
    LoaiDa: int
    NanMun: int
    ChamSocDa: int
    LoaiNhanMun: int

# === ĐỊNH NGHĨA CÁC ENDPOINTS ===

@app.get("/")
def read_root():
    return {"message": "Welcome to the Skin Diagnosis AI API with YOLOv8 and Scikit-learn models."}

@app.post("/detect")
async def detect_acne(file: UploadFile = File(...)):
    if not yolo_model:
        return {"error": "YOLOv8 model is not available."}
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes))
    results = yolo_model(img)
    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            cls = box.cls[0]
            detections.append({
                "class": yolo_model.names[int(cls)],
                "confidence": float(conf),
                "box": [int(x1), int(y1), int(x2), int(y2)]
            })
    return {"detections": detections}

# --- ENDPOINT 2: PHÂN LOẠI MỤN (CẬP NHẬT) ---
@app.post("/classify")
async def classify_acne(data: ClinicalData):
    if not classification_pipeline or not label_encoder:
        return {"error": "Classification model is not available."}

    input_df = pd.DataFrame([data.dict()])
    prediction_encoded = classification_pipeline.predict(input_df)
    probabilities = classification_pipeline.predict_proba(input_df)
    predicted_class_name = label_encoder.inverse_transform(prediction_encoded)[0]
    confidence = np.max(probabilities)

    # (MỚI) Lấy thông tin chi tiết về bệnh học từ dữ liệu đã tải
    # Dùng .get() để tránh lỗi nếu không tìm thấy loại mụn
    disease_details = diseases_data.get(predicted_class_name, {})

    # (MỚI) Gộp kết quả dự đoán với thông tin chi tiết
    final_response = {
        "predicted_acne_type": predicted_class_name,
        "confidence": float(confidence),
        "details": disease_details
    }

    return final_response