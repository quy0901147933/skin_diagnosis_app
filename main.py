# main.py
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import torch
from ultralytics import YOLO
from PIL import Image
import io
import json

# ==============================================================================
# PHẦN 1: ĐỊNH NGHĨA CÁC HÀM VÀ BIẾN TOÀN CỤC
# ==============================================================================

# Hàm này phải tồn tại ở đây để joblib có thể tải pipeline thành công
# vì pipeline đã được huấn luyện với hàm này.
def create_extra_features(df):
    """Tạo các đặc trưng mới từ dữ liệu đầu vào."""
    df_new = df.copy()
    # Chuyển đổi các cột sang kiểu số để tính toán
    for col in ['MucDoDau', 'ThoiGian', 'LanRong']:
        df_new[col] = pd.to_numeric(df_new[col], errors='coerce').fillna(0)
    
    df_new['Dau_x_ThoiGian'] = df_new['MucDoDau'] * df_new['ThoiGian']
    df_new['ChiSoViem'] = (df_new['MucDoDau'] > 1).astype(int) + df_new['LanRong']
    return df_new

# ==============================================================================
# PHẦN 2: KHỞI TẠO ỨNG DỤNG VÀ TẢI CÁC MÔ HÌNH
# ==============================================================================

app = FastAPI(title="Skin Diagnosis AI API")

# Sử dụng một dictionary để lưu trữ các model đã tải
models = {}

@app.on_event("startup")
def load_models():
    """Tải tất cả các mô hình và dữ liệu cần thiết khi server khởi động."""
    print("INFO:     Server startup: Loading models and data...")
    
    # Tải dữ liệu bệnh học
    try:
        with open("./diseases.json", "r", encoding="utf-8") as f:
            models["diseases_data"] = json.load(f)
        print("✅ Diseases data loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading diseases.json: {e}")
        models["diseases_data"] = {}

    # Tải Label Encoder
    try:
        models["label_encoder"] = joblib.load("label_encoder.joblib")
        print("✅ Label encoder loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading label_encoder.joblib: {e}")
        models["label_encoder"] = None

    # Tải Pipeline chẩn đoán
    try:
        models["classification_pipeline"] = joblib.load("acne_diagnosis_pipeline.joblib")
        print("✅ Classification pipeline loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading classification pipeline: {e}")
        models["classification_pipeline"] = None
        
    # Tải mô hình YOLOv8
    try:
        models["yolo_model"] = YOLO("best.pt")
        print("✅ YOLOv8 model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading YOLOv8 model: {e}")
        models["yolo_model"] = None
    
    print("INFO:     Model loading complete.")

# ==============================================================================
# PHẦN 3: ĐỊNH NGHĨA CẤU TRÚC DỮ LIỆU (PYDANTIC MODELS)
# ==============================================================================

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

# ==============================================================================
# PHẦN 4: ĐỊNH NGHĨA CÁC API ENDPOINTS
# ==============================================================================

@app.get("/")
def read_root():
    """Endpoint gốc để kiểm tra server có hoạt động không."""
    return {"message": "Welcome to the Skin Diagnosis AI API. Endpoints available: /detect and /classify"}

@app.post("/detect")
async def detect_acne(file: UploadFile = File(...)):
    """
    Nhận một file ảnh, dùng YOLOv8 để phát hiện các đối tượng và trả về tọa độ.
    """
    if not models.get("yolo_model"):
        raise HTTPException(status_code=503, detail="YOLOv8 model is not available.")

    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Chạy model YOLO
    results = models["yolo_model"](img)
    
    # Xử lý kết quả
    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            cls = box.cls[0]
            detections.append({
                "class": models["yolo_model"].names[int(cls)],
                "confidence": float(conf),
                "box": [int(x1), int(y1), int(x2), int(y2)]
            })

    return {"detections": detections}

@app.post("/classify")
async def classify_acne(data: ClinicalData):
    """
    Nhận dữ liệu lâm sàng (11 trường) và dùng mô hình Scikit-learn để phân loại.
    """
    pipeline = models.get("classification_pipeline")
    encoder = models.get("label_encoder")
    diseases = models.get("diseases_data")

    if not pipeline or not encoder:
        raise HTTPException(status_code=503, detail="Classification model is not available.")

    # Chuyển dữ liệu Pydantic thành DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    try:
        # Pipeline sẽ tự động thực hiện tất cả các bước xử lý
        prediction_encoded = pipeline.predict(input_df)
        probabilities = pipeline.predict_proba(input_df)
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during model prediction: {e}")

    # Giải mã kết quả
    predicted_class_name = encoder.inverse_transform(prediction_encoded)[0]
    confidence = np.max(probabilities)
    disease_details = diseases.get(predicted_class_name, {})

    # Gộp kết quả
    final_response = {
        "predicted_acne_type": predicted_class_name,
        "confidence": float(confidence),
        "details": disease_details
    }

    return final_response
