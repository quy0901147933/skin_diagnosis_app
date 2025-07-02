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
import json
# (MỚI) Import thêm FunctionTransformer và Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# (MỚI) Import các lớp mô hình để xây dựng lại pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier


# === KHỞI TẠO ỨNG DỤNG ===
app = FastAPI(title="Skin Diagnosis AI API")

# === ĐỊNH NGHĨA CÁC HÀM VÀ BIẾN TOÀN CỤC ===

# Hàm tạo Feature mới (giữ nguyên)
def create_extra_features(df):
    df_new = df.copy()
    # Chuyển đổi các cột sang kiểu số để tính toán, phòng trường hợp dữ liệu đầu vào là chuỗi
    for col in ['MucDoDau', 'ThoiGian', 'LanRong']:
        df_new[col] = pd.to_numeric(df_new[col], errors='coerce').fillna(0)
    
    df_new['Dau_x_ThoiGian'] = df_new['MucDoDau'] * df_new['ThoiGian']
    df_new['ChiSoViem'] = (df_new['MucDoDau'] > 1).astype(int) + df_new['LanRong']
    return df_new

# Tải dữ liệu bệnh học
try:
    with open("./diseases.json", "r", encoding="utf-8") as f:
        diseases_data = json.load(f)
    print("✅ Diseases data loaded successfully.")
except Exception as e:
    print(f"❌ Error loading diseases.json: {e}")
    diseases_data = {}

# Tải Label Encoder
try:
    label_encoder = joblib.load("label_encoder.joblib")
    print("✅ Label encoder loaded successfully.")
except Exception as e:
    print(f"❌ Error loading label_encoder.joblib: {e}")
    label_encoder = None

# Tải Pipeline chẩn đoán
try:
    classification_pipeline = joblib.load("acne_diagnosis_pipeline.joblib")
    print("✅ Classification pipeline loaded successfully.")
except Exception as e:
    print(f"❌ Error loading classification pipeline: {e}")
    classification_pipeline = None
    
# Tải mô hình YOLOv8
try:
    yolo_model = YOLO("best.pt")
    print("✅ YOLOv8 model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading YOLOv8 model: {e}")
    yolo_model = None


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
    return {"message": "Welcome to the Skin Diagnosis AI API v2."}

# Endpoint cho YOLOv8 (giữ nguyên)
@app.post("/detect")
async def detect_acne(file: UploadFile = File(...)):
    # ... (giữ nguyên code của bạn cho phần này)
    pass 

# Endpoint cho phân loại (Sửa đổi logic gọi pipeline)
@app.post("/classify")
async def classify_acne(data: ClinicalData):
    if not classification_pipeline or not label_encoder:
        return {"error": "Classification model is not available."}

    # Chuyển dữ liệu Pydantic thành DataFrame để mô hình có thể đọc
    input_df = pd.DataFrame([data.dict()])
    
    # (SỬA ĐỔI QUAN TRỌNG)
    # Bây giờ chúng ta chỉ cần truyền DataFrame thô vào pipeline.
    # Pipeline sẽ tự động thực hiện TẤT CẢ các bước, bao gồm cả create_extra_features
    try:
        prediction_encoded = classification_pipeline.predict(input_df)
        probabilities = classification_pipeline.predict_proba(input_df)
    except Exception as e:
        # Nếu có lỗi trong quá trình predict, trả về lỗi 500
        print(f"Error during prediction: {e}")
        return {"error": f"An error occurred during model prediction: {e}"}

    # Giải mã kết quả
    predicted_class_name = label_encoder.inverse_transform(prediction_encoded)[0]
    confidence = np.max(probabilities)
    disease_details = diseases_data.get(predicted_class_name, {})

    # Gộp kết quả
    final_response = {
        "predicted_acne_type": predicted_class_name,
        "confidence": float(confidence),
        "details": disease_details
    }

    return final_response
