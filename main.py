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
import sys
import os

# THAY ĐỔI QUAN TRỌNG:
# Vì pipeline_utils.py nằm cùng thư mục với main.py,
# chúng ta sẽ nhập trực tiếp như thế này.
from pipeline_utils import create_extra_features

# ==============================================================================
# PHẦN 1: KHỞI TẠO ỨNG DỤNG VÀ BIẾN TOÀN CỤC
# ==============================================================================

app = FastAPI(
    title="Skin Diagnosis AI API",
    description="API cho phép chẩn đoán bệnh về da dựa trên hình ảnh và dữ liệu lâm sàng.",
    version="1.1.0"
)

# Sử dụng một dictionary để lưu trữ các model đã tải để quản lý dễ dàng
models = {}

# Xác định đường dẫn gốc của dự án để tải file một cách đáng tin cậy
# Điều này rất hữu ích khi chạy trong các môi trường khác nhau như Docker hoặc Render
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================
# PHẦN 2: TẢI MÔ HÌNH KHI KHỞI ĐỘNG
# ==============================================================================

@app.on_event("startup")
def load_models():
    """Tải tất cả các mô hình và dữ liệu cần thiết khi server khởi động."""
    print("INFO:     Server startup: Bắt đầu quá trình tải mô hình...")

    # Thêm hàm vào module __main__ một cách tường minh để joblib có thể tìm thấy
    sys.modules['__main__'].create_extra_features = create_extra_features

    # Hàm trợ giúp để tải file một cách an toàn
    def load_file(file_name, loader_func, *args, **kwargs):
        path = os.path.join(BASE_DIR, file_name)
        try:
            model = loader_func(path, *args, **kwargs)
            print(f"✅ Tải {file_name} thành công.")
            return model
        except Exception as e:
            print(f"❌ Lỗi khi tải {file_name}: {e}")
            return None

    # Tải dữ liệu bệnh học
    def json_loader(path, encoding="utf-8"):
        with open(path, "r", encoding=encoding) as f:
            return json.load(f)
    models["diseases_data"] = load_file("diseases.json", json_loader) or {}

    # Tải Label Encoder
    models["label_encoder"] = load_file("label_encoder.joblib", joblib.load)

    # Tải Pipeline chẩn đoán
    models["classification_pipeline"] = load_file("acne_diagnosis_pipeline.joblib", joblib.load)
        
    # Tải mô hình YOLOv8
    models["yolo_model"] = load_file("best.pt", YOLO)
    
    print("INFO:     Hoàn tất quá trình tải mô hình.")

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

@app.get("/", summary="Kiểm tra trạng thái API", tags=["General"])
def read_root():
    """Endpoint gốc để kiểm tra server có hoạt động không."""
    return {"message": "Chào mừng đến với API Chẩn Đoán Da liễu. Các endpoints có sẵn: /detect và /classify"}

@app.post("/detect", summary="Phát hiện mụn từ hình ảnh", tags=["YOLOv8"])
async def detect_acne(file: UploadFile = File(..., description="File ảnh cần phân tích.")):
    """
    Nhận một file ảnh, dùng YOLOv8 để phát hiện các loại mụn và trả về tọa độ,
    tên lớp và độ tin cậy cho mỗi phát hiện.
    """
    if not models.get("yolo_model"):
        raise HTTPException(status_code=503, detail="Mô hình YOLOv8 chưa sẵn sàng. Vui lòng thử lại sau.")

    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="File ảnh không hợp lệ.")

    # Chạy model YOLO
    results = models["yolo_model"](img)
    
    # Xử lý kết quả
    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            cls_id = int(box.cls[0])
            class_name = models["yolo_model"].names[cls_id]
            
            detections.append({
                "class": class_name,
                "confidence": round(float(conf), 4),
                "box": [int(x1), int(y1), int(x2), int(y2)]
            })

    if not detections:
        return {"message": "Không phát hiện được đối tượng nào trong ảnh."}

    return {"detections": detections}

@app.post("/classify", summary="Chẩn đoán bệnh từ dữ liệu lâm sàng", tags=["Scikit-learn"])
async def classify_acne(data: ClinicalData):
    """
    Nhận 11 trường dữ liệu lâm sàng và dùng mô hình Scikit-learn để phân loại bệnh,
    trả về loại bệnh, độ tin cậy và thông tin chi tiết.
    """
    pipeline = models.get("classification_pipeline")
    encoder = models.get("label_encoder")
    diseases = models.get("diseases_data")

    if not pipeline or not encoder:
        raise HTTPException(status_code=503, detail="Mô hình phân loại chưa sẵn sàng. Vui lòng thử lại sau.")

    # Chuyển dữ liệu Pydantic thành DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    try:
        # Pipeline sẽ tự động thực hiện tất cả các bước xử lý và dự đoán
        prediction_encoded = pipeline.predict(input_df)
        probabilities = pipeline.predict_proba(input_df)
    except Exception as e:
        print(f"Lỗi trong quá trình dự đoán: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi xảy ra trong quá trình dự đoán của mô hình: {e}")

    # Giải mã kết quả
    predicted_class_name = encoder.inverse_transform(prediction_encoded)[0]
    confidence = np.max(probabilities)
    disease_details = diseases.get(predicted_class_name, {
        "description": "Không có thông tin chi tiết cho loại bệnh này.",
        "symptoms": [],
        "treatment": []
    })

    # Gộp kết quả
    final_response = {
        "predicted_acne_type": predicted_class_name,
        "confidence": round(float(confidence), 4),
        "details": disease_details
    }

    return final_response
