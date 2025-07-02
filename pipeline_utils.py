import pandas as pd

def create_extra_features(df):
    """
    Tạo các đặc trưng bổ sung từ DataFrame đầu vào.
    Hàm này được tách ra để đảm bảo tính nhất quán khi lưu và tải pipeline.
    """
    # Tạo một bản sao để tránh thay đổi DataFrame gốc
    df_new = df.copy()

    # Các cột cần được chuyển đổi sang dạng số
    numeric_cols = ['MucDoDau', 'ThoiGian', 'LanRong']

    # Chuyển đổi các cột sang kiểu số, thay thế các giá trị không hợp lệ bằng 0
    for col in numeric_cols:
        df_new[col] = pd.to_numeric(df_new[col], errors='coerce').fillna(0)

    # Tạo các đặc trưng mới
    df_new['Dau_x_ThoiGian'] = df_new['MucDoDau'] * df_new['ThoiGian']
    df_new['ChiSoViem'] = (df_new['MucDoDau'] > 1).astype(int) + df_new['LanRong']
    
    return df_new