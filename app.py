import streamlit as st
import pandas as pd
import joblib
import os

# =========================
# CẤU HÌNH ĐƯỜNG DẪN
# =========================
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "XGBoost_best.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "XGBoost_scaler.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "XGBoost_label_encoder.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "models", "XGBoost_important_features.pkl")

# =========================
# LOAD MODEL & ARTIFACTS
# =========================
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoders = joblib.load(ENCODER_PATH)
    important_features = joblib.load(FEATURE_PATH)
    return model, scaler, label_encoders, important_features

model, scaler, label_encoders, important_features = load_artifacts()

# =========================
# TÊN THUỘC TÍNH TIẾNG VIỆT
# =========================
feature_name_vi = {
    'Administrative': 'Số trang quản trị',
    'Administrative_Duration': 'Thời gian trang quản trị',
    'Informational': 'Số trang thông tin',
    'Informational_Duration': 'Thời gian trang thông tin',
    'ProductRelated': 'Số trang sản phẩm',
    'ProductRelated_Duration': 'Thời gian trang sản phẩm',
    'BounceRates': 'Tỷ lệ thoát',
    'ExitRates': 'Tỷ lệ rời trang',
    'PageValues': 'Giá trị trang'
}

# =========================
# GIAO DIỆN
# =========================
st.set_page_config(page_title="Dự đoán ý định mua hàng", layout="wide")

st.markdown("""
<style>
.header {
    background-color: #2563eb;
    padding: 25px;
    border-radius: 12px;
    color: white;
    text-align: center;
    margin-bottom: 25px;
}
.section {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header">
    <h2>Hệ thống dự đoán ý định mua hàng</h2>
    <p>Ứng dụng mô hình XGBoost nhằm dự đoán khả năng
        khách truy cập website thương mại điện tử thực hiện mua hàng</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section">', unsafe_allow_html=True)
st.subheader("Thông tin khách truy cập website")

input_data = {}
num_cols = 4

for i in range(0, len(important_features), num_cols):
    cols = st.columns(num_cols)
    for col, feature in zip(cols, important_features[i:i + num_cols]):
        with col:
            label_vi = feature_name_vi.get(feature, feature)
            input_data[feature] = st.number_input(
                label=label_vi,
                min_value=0.0,
                value=0.0
            )

st.markdown('</div>', unsafe_allow_html=True)

# =========================
# DỰ ĐOÁN
# =========================
if st.button("🔮 Dự đoán"):
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    st.subheader("📊 Kết quả dự đoán")

    if prediction == 1:
        st.success("✅ Khách hàng CÓ khả năng mua hàng")
    else:
        st.warning("❌ Khách hàng KHÔNG có khả năng mua hàng")

    st.write("Xác suất dự đoán:")
    st.dataframe(pd.DataFrame({
        "Lớp": model.classes_,
        "Xác suất": probability
    }))

