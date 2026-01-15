import streamlit as st
import pandas as pd
import joblib
import os


BASE_PATH = r"C:\Users\pc\Desktop\Project_Random_Forest_2\models"

MODEL_PATH = os.path.join(BASE_PATH, "RandomForest_best.pkl")
SCALER_PATH = os.path.join(BASE_PATH, "RandomForest_scaler.pkl")
ENCODER_PATH = os.path.join(BASE_PATH, "RandomForest_label_encoders.pkl")
FEATURE_PATH = os.path.join(BASE_PATH, "RandomForest_important_features.pkl")

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoders = joblib.load(ENCODER_PATH)
    important_features = joblib.load(FEATURE_PATH)
    return model, scaler, label_encoders, important_features

model, scaler, label_encoders, important_features = load_artifacts()

# =========================
# DỊCH TÊN THUỘC TÍNH SANG TIẾNG VIỆT
# =========================
feature_name_vi = {
    'Administrative': 'Số trang quản trị',
    'Administrative_Duration': 'Thời gian trên trang quản trị',
    'Informational': 'Số trang thông tin',
    'Informational_Duration': 'Thời gian trên trang thông tin',
    'ProductRelated': 'Số trang liên quan đến sản phẩm',
    'ProductRelated_Duration': 'Thời gian trên trang sản phẩm',
    'BounceRates': 'Tỷ lệ thoát ngay',
    'ExitRates': 'Tỷ lệ rời trang',
    'PageValues': 'Giá trị trang'
}

# Danh sách feature tiếng Việt để hiển thị
important_features_vi = [
    feature_name_vi[f] if f in feature_name_vi else f
    for f in important_features
]


st.set_page_config(
    page_title="Dự đoán ý định mua hàng",
    layout="wide"
)

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
.result {
    background-color: #ecfdf5;
    padding: 20px;
    border-radius: 12px;
    border-left: 6px solid #10b981;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Header
# =========================
st.markdown("""
<div class="header">
    <h2>Hệ thống dự đoán ý định mua hàng</h2>
    <p>
        Ứng dụng mô hình Random Forest nhằm dự đoán khả năng
        khách truy cập website thương mại điện tử thực hiện mua hàng
    </p>
</div>
""", unsafe_allow_html=True)


st.markdown('<div class="section">', unsafe_allow_html=True)
st.subheader("Thông tin khách truy cập website")

input_data = {}

num_cols = 4
features_numeric = important_features
# features_numeric = [f for f in important_features if f not in label_encoders]

for i in range(0, len(features_numeric), num_cols):
    cols = st.columns(num_cols)
    for col, feature in zip(cols, features_numeric[i:i + num_cols]):
        with col:
            # Lấy tên tiếng Việt để hiển thị
            label_vi = feature_name_vi.get(feature, feature)

            input_data[feature] = st.number_input(
                label=label_vi,   # dùng tiếng Việt thay vì tiếng Anh
                min_value=0.0,
                value=0.0
            )


st.markdown("</div>", unsafe_allow_html=True)





# =====================
# Dự đoán
# =====================
if st.button("🔮 Dự đoán"):
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    st.subheader("📊 Kết quả dự đoán")

    if prediction == 1:
        st.success("Khách hàng **CÓ khả năng mua hàng**")
    else:
        st.warning("Khách hàng **KHÔNG có khả năng mua hàng**")

    st.write("Xác suất dự đoán:")
    st.dataframe(
        pd.DataFrame({
            "Lớp": model.classes_,
            "Xác suất": probability
        })
    )
