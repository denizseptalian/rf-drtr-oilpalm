import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image

# Inisialisasi client Roboflow
client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="OPbzKBpjX4FDciwm3Bk3"
)

# Streamlit UI
st.title('Deteksi Objek dengan Roboflow')
uploaded_image = st.file_uploader("Pilih gambar untuk dianalisis", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Gambar yang diupload", use_column_width=True)

    img_bytes = uploaded_image.getvalue()

    # Ganti dengan model_id kamu, biasanya seperti "workspace/project/version"
    model_id = "saraswanti/detect-count-and-visualize-3/1"

    result = client.infer(model_id, image=img_bytes)

    # Menampilkan hasil deteksi
    st.subheader("Hasil Deteksi:")
    for prediction in result["predictions"]:
        st.write(f"Label: {prediction['class']} - Confidence: {prediction['confidence']*100:.2f}% - BBox: {prediction['x']}, {prediction['y']}, {prediction['width']}, {prediction['height']}")
