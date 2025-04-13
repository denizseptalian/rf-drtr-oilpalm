import streamlit as st
import cv2
import numpy as np
from PIL import Image
from collections import Counter

# Simulasi (placeholder) hasil deteksi
def mock_draw_results(image):
    img = np.array(image.convert("RGB"))
    class_counts = Counter({"Matang": 3, "Mengkal": 1, "Mentah": 2})  # Contoh jumlah deteksi

    # Buat rectangle dummy
    h, w, _ = img.shape
    cv2.rectangle(img, (10, 10), (w//3, h//4), (0, 255, 0), 2)
    cv2.putText(img, "Matang: 0.85", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return img, class_counts

# Streamlit App UI
st.title("Deteksi dan Klasifikasi Kematangan Buah Sawit")

st.image("Buah-Kelapa-Sawit.jpg", use_column_width=True)

# Pilihan metode input
option = st.radio("Pilih metode input gambar:", ("Upload Gambar", "Gunakan Kamera"))

image = None

if option == "Upload Gambar":
    uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diunggah", use_column_width=True)

elif option == "Gunakan Kamera":
    camera_file = st.camera_input("Ambil gambar dengan kamera")
    if camera_file:
        image = Image.open(camera_file)
        st.image(image, caption="Gambar dari Kamera", use_column_width=True)

# Proses prediksi dummy
if image and st.button("Prediksi"):
    st.warning("Model YOLOv8 belum tersedia. Menampilkan hasil simulasi prediksi.")

    processed_image, class_counts = mock_draw_results(image)
    st.image(processed_image, caption="Hasil Deteksi (Simulasi)", use_column_width=True)

    st.subheader("Jumlah Objek per Kelas (Simulasi)")
    for class_name, count in class_counts.items():
        st.write(f"{class_name}: {count}")
