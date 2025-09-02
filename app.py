import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# تحميل الموديل
model = load_model("mask_detector_model.h5", compile=False)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

st.title("😷 Face Mask Detection (Image Upload)")
st.write("ارفع صورة وتأكد إذا الشخص مرتدي الكمامة أم لا")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    result_text = "⚠️ No face detected"
    result_color = "orange"

    for (x, y, w, h) in faces:
        face = img_bgr[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (224, 224))
        face_normalized = face_resized / 255.0
        face_input = np.expand_dims(face_normalized, axis=0)

        result = model.predict(face_input, verbose=0)[0][0]

        if result > 0.5:
            label = "❌ No Mask"
            color = (0, 0, 255)
            result_text = "❌ الشخص غير مرتدي كمامة"
            result_color = "red"
        else:
            label = "✅ Mask On"
            color = (0, 255, 0)
            result_text = "✅ الشخص مرتدي الكمامة"
            result_color = "green"

        cv2.rectangle(img_bgr, (x, y), (x+w, y+h), color, 3)
        cv2.putText(img_bgr, label, (x, y-15), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)

    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Processed Image", use_container_width=True)
    st.markdown(f"<h3 style='color:{result_color};'>{result_text}</h3>", unsafe_allow_html=True)
