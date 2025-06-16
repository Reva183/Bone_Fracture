import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np
import tensorflow as tf
import cv2  # For image resizing and conversion

# Load your model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("best_model (1).h5") #model (1).h5
    return model

model = load_model()

st.title("🦴 Bone Fracture Detector")

# File uploader
uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "jpeg", "png"])

# Function to preprocess image
def preprocess_image(image, target_size=(256, 256)):
    img = image.convert("RGB")  # Ensure 3 channels
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# --- If file is uploaded ---
if uploaded_file is not None:
    # Load original image
    original_image = Image.open(uploaded_file)

    # Preprocess image
    processed_image = preprocess_image(original_image)

    # Predict using model
    prediction = model.predict(processed_image)[0][0]  # Assuming binary classification

    # Interpret prediction
    if prediction > 0.5:
        result_text = f"Fracture Detected ✅ (Confidence: {prediction * 100:.2f}%)"
    else:
        result_text = f"No Fracture Detected 🟩 (Confidence: {(1 - prediction) * 100:.2f}%)"

    # Enhance image (simulated model visualization)
    modal_image = ImageEnhance.Contrast(original_image).enhance(2.0)

    # --- Show Model Output ---
    st.subheader("🧠 Model Output")
    st.success(result_text)

    # --- Side-by-side images ---
    col1, col2 = st.columns(2)

    with col1:
        st.image(original_image, caption="Original X-ray", use_container_width=True)

    with col2:
        st.image(modal_image, caption="Processed Image", use_container_width=True)

else:
    st.info("Please upload an image to begin fracture detection.")