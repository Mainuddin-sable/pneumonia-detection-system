
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import json

from PIL import Image
from pathlib import Path

# --------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🫁",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "pneumonia_model.keras"
CONFIG_PATH = BASE_DIR / "models" / "config.json"


# --------------------------------
# 2. LOAD MODEL AND CONFIGURATION
# --------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


with open(CONFIG_PATH, "r") as file:
    config = json.load(file)

IMAGE_SIZE = config["image_size"]
THRESHOLD = config["threshold"]

st.title("🫁 Chest X-ray Pneumonia Detection")
st.write("Deep learning classification using MobileNetV2")

st.warning(
    "Educational project only. "
    "Not intended for medical diagnosis."
)

try:
    model = load_model()
    st.success("AI model loaded successfully!")

except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.stop()


# --------------------------------
# 3. PREPARE GRAD-CAM
# --------------------------------

base_model = model.get_layer("mobilenetv2_1.00_224")

last_conv_layer = base_model.get_layer("Conv_1")

feature_extractor = tf.keras.Model(
    inputs=base_model.input,
    outputs=[
        last_conv_layer.output,
        base_model.output
    ]
)


# --------------------------------
# 4. GRAD-CAM FUNCTION
# --------------------------------

def generate_gradcam(image_batch, predicted_class):

    with tf.GradientTape() as tape:

        # Apply model preprocessing
        x = model.layers[0](image_batch, training=False)

        # Extract convolutional features
        conv_outputs, features = feature_extractor(
            x, training=False
        )

        # Apply classification layers
        for layer in model.layers[2:]:
            features = layer(features, training=False)

        predictions = features

        # Select the predicted class score
        if predicted_class == 1:
            target_score = predictions[:, 0]
        else:
            target_score = 1.0 - predictions[:, 0]

    # Calculate gradients
    gradients = tape.gradient(target_score, conv_outputs)

    # Calculate channel importance
    weights = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    # Generate heatmap
    heatmap = tf.reduce_sum(
        conv_outputs[0] * weights,
        axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)

    heatmap = heatmap / (
        tf.reduce_max(heatmap) + 1e-8
    )

    # Resize heatmap
    heatmap = tf.image.resize(
        heatmap[..., tf.newaxis],
        (IMAGE_SIZE, IMAGE_SIZE)
    ).numpy().squeeze()

    # Convert heatmap into color
    heatmap_uint8 = np.uint8(255 * heatmap)

    colored_heatmap = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    colored_heatmap = cv2.cvtColor(
        colored_heatmap,
        cv2.COLOR_BGR2RGB
    )

    # Create original image
    original = np.uint8(
        image_batch[0] * 255
    )

    # Overlay heatmap
    overlay = cv2.addWeighted(
        original,
        0.6,
        colored_heatmap,
        0.4,
        0
    )

    return overlay


# --------------------------------
# 5. UPLOAD X-RAY IMAGE
# --------------------------------

uploaded_file = st.file_uploader(
    "Upload a chest X-ray image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------
# 6. IMAGE PROCESSING
# --------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    resized_image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    image_array = np.array(
        resized_image,
        dtype=np.float32
    ) / 255.0

    image_batch = np.expand_dims(
        image_array,
        axis=0
    )

    st.subheader("Uploaded Chest X-ray")

    st.image(
        image,
        width=400
    )


    # --------------------------------
    # 7. MODEL PREDICTION
    # --------------------------------

    if st.button("Analyze X-ray"):

        with st.spinner("Analyzing X-ray..."):

            prediction = model.predict(
                image_batch,
                verbose=0
            )

            pneumonia_probability = float(
                prediction[0][0]
            )

            if pneumonia_probability >= THRESHOLD:
                predicted_class = 1
                result = "PNEUMONIA"
            else:
                predicted_class = 0
                result = "NORMAL"

            class_probability = (
                pneumonia_probability
                if predicted_class == 1
                else 1 - pneumonia_probability
            )

            # Generate Grad-CAM
            gradcam_image = generate_gradcam(
                image_batch,
                predicted_class
            )

        # --------------------------------
        # 8. DISPLAY RESULTS
        # --------------------------------

        st.subheader("Prediction Result")

        if predicted_class == 1:
            st.error(f"Prediction: {result}")
        else:
            st.success(f"Prediction: {result}")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Pneumonia probability",
                f"{pneumonia_probability * 100:.2f}%"
            )

        with col2:
            st.metric(
                "Predicted class score",
                f"{class_probability * 100:.2f}%"
            )

        st.caption(
            f"Decision threshold: {THRESHOLD}"
        )

        # --------------------------------
        # 9. DISPLAY GRAD-CAM
        # --------------------------------

        st.subheader("Grad-CAM Explainability")

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                resized_image,
                caption="Original X-ray",
                use_container_width=True
            )

        with col2:
            st.image(
                gradcam_image,
                caption="Grad-CAM Heatmap",
                use_container_width=True
            )

        st.info(
            "The heatmap highlights image regions that "
            "influenced the model's prediction. It does "
            "not establish a medical diagnosis."
        )
