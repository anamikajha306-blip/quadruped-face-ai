import streamlit as st
import math
import joblib
import mediapipe as mp
import numpy as np
from PIL import Image

# -----------------------------
# PAGE SETTINGS
# -----------------------------

st.set_page_config(
    page_title="Quadruped Emotion AI",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Quadruped Emotion AI")
st.write("Camera-based facial expression recognition")

# -----------------------------
# LOAD MODEL
# -----------------------------

MODEL_FILE = "expression_model.joblib"

model = joblib.load(MODEL_FILE)

# -----------------------------
# MEDIAPIPE SETUP
# -----------------------------

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/face_landmarker.task"
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1,
)

# -----------------------------
# FEATURE FUNCTIONS
# -----------------------------

def distance(a, b):
    return math.sqrt(
        (a.x - b.x) ** 2 +
        (a.y - b.y) ** 2 +
        (a.z - b.z) ** 2
    )


def ratio(a, b, c, d):
    numerator = distance(a, b)
    denominator = distance(c, d)

    if denominator == 0:
        return 0

    return numerator / denominator


def extract_features(lm):

    left_eye_top = lm[159]
    left_eye_bottom = lm[145]

    right_eye_top = lm[386]
    right_eye_bottom = lm[374]

    left_brow = lm[105]
    right_brow = lm[334]

    mouth_top = lm[13]
    mouth_bottom = lm[14]

    mouth_left = lm[61]
    mouth_right = lm[291]

    nose_top = lm[6]
    nose_bottom = lm[2]

    face_left = lm[234]
    face_right = lm[454]

    face_top = lm[10]
    face_bottom = lm[152]

    features = [
        ratio(left_eye_top, left_eye_bottom, face_left, face_right),
        ratio(right_eye_top, right_eye_bottom, face_left, face_right),
        ratio(mouth_top, mouth_bottom, face_top, face_bottom),
        ratio(mouth_left, mouth_right, face_left, face_right),
        ratio(nose_top, nose_bottom, face_top, face_bottom),
        ratio(left_brow, left_eye_top, face_top, face_bottom),
        ratio(right_brow, right_eye_top, face_top, face_bottom),
        ratio(mouth_top, nose_bottom, face_top, face_bottom),
        ratio(face_top, face_bottom, face_left, face_right),
        ratio(left_brow, right_brow, face_left, face_right),
    ]

    return features


# -----------------------------
# CAMERA
# -----------------------------

camera_image = st.camera_input("Take a picture")

# -----------------------------
# PROCESS IMAGE
# -----------------------------

if camera_image is not None:

    image = Image.open(camera_image).convert("RGB")
    image_rgb = np.array(image) 

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_rgb
    )

    with FaceLandmarker.create_from_options(options) as landmarker:

        result = landmarker.detect(mp_image)

    if result.face_landmarks:

        landmarks = result.face_landmarks[0]

        features = extract_features(landmarks)

        prediction = model.predict([features])[0]

        probabilities = model.predict_proba([features])[0]

        confidence = max(probabilities) * 100

        st.success(
            f"Expression: {prediction.upper()}"
        )

        st.metric(
            "Confidence",
            f"{confidence:.1f}%"
        )

        # Show all probabilities

        st.subheader("Prediction probabilities")

        classes = model.classes_

        for cls, prob in zip(classes, probabilities):

            st.write(
                f"**{cls.upper()}** — {prob * 100:.1f}%"
            )

    else:

        st.error(
            "No face detected. Please try another picture."
        )
