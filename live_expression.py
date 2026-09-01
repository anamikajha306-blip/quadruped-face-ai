import cv2
import math
import joblib
import mediapipe as mp

# ==============================
# LOAD TRAINED MODEL
# ==============================

MODEL_FILE = "expression_model.joblib"

model = joblib.load(MODEL_FILE)

print("Emotion model loaded!")

# ==============================
# MEDIAPIPE SETUP
# ==============================

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/face_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
)

# ==============================
# FEATURE FUNCTIONS
# ==============================

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

        ratio(
            left_eye_top,
            left_eye_bottom,
            face_left,
            face_right
        ),

        ratio(
            right_eye_top,
            right_eye_bottom,
            face_left,
            face_right
        ),

        ratio(
            mouth_top,
            mouth_bottom,
            face_top,
            face_bottom
        ),

        ratio(
            mouth_left,
            mouth_right,
            face_left,
            face_right
        ),

        ratio(
            nose_top,
            nose_bottom,
            face_top,
            face_bottom
        ),

        ratio(
            left_brow,
            left_eye_top,
            face_top,
            face_bottom
        ),

        ratio(
            right_brow,
            right_eye_top,
            face_top,
            face_bottom
        ),

        ratio(
            mouth_top,
            nose_bottom,
            face_top,
            face_bottom
        ),

        ratio(
            face_top,
            face_bottom,
            face_left,
            face_right
        ),

        ratio(
            left_brow,
            right_brow,
            face_left,
            face_right
        ),
    ]

    return features


# ==============================
# CAMERA
# ==============================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("Camera started!")
print("Press Q to quit.")

# MediaPipe video timestamps must increase
timestamp_ms = 0

with FaceLandmarker.create_from_options(options) as landmarker:

    while True:

        success, frame = cap.read()

        if not success:
            print("Could not read camera frame.")
            break

        # Mirror camera
        frame = cv2.flip(frame, 1)

        # Convert BGR → RGB
        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp_ms += 33

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        if result.face_landmarks:

            landmarks = result.face_landmarks[0]

            features = extract_features(
                landmarks
            )

            # Predict expression
            prediction = model.predict(
                [features]
            )[0]

            # Prediction probabilities
            probabilities = model.predict_proba(
                [features]
            )[0]

            classes = model.classes_

            confidence = max(probabilities) * 100

            # Display prediction
            text = (
                f"{prediction.upper()} "
                f"{confidence:.1f}%"
            )

            cv2.putText(
                frame,
                text,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )

            # Draw landmarks
            for landmark in landmarks:

                x = int(
                    landmark.x * frame.shape[1]
                )

                y = int(
                    landmark.y * frame.shape[0]
                )

                cv2.circle(
                    frame,
                    (x, y),
                    1,
                    (0, 255, 0),
                    -1
                )

        else:

            cv2.putText(
                frame,
                "NO FACE DETECTED",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2
            )

        cv2.imshow(
            "Quadruped Emotion AI",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break


cap.release()
cv2.destroyAllWindows()

print("Camera stopped.")