import cv2
import mediapipe as mp

# MediaPipe Tasks
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# --------------------------------------------------
# Load Face Landmarker
# --------------------------------------------------

base_options = python.BaseOptions(
    model_asset_path="models/face_landmarker.task"
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
)

landmarker = vision.FaceLandmarker.create_from_options(options)


# --------------------------------------------------
# Open Camera
# --------------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("Camera started.")
print("Press ESC to exit.")


# MediaPipe video timestamps must increase
timestamp_ms = 0


# --------------------------------------------------
# Main Loop
# --------------------------------------------------

while True:

    success, frame = cap.read()

    if not success:
        print("ERROR: Could not read camera frame.")
        break

    # Mirror the camera
    frame = cv2.flip(frame, 1)

    # OpenCV uses BGR
    # MediaPipe expects RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Convert to MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect landmarks
    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )

    timestamp_ms += 33


    # --------------------------------------------------
    # Draw landmarks
    # --------------------------------------------------

    if result.face_landmarks:

        for face_landmarks in result.face_landmarks:

            h, w, _ = frame.shape

            for landmark in face_landmarks:

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                cv2.circle(
                    frame,
                    (x, y),
                    1,
                    (0, 255, 0),
                    -1
                )


        # Display number of detected faces
        cv2.putText(
            frame,
            "FACE DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    else:

        cv2.putText(
            frame,
            "NO FACE",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )


    # Show camera
    cv2.imshow(
        "Quadruped Face AI",
        frame
    )


    # ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break


# --------------------------------------------------
# Cleanup
# --------------------------------------------------

cap.release()
cv2.destroyAllWindows()
landmarker.close()

print("Camera stopped.")