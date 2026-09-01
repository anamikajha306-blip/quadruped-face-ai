import os
import csv
import random
import cv2
import math
import mediapipe as mp

DATASET_DIR = "dataset/train"
OUTPUT_FILE = "expression_features.csv"
IMAGES_PER_CLASS = 2000

CLASSES = {
    "Angry": "angry",
    "Fear": "tensed",
    "Neutral": "neutral",
    "Sad": "sad",
}

# MediaPipe setup
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


def extract_features(image_path, landmarker):

    image = cv2.imread(image_path)

    if image is None:
        return None

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_rgb
    )

    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    lm = result.face_landmarks[0]

    # Important MediaPipe facial landmarks
    # Left/right eye
    left_eye_top = lm[159]
    left_eye_bottom = lm[145]

    right_eye_top = lm[386]
    right_eye_bottom = lm[374]

    # Eyebrows
    left_brow = lm[105]
    right_brow = lm[334]

    # Mouth
    mouth_top = lm[13]
    mouth_bottom = lm[14]
    mouth_left = lm[61]
    mouth_right = lm[291]

    # Nose
    nose_top = lm[6]
    nose_bottom = lm[2]

    # Face width
    face_left = lm[234]
    face_right = lm[454]

    # Face height
    face_top = lm[10]
    face_bottom = lm[152]

    # Feature measurements
    features = [

        # Eye openness
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

        # Mouth openness
        ratio(
            mouth_top,
            mouth_bottom,
            face_top,
            face_bottom
        ),

        # Mouth width
        ratio(
            mouth_left,
            mouth_right,
            face_left,
            face_right
        ),

        # Nose length
        ratio(
            nose_top,
            nose_bottom,
            face_top,
            face_bottom
        ),

        # Eyebrow distances
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

        # Mouth-to-nose relationship
        ratio(
            mouth_top,
            nose_bottom,
            face_top,
            face_bottom
        ),

        # Overall face proportions
        ratio(
            face_top,
            face_bottom,
            face_left,
            face_right
        ),

        # Distance between eyebrows
        ratio(
            left_brow,
            right_brow,
            face_left,
            face_right
        ),
    ]

    return features


def main():

    print("Starting expression feature extraction...")
    print()

    rows = []
    failed = 0

    with FaceLandmarker.create_from_options(options) as landmarker:

        for folder_name, label in CLASSES.items():

            folder = os.path.join(
                DATASET_DIR,
                folder_name
            )

            images = [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith(
                    (".png", ".jpg", ".jpeg")
                )
            ]

            random.seed(42)
            random.shuffle(images)

            images = images[:IMAGES_PER_CLASS]

            print(
                f"{label}: processing {len(images)} images"
            )

            successful = 0

            for i, image_path in enumerate(images):

                features = extract_features(
                    image_path,
                    landmarker
                )

                if features is None:
                    failed += 1
                    continue

                rows.append(
                    features + [label]
                )

                successful += 1

                if (i + 1) % 100 == 0:
                    print(
                        f"  {i + 1}/{len(images)}"
                    )

            print(
                f"  Successful: {successful}"
            )
            print()

    if not rows:
        print("ERROR: No features extracted.")
        return

    header = [
        "eye_left",
        "eye_right",
        "mouth_open",
        "mouth_width",
        "nose_length",
        "left_brow_eye",
        "right_brow_eye",
        "mouth_nose",
        "face_proportion",
        "brow_distance",
        "label"
    ]

    with open(
        OUTPUT_FILE,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print("==============================")
    print("FEATURE EXTRACTION COMPLETE")
    print("==============================")
    print(f"Samples: {len(rows)}")
    print(f"Failed:  {failed}")
    print(f"Features: {len(header) - 1}")
    print(f"Saved: {OUTPUT_FILE}")
    print("==============================")


if __name__ == "__main__":
    main()