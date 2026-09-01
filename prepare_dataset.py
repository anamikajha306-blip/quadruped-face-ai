import os
import csv
import random
import cv2
import mediapipe as mp

# ==============================
# SETTINGS
# ==============================

DATASET_DIR = "dataset/train"
OUTPUT_FILE = "training_data.csv"

# Number of images per expression
IMAGES_PER_CLASS = 2000

# Our four classes
CLASSES = {
    "Angry": "angry",
    "Fear": "tensed",
    "Neutral": "neutral",
    "Sad": "sad",
}

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
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1,
)

# ==============================
# LANDMARK FEATURE EXTRACTION
# ==============================

def extract_features(image_path, landmarker):

    image = cv2.imread(image_path)

    if image is None:
        return None

    # OpenCV uses BGR, MediaPipe expects RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_rgb
    )

    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    landmarks = result.face_landmarks[0]

    # Use x, y, z for every landmark
    features = []

    # Normalize around the nose landmark (landmark 1)
    nose = landmarks[1]

    for landmark in landmarks:
        features.append(landmark.x - nose.x)
        features.append(landmark.y - nose.y)
        features.append(landmark.z - nose.z)

    return features


# ==============================
# MAIN
# ==============================

def main():

    print("Starting dataset preparation...")
    print()

    all_rows = []
    total_processed = 0
    total_failed = 0

    with FaceLandmarker.create_from_options(options) as landmarker:

        for folder_name, label in CLASSES.items():

            folder = os.path.join(DATASET_DIR, folder_name)

            if not os.path.exists(folder):
                print(f"ERROR: Folder not found: {folder}")
                continue

            images = [
                os.path.join(folder, filename)
                for filename in os.listdir(folder)
                if filename.lower().endswith((".png", ".jpg", ".jpeg"))
            ]

            random.seed(42)
            random.shuffle(images)

            images = images[:IMAGES_PER_CLASS]

            print(f"{folder_name} -> {label}")
            print(f"Images selected: {len(images)}")

            class_count = 0

            for i, image_path in enumerate(images):

                features = extract_features(
                    image_path,
                    landmarker
                )

                if features is None:
                    total_failed += 1
                    continue

                all_rows.append(
                    features + [label]
                )

                class_count += 1
                total_processed += 1

                if (i + 1) % 100 == 0:
                    print(
                        f"  Processed {i + 1}/{len(images)}"
                    )

            print(
                f"  Successful: {class_count}"
            )
            print()

    # ==============================
    # SAVE CSV
    # ==============================

    if not all_rows:
        print("ERROR: No facial landmarks were extracted.")
        return

    number_of_features = len(all_rows[0]) - 1

    header = [
        f"feature_{i}"
        for i in range(number_of_features)
    ]

    header.append("label")

    with open(
        OUTPUT_FILE,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow(header)
        writer.writerows(all_rows)

    print("==============================")
    print("DATASET PREPARATION COMPLETE")
    print("==============================")
    print(f"Successful samples: {total_processed}")
    print(f"Failed samples:     {total_failed}")
    print(f"Features per face:  {number_of_features}")
    print(f"Output file:        {OUTPUT_FILE}")
    print("==============================")


if __name__ == "__main__":
    main()