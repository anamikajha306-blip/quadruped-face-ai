import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

print("Loading dataset...")

df = pd.read_csv("training_data.csv")

X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Training SVM...")

model = make_pipeline(
    StandardScaler(),
    SVC(
        kernel="rbf",
        C=10,
        gamma="scale"
    )
)

model.fit(X_train, y_train)

print("Training complete!")

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print()
print("==============================")
print("SVM ACCURACY:", accuracy)
print("==============================")
print()

print(classification_report(y_test, predictions))

joblib.dump(model, "emotion_model_svm.joblib")

print("Saved: emotion_model_svm.joblib")