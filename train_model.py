import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("training_data.csv")

print("Dataset loaded:", df.shape)

# Separate features and labels
X = df.drop("label", axis=1)
y = df["label"]

print("Classes:", y.unique())

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Create model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

print("Training model...")

model.fit(X_train, y_train)

print("Training complete!")

# Test
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print()
print("================================")
print("MODEL ACCURACY:", accuracy)
print("================================")
print()

print(classification_report(y_test, predictions))

# Save model
joblib.dump(model, "emotion_model.joblib")

print("Model saved as:")
print("emotion_model.joblib")