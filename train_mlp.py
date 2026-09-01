import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
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
print("Training neural network...")

model = make_pipeline(
    StandardScaler(),
    MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        batch_size=64,
        learning_rate_init=0.001,
        max_iter=50,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
        verbose=True
    )
)

model.fit(X_train, y_train)

print("Training complete!")

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print()
print("==============================")
print("MLP ACCURACY:", accuracy)
print("==============================")
print()

print(classification_report(y_test, predictions))

joblib.dump(model, "emotion_model_mlp.joblib")

print("Saved: emotion_model_mlp.joblib")