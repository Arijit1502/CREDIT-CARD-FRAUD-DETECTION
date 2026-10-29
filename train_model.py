import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score
)

# Step 1: Load dataset
df = pd.read_csv("creditcard.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# Step 2: Separate features and target
X = df.drop(columns=["Class"])
y = df["Class"]

print("\nTransaction classes:")
print(y.value_counts())

# Step 3: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# Step 4: Scale Time and Amount
preprocessor = ColumnTransformer(
    transformers=[
        ("scaler", StandardScaler(), ["Time", "Amount"])
    ],
    remainder="passthrough"
)

# Step 5: Create machine learning models
models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

best_model = None
best_score = -1
best_name = None

# Step 6: Train and evaluate the models
for name, classifier in models.items():

    print("\n" + "=" * 50)
    print("Training:", name)

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    model.fit(X_train, y_train)

    # Predict probabilities
    probabilities = model.predict_proba(X_test)[:, 1]

    # Classify transactions using a threshold of 0.5
    predictions = (probabilities >= 0.5).astype(int)

    # Calculate average precision
    score = average_precision_score(y_test, probabilities)

    print("Average Precision:", round(score, 4))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)

    print("\nConfusion Matrix:")
    print(cm)

    # Plot confusion matrix
    plt.figure(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.tight_layout()

    filename = name.lower().replace(" ", "_")
    plt.savefig(f"{filename}_confusion_matrix.png")
    plt.close()

    # Select the model with the highest average precision
    if score > best_score:
        best_score = score
        best_model = model
        best_name = name

# Step 7: Save the best model
joblib.dump(best_model, "fraud_model.joblib")

print("\n" + "=" * 50)
print("Training completed!")
print("Best model:", best_name)
print("Best Average Precision:", round(best_score, 4))
print("Model saved as fraud_model.joblib")