import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, roc_curve, auc)

#Load Dataset
train_df = pd.read_csv(r"C:\Users\Student\OneDrive - University of KwaZulu-Natal\Desktop\Data Science Folder\Data-science-intern\LEVEL 2\churn-bigml-80.csv")
test_df  = pd.read_csv(r"C:\Users\Student\OneDrive - University of KwaZulu-Natal\Desktop\Data Science Folder\Data-science-intern\LEVEL 2\churn-bigml-20.csv")

#1.Preprocessing
def preprocess(df):
    df = df.copy()
    # Encode binary columns
    df["International plan"] = df["International plan"].map({"Yes": 1, "No": 0})
    df["Voice mail plan"]    = df["Voice mail plan"].map({"Yes": 1, "No": 0})
    # Encode State with LabelEncoder
    le = LabelEncoder()
    df["State"] = le.fit_transform(df["State"])
    # Convert Churn bool to int
    df["Churn"] = df["Churn"].astype(int)
    return df

train_df = preprocess(train_df)
test_df  = preprocess(test_df)

X_train = train_df.drop(columns=["Churn"])
y_train = train_df["Churn"]
X_test  = test_df.drop(columns=["Churn"])
y_test  = test_df["Churn"]

# Scale features
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(" Preprocessing done")
print(f"   Train: {X_train.shape} | Test: {X_test.shape}")
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM":                 SVC(probability=True, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        "Accuracy":  round(accuracy_score(y_test, y_pred), 3),
        "Precision": round(precision_score(y_test, y_pred), 3),
        "Recall":    round(recall_score(y_test, y_pred), 3),
        "F1 Score":  round(f1_score(y_test, y_pred), 3)
    }
    print(f"\n{name}")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

#  Step 3: ROC Curves 
plt.figure(figsize=(8, 6))
colors = ["#3498db", "#2ecc71", "#e74c3c"]

for (name, model), color in zip(models.items(), colors):
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2,
             label=f"{name} (AUC = {roc_auc:.3f})")

plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — Churn Prediction")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=150, bbox_inches="tight")
plt.show()

#4: Model Comparison Chart 
results_df = pd.DataFrame(results).T
print("\n    Model Comparison ")
print(results_df)

results_df.plot(kind="bar", figsize=(10, 5), edgecolor="black")
plt.title("Model Comparison — Classification Metrics")
plt.xlabel("Model")
plt.ylabel("Score")
plt.xticks(rotation=15)
plt.ylim(0, 1)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n All plots saved!")