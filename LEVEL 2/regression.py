import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ── Load Data ───────────────────────────────────────────────────
columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
           "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]

df = pd.read_csv(r"C:\Users\Student\OneDrive - University of KwaZulu-Natal\Desktop\Data Science Folder\Data-science-intern\LEVEL 2\4) house Prediction Data Set.csv",
                 sep=r"\s+", engine="python", header=None, names=columns)

# ── Step 1: Split Features & Target ────────────────────────────
X = df.drop(columns=["MEDV"])  # features
y = df["MEDV"]                 # target (house price)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"Training set: {X_train.shape}")
print(f"Testing set:  {X_test.shape}")

# ── Step 2: Train 3 Models ──────────────────────────────────────
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree":     DecisionTreeRegressor(random_state=42),
    "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)
    results[name] = {"RMSE": round(rmse, 3), "R²": round(r2, 3)}
    print(f"\n{name}")
    print(f"  RMSE : {rmse:.3f}")
    print(f"  R²   : {r2:.3f}")

# ── Step 3: Visualizations ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Regression Model Comparison", fontsize=14, fontweight="bold")

for ax, (name, model) in zip(axes, models.items()):
    y_pred = model.predict(X_test)
    ax.scatter(y_test, y_pred, alpha=0.5, color="#3498db", s=20)
    ax.plot([y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()], "r--", linewidth=2)
    ax.set_title(f"{name}\nR²={results[name]['R²']}  RMSE={results[name]['RMSE']}")
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")

plt.tight_layout()
plt.savefig("regression_results.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Step 4: Feature Importance (Random Forest) ─────────────────
rf_model = models["Random Forest"]
importance = pd.Series(rf_model.feature_importances_, index=X.columns)
importance = importance.sort_values(ascending=True)

plt.figure(figsize=(8, 6))
importance.plot(kind="barh", color="#2ecc71", edgecolor="black")
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n Plots saved!")
print("\n    Model Comparison Summary   ")
print(pd.DataFrame(results).T)