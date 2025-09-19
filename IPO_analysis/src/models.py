# ============================
# Step 6: Model Training & Evaluation (with model saving)
# ============================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pickle  # <-- Added for saving models

# ----------------------------
# Load feature-engineered dataset
# ----------------------------
df = pd.read_csv(r"D:\IPO_analysis\data\ipo_features.csv")

# ----------------------------
# Select target column
# ----------------------------
target = "is_listing_win"  # Predict if IPO had positive listing gain

# ----------------------------
# Prepare features and target
# ----------------------------
X = df.drop([target, "Date", "IPO_Name"], axis=1)
y = df[target]

# ----------------------------
# Impute missing values
# ----------------------------
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# ----------------------------
# Train-test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# Scale features
# ----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# Models to evaluate
# ----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=200, random_state=42, use_label_encoder=False, eval_metric='logloss')
}

results = {}

# ----------------------------
# Train, Evaluate & Save Models
# ----------------------------
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    
    print(f"\n📊 {name} Results:")
    print("Accuracy:", round(acc*100, 2), "%")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    # Save model as .pkl
    model_filename = f"D:\\IPO_analysis\\models\\{name.replace(' ', '_').lower()}_model.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Model saved as {model_filename}")

# ----------------------------
# Compare model accuracies
# ----------------------------
plt.figure(figsize=(8,5))
sns.barplot(x=list(results.keys()), y=list(results.values()))
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=20)
plt.show()
