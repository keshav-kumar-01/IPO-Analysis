# ============================
# Step 7: Hyperparameter Tuning & Feature Importance (with model saving)
# ============================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pickle  # for saving models

# ----------------------------
# Load feature-engineered dataset
# ----------------------------
df = pd.read_csv(r"D:\IPO_analysis\data\ipo_features.csv")
target = "is_listing_win"

# Drop non-features
X = df.drop([target, "Date", "IPO_Name"], axis=1)
y = df[target]

# Impute missing values
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Keep feature names
feature_names = X.columns.tolist()

# ----------------------------
# 1️⃣ Random Forest Hyperparameter Tuning
# ----------------------------
rf = RandomForestClassifier(random_state=42)

param_grid_rf = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

rf_search = RandomizedSearchCV(
    rf, param_distributions=param_grid_rf,
    n_iter=20, cv=5, scoring='accuracy', random_state=42, n_jobs=-1
)
rf_search.fit(X_train_scaled, y_train)
best_rf = rf_search.best_estimator_

# Evaluate best RF
y_pred_rf = best_rf.predict(X_test_scaled)
print("🔥 Random Forest Best Params:", rf_search.best_params_)
print("Accuracy:", round(accuracy_score(y_test, y_pred_rf)*100, 2), "%")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

# Save RF model + preprocessing
rf_to_save = {
    "model": best_rf,
    "features": feature_names,
    "imputer": imputer,
    "scaler": scaler
}
with open(r"D:\IPO_analysis\models\best_rf_model.pkl", "wb") as f:
    pickle.dump(rf_to_save, f)
print("✅ Random Forest model + pipeline saved as best_rf_model.pkl")

# Feature Importance
importances = best_rf.feature_importances_
feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_imp = feat_imp.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=feat_imp)
plt.title("Random Forest Feature Importances")
plt.show()

# ----------------------------
# 2️⃣ XGBoost Hyperparameter Tuning
# ----------------------------
xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

xgb_search = RandomizedSearchCV(
    xgb, param_distributions=param_grid_xgb,
    n_iter=20, cv=5, scoring='accuracy', random_state=42, n_jobs=-1
)
xgb_search.fit(X_train_scaled, y_train)
best_xgb = xgb_search.best_estimator_

# Evaluate best XGBoost
y_pred_xgb = best_xgb.predict(X_test_scaled)
print("🔥 XGBoost Best Params:", xgb_search.best_params_)
print("Accuracy:", round(accuracy_score(y_test, y_pred_xgb)*100, 2), "%")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))
print("Classification Report:\n", classification_report(y_test, y_pred_xgb))

# Save XGBoost model + preprocessing
xgb_to_save = {
    "model": best_xgb,
    "features": feature_names,
    "imputer": imputer,
    "scaler": scaler
}
with open(r"D:\IPO_analysis\models\best_xgb_model.pkl", "wb") as f:
    pickle.dump(xgb_to_save, f)
print("✅ XGBoost model + pipeline saved as best_xgb_model.pkl")

# Feature Importance
xgb_importances = best_xgb.feature_importances_
feat_imp_xgb = pd.DataFrame({'Feature': feature_names, 'Importance': xgb_importances})
feat_imp_xgb = feat_imp_xgb.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=feat_imp_xgb)
plt.title("XGBoost Feature Importances")
plt.show()
