"""Band gap predictor"""

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, classification_report
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
import os

# Image directory
os.makedirs("model_plots_metal", exist_ok=True)

# Loading the model
data = pd.read_csv("df_model_dataset.csv")

data_cols = ["is_direct_gap", "density_g_cm3", "n_sites", 
             "volume_A3", "mag_per_atom", "n_elements", "spacegroup_number", "crystal_system", "dos_integral_vb", "dos_peak_count", "dos_vbm_slope", "dos_mean_density"]

# Data split 
X = data[data_cols]

Y = data["band_gap_eV"]

bins = [0, 0.0001, 1.5000, 2.9983, Y.max()]
labels = ["metals", "narrow_gap_semiconductor", "semiconductor", "insulator"]
y_cat = pd.cut(Y, bins=bins, labels=labels , include_lowest=True)
X_train, X_test, Y_train, Y_test, y_cat_train, y_cat_test = train_test_split(X, Y, y_cat, test_size=0.20, random_state=42, stratify=y_cat)

# Classifier
param_dist_cls = {
    "max_depth": [5, 7, 10, 12],
    "min_samples_leaf": [1, 2, 5, 10],
    "max_features": ["sqrt", "log2", 0.3, 0.5],
    "n_estimators": [100, 200]
}

cls = RandomForestClassifier(random_state=42)

rs_cls = RandomizedSearchCV(
    estimator=cls,
    param_distributions=param_dist_cls, 
    n_iter=20,
    cv=5,
    scoring="f1_weighted",
    random_state=42,
    n_jobs=-1
)

rs_cls.fit(X_train, y_cat_train)

best_params_cls = rs_cls.best_params_

print("Best params:", best_params_cls)
print("Best CV accuracy:", rs_cls.best_score_)


Y_pred_train_cls = rs_cls.best_estimator_.predict(X_train)
Y_pred_test_cls = rs_cls.best_estimator_.predict(X_test)


# Regressor 
param_dist = {
    "max_depth": [5, 7, 10, 12],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "n_estimators": [100, 200], 
    "learning_rate": [0.01, 0.005, 0.1, 0.2, 0.3],
    "reg_alpha": [0, 0.01, 0.1, 1.0],
    "reg_lambda": [0, 1.0, 5.0, 10.0]
}


rf = XGBRegressor(random_state=42)

rs = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist, 
    n_iter=20, 
    cv=5, 
    scoring="neg_root_mean_squared_error", 
    random_state=42,
    n_jobs=-1 
    )

rs.fit(X_train, Y_train)

best_params = rs.best_params_

print("Best params:", best_params)
print("Best CV RMSE:", -rs.best_score_)

classes = ["metals", "narrow_gap_semiconductor", "semiconductor", "insulator"]
regressors = {}

for cls_name in classes:
    mask = y_cat_train == cls_name 
    reg=XGBRegressor(**best_params, random_state=42) 
    reg.fit(X_train[mask], Y_train[mask]) 
    regressors[cls_name] = reg 

Y_pred_test = np.zeros(len(X_test)) 
Y_pred_train = np.zeros(len(X_train))

for cls_name in classes:
    mask_train = Y_pred_train_cls == cls_name
    mask_test = Y_pred_test_cls == cls_name
    if mask_train.sum() > 0:
        Y_pred_train[mask_train] = regressors[cls_name].predict(X_train[mask_train])
    if mask_test.sum() > 0:
        Y_pred_test[mask_test] = regressors[cls_name].predict(X_test[mask_test])

# Metrics
r2_train = r2_score(Y_train, Y_pred_train)
r2_test = r2_score(Y_test, Y_pred_test)
RMSE_train = np.sqrt(mean_squared_error(Y_train, Y_pred_train))
RMSE_test = np.sqrt(mean_squared_error(Y_test, Y_pred_test))
MAE_train = mean_absolute_error(Y_train, Y_pred_train)
MAE_test = mean_absolute_error(Y_test, Y_pred_test)

print(f"Train R^2: {r2_train} | Test R^2: {r2_test}")
print(f"Train RMSE: {RMSE_train} | Test RMSE: {RMSE_test}")
print(f"Train MAE: {MAE_train} | Test MAE: {MAE_test}")

# Graphics
fig, ax = plt.subplots(figsize=(8,5))
ax.scatter(Y_test, Y_pred_test, alpha=0.3, s=5)
ax.plot([0, 10], [0, 10], 'r--', linewidth=2)
ax.set_xlabel("Y_test")
ax.set_ylabel("Y_pred_test")
ax.set_title("Prediction vs actual")
plt.savefig("model_plots/Prediction_Graph.png", dpi=150, bbox_inches="tight")
plt.close()

residuals = Y_test - Y_pred_test
plt.hist(residuals, bins=80, edgecolor='white', linewidth=0.5)
plt.axvline(0, color='red', linestyle='--')
plt.xlabel("Residual (Actual - Predicted)")
plt.ylabel("Count")
plt.title("Residual Distribution")
plt.savefig("model_plots/residual.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"Residual mean: {residuals.mean():.4f}")
print(f"Residual std: {residuals.std():.4f}")

residuals = Y_test - Y_pred_test
worst = pd.DataFrame({
    "actual": Y_test,
    "predicted": Y_pred_test,
    "residual": residuals,
    "abs_residuals": np.abs(residuals),
})

worst = worst.sort_values("abs_residuals", ascending=False).head(20)
print(worst)

# Best Predictors 
importances = pd.Series(rs.best_estimator_.feature_importances_, 
                         index=X.columns)
print(importances.sort_values(ascending=False).head(20))

importances = pd.Series(rs.best_estimator_.feature_importances_, 
                         index=X.columns)
print(f"Features with importance < 0.001: {(importances < 0.001).sum()}")
print(f"Features with importance < 0.005: {(importances < 0.005).sum()}")
print(f"Total features: {len(importances)}")