"""Data Preparation"""

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, classification_report
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
import os

# what i HAVE TO DO:
# add the first dataset so I can see if I get better values when I divide into metal/non metal,
# Change accordingly so i remove the classifier and turn it into an metal/non metal classifier
# Change feature engineer to do that in the first dataset 

os.makedirs("model_plots_metal", exist_ok=True)

data = pd.read_csv("df_model_dataset.csv")

data_cols = ["is_direct_gap", "density_g_cm3", "n_sites", 
             "volume_A3", "mag_per_atom", "n_elements", "spacegroup_number", "crystal_system", "dos_integral_vb", "dos_peak_count", "dos_vbm_slope", "dos_mean_density"]


X = data[data_cols]

Y = data["band_gap_eV"]

print(data.groupby("material_class")["band_gap_eV"].min())

bins = [0, 0.0001, 1.5000, 2.9983, Y.max()]
labels = ["metals", "narrow_gap_semiconductor", "semiconductor", "insulator"]
y_cat = pd.cut(Y, bins=bins, labels=labels , include_lowest=True)

print(y_cat)
print(y_cat.value_counts(normalize=True))

X_train, X_test, Y_train, Y_test, y_cat_train, y_cat_test = train_test_split(X, Y, y_cat, test_size=0.20, random_state=42, stratify=y_cat)

print((Y_test < 0.1).sum())
print((Y_test < 0.5).sum())
print(X_train.head())
print("----")
print(X_test.head())
print("----")
print(Y_train.head())
print("----")
print(Y_test.head())

# param_dist_cls = {
#     "max_depth": [5, 7, 10, 12],
#     "min_samples_leaf": [1, 2, 5, 10],
#     "max_features": ["sqrt", "log2", 0.3, 0.5],
#     "n_estimators": [100, 200]
# }

param_dist = {
    "max_depth": [5, 7, 10, 12],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "n_estimators": [100, 200], 
    "learning_rate": [0.01, 0.005, 0.1, 0.2, 0.3],
    "reg_alpha": [0, 0.01, 0.1, 1.0],
    "reg_lambda": [0, 1.0, 5.0, 10.0]
}

# cls = RandomForestClassifier(random_state=42)

# rs_cls = RandomizedSearchCV(
#     estimator=cls,
#     param_distributions=param_dist_cls, 
#     n_iter=20,
#     cv=5,
#     scoring="f1_weighted",
#     random_state=42,
#     n_jobs=-1
# )


# rs_cls.fit(X_train, y_cat_train)

# best_params_cls = rs_cls.best_params_

# print("Best params:", best_params_cls)
# print("Best CV accuracy:", rs_cls.best_score_)


# Y_pred_train_cls = rs_cls.best_estimator_.predict(X_train)
# Y_pred_test_cls = rs_cls.best_estimator_.predict(X_test)


# print(classification_report(y_cat_test, Y_pred_test_cls))

# print(pd.Series(Y_pred_test_cls).value_counts(normalize=True))

# print(pd.crosstab(y_cat_test, Y_pred_test_cls, normalize="index"))

# print((y_cat_train == "semiconductor").sum())
# print((y_cat_train == "narrow_gap_semiconductor").sum())
# print((y_cat_train == "insulator").sum())

# Model used
rf = XGBRegressor(random_state=42)

# Search for best hyperperameters
rs = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist, 
    n_iter=20, # test in 20 different param_dist combinations
    cv=5, #cross-val (splits data into 4 parts, trains on 4 and validates on 1 -> repeats 5 times)
    scoring="neg_root_mean_squared_error", #metric to evaluate the model
    random_state=42,
    n_jobs=-1 #Use all CPU cores so it runs faster 
    )

# Debug: check data types
print("X_train dtypes:")
print(X_train.dtypes)
print("\nY_train type and dtype:")
print(type(Y_train), Y_train.dtype)
print("\nX_train info:")
print(X_train.info())

rs.fit(X_train, Y_train)

best_params = rs.best_params_

print("Best params:", best_params)
print("Best CV RMSE:", -rs.best_score_)

classes = ["narrow_gap_semiconductor", "semiconductor", "insulator"]
regressors = {}

for cls_name in classes:
    mask = y_cat_train == cls_name # the mask is when the Y_train is insulator, semiconductor, ng semiconductor
    reg=XGBRegressor(**best_params, random_state=42) # it applies the regressor with the best params
    reg.fit(X_train[mask], Y_train[mask]) # I train only the values from the classes
    regressors[cls_name] = reg # I store each model corresponding with a class -> model 1, ng semiconductor; model 2, semiconductor; model 3, insultor



# Y_pred_test = np.zeros(len(X_test)) # empty arrays to store predictions
# Y_pred_train = np.zeros(len(X_train))

# for cls_name in classes:
#     mask_train = Y_pred_train_cls == cls_name
#     mask_test = Y_pred_test_cls == cls_name
#     if mask_train.sum() > 0: # only if there a samples
#         Y_pred_train[mask_train] = regressors[cls_name].predict(X_train[mask_train]) # predict using the models in the regressors for the class 
#     if mask_test.sum() > 0: 
#         Y_pred_test[mask_test] = regressors[cls_name].predict(X_test[mask_test])


Y_pred_train = rs.predict(X_train)
Y_pred_test = rs.predict(X_test)



r2_train = r2_score(Y_train, Y_pred_train)
r2_test = r2_score(Y_test, Y_pred_test)
RMSE_train = np.sqrt(mean_squared_error(Y_train, Y_pred_train))
RMSE_test = np.sqrt(mean_squared_error(Y_test, Y_pred_test))
MAE_train = mean_absolute_error(Y_train, Y_pred_train)
MAE_test = mean_absolute_error(Y_test, Y_pred_test)

print(f"Train R^2: {r2_train} | Test R^2: {r2_test}")
print(f"Train RMSE: {RMSE_train} | Test RMSE: {RMSE_test}")
print(f"Train MAE: {MAE_train} | Test MAE: {MAE_test}")


fig, ax = plt.subplots(figsize=(8,5))
ax.scatter(Y_test, Y_pred_test, alpha=0.3, s=5)
ax.plot([0, 10], [0, 10], 'r--', linewidth=2)
ax.set_xlabel("Y_pred_test")
ax.set_ylabel("Y_test")
ax.set_title("Prediction vs actual")
plt.savefig("model_plots/firstpred.png", dpi=150, bbox_inches="tight")
plt.close()


importances = pd.Series(rs.best_estimator_.feature_importances_, 
                         index=X.columns)
print(importances.sort_values(ascending=False).head(20))

importances = pd.Series(rs.best_estimator_.feature_importances_, 
                         index=X.columns)
print(f"Features with importance < 0.001: {(importances < 0.001).sum()}")
print(f"Features with importance < 0.005: {(importances < 0.005).sum()}")
print(f"Total features: {len(importances)}")

# Keep only features with importance above threshold
# threshold = 0.005
# selected_features = importances[importances >= threshold].index.tolist()
# print(f"Selected {len(selected_features)} features:")
# print(selected_features)

# X_train_selected = X_train[selected_features]
# X_test_selected = X_test[selected_features]


# rf = XGBRegressor(random_state=42)

# rs = RandomizedSearchCV(
#     estimator=rf,
#     param_distributions=param_dist, 
#     n_iter=20,
#     cv=5,
#     scoring="neg_root_mean_squared_error",
#     random_state=42,
#     n_jobs=-1
#     )

# rs.fit(X_train_selected, Y_train)

# Y_pred_train = rs.predict(X_train_selected)
# Y_pred_test = rs.predict(X_test_selected)

# r2_train = r2_score(Y_train, Y_pred_train)
# r2_test = r2_score(Y_test, Y_pred_test)
# RMSE_train = np.sqrt(mean_squared_error(Y_train, Y_pred_train))
# RMSE_test = np.sqrt(mean_squared_error(Y_test, Y_pred_test))
# MAE_train = mean_absolute_error(Y_train, Y_pred_train)
# MAE_test = mean_absolute_error(Y_test, Y_pred_test)

# print(f"--- Results with selected features ---")
# print(f"Train R^2: {r2_train} | Test R^2: {r2_test}")
# print(f"Train RMSE: {RMSE_train} | Test RMSE: {RMSE_test}")
# print(f"Train MAE: {MAE_train} | Test MAE: {MAE_test}")
# print("Best params:", rs.best_params_)


# fig, ax = plt.subplots(figsize=(8,5))
# ax.scatter(Y_test, Y_pred_test, alpha=0.3, s=5)
# ax.plot([0, 10], [0, 10], 'r--', linewidth=2)
# ax.set_xlabel("Y_pred_test")
# ax.set_ylabel("Y_test")
# ax.set_title("Prediction vs actual")
# plt.savefig("model_plots/prediction_graph.png", dpi=150, bbox_inches="tight")
# plt.close()

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
    #"true_class": y_cat_test,
    #"predicted_class": Y_pred_test_cls
})

worst = worst.sort_values("abs_residuals", ascending=False).head(20)
print(worst)
