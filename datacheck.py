""" Band Gap Data Checker """

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from matminer.featurizers.composition import ElementProperty
from pymatgen.core.composition import Composition



# Figure Folder
os.makedirs("eda_plots", exist_ok=True)

# Read the dataset
df = pd.read_csv("band_gap_dataset.csv")

# Print several things of the dataset
print(df.shape)
print(df.head())
print(df.dtypes)

# Defining what the predictors and what is beeing predicted

Y = df["band_gap_eV"]

X = df[["formation_energy_eV_atom", "is_direct_gap", "density_g_cm3", "n_elements", "n_sites", "volume_A3", "total_magnetization", "cbm_eV", "vbm_eV"]]

print(Y.shape)
print(Y.head())
print(X.shape)
print(X.head())

# Histogram of the energy bandgap
df["band_gap_eV"].hist(bins=80)
plt.savefig("eda_plots/bandgap_histogram.png", dpi = 150, bbox_inches="tight")
plt.close()

# Check which values are 0 (metals) and above 0 (non-metals)

print((df["band_gap_eV"] == 0).sum())
print((df["band_gap_eV"] > 0).sum())

# Create a list with the non-metal (for the second part of predicting the energy gap if they are non-metal)

df_nonmetal = df[df["band_gap_eV"] > 0.0].copy()
print(df_nonmetal)

idx = (df_nonmetal["band_gap_eV"] > 1.4) & (df_nonmetal["band_gap_eV"] < 1.6)
print(df_nonmetal.loc[idx, ["formula", "band_gap_eV", "material_class"]])

idx_1 = df_nonmetal["band_gap_eV"] == 2.9983
print(df_nonmetal.loc[idx_1, ["formula", "band_gap_eV", "material_class"]])

# Histograms of the logged energy bandgap and the energy bandgap
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

df_nonmetal["log_bandgap_eV"] = np.log(df_nonmetal["band_gap_eV"]) # log of the 

df_nonmetal["band_gap_eV"].hist(bins=100, ax=axes[0], linewidth=0.5, edgecolor="white")
df_nonmetal["log_bandgap_eV"].hist(bins=100, ax=axes[1], linewidth=0.5, edgecolor="white")

axes[0].set_title("Raw Band Gap")
axes[1].set_title("Log Band Gap")
axes[0].set_xlabel("Band Gap (eV)")
axes[1].set_xlabel("log(Band Gap)")
plt.savefig("eda_plots/bandgap_log_bandgap_histogram.png", dpi = 150, bbox_inches="tight")
plt.close()

# Other checks
print(df_nonmetal["band_gap_eV"].skew()) # The skew of each gra+h
print(df_nonmetal["log_bandgap_eV"].skew())

print(df_nonmetal["band_gap_eV"].describe()) # Check different parameters

# Check the minimum value that we have to see if it corresponds to a metal or non-metal
idx = df_nonmetal["band_gap_eV"].idxmin() 
print(df_nonmetal.loc[idx])

print(df_nonmetal.loc[idx, ["formula", "vbm_eV", "cbm_eV", "band_gap_eV", "crystal_system", "material_class"]])

# Check if we have any values at NaN
print(df_nonmetal.isnull().any())

# Correlation plot to check how features correlate with each other
corr = df_nonmetal[["band_gap_eV", "formation_energy_eV_atom", 
                     "density_g_cm3", "n_elements", "n_sites", 
                     "volume_A3", "total_magnetization", 
                     "cbm_eV", "vbm_eV", "is_direct_gap"]].corr()

mask = np.triu(np.ones_like(corr, dtype=bool)) # mask to get only the upper triangle (can remove if needed)

fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(corr,
            mask=mask, 
            annot=True,      # show numbers inside cells
            fmt=".2f",       # 2 decimal places
            cmap="coolwarm", # blue=negative, red=positive
            center=0,        # white = zero correlation
            ax=ax)

ax.set_title("Correlation Matrix")
plt.tight_layout()
plt.savefig("eda_plots/correlation_matrix.png", dpi = 150, bbox_inches="tight")
plt.close()

# Non-visual check to see which parameters have high correlation so we exclude/don't exclude redudant features depending on the model

threshold = 0.7
pair = []

for i in range(len(corr.columns)):
    for j in range((i+1), len(corr.columns)):
        if abs(corr.iloc[i,j]) > threshold:
            pair.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))

for a,b, val in pair:         
    print(f"{a} vs {b} -> {val:.3f}")

# Scatter plot of each feature vs the target 

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df_nonmetal["vbm_eV"], df_nonmetal["band_gap_eV"])
ax.set_xlabel("VBM (eV)")
ax.set_ylabel("Band Gap (eV)")
ax.set_title("VBM vs Band Gap")
plt.savefig("eda_plots/vbm_vs_bandgap.png", dpi = 150, bbox_inches="tight")
plt.close()

# I have an outlier in the vbm vs bandgap graph so I'll check it

idx = df_nonmetal["band_gap_eV"].idxmax()
print(print(df_nonmetal.loc[idx, ["formula", "band_gap_eV", "vbm_eV", "crystal_system", "material_class"]]))

# Removing the H2, since we're looking only at solid materials, and H2 is clearly an outlier

df_nonmetal = df_nonmetal[df_nonmetal["formula"] != "H2"]
print(df_nonmetal.shape)

# Scatter plot of each feature vs the target after removing H2

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df_nonmetal["vbm_eV"], df_nonmetal["band_gap_eV"])
ax.set_xlabel("VBM (eV)")
ax.set_ylabel("Band Gap (eV)")
ax.set_title("VBM vs Band Gap")
plt.savefig("eda_plots/vbm_vs_bandgap_no_H2.png", dpi = 150, bbox_inches="tight")
plt.close()

# Plot of log_band_gap vs vbm to check heteroscedacity this way

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(df_nonmetal["vbm_eV"], df_nonmetal["band_gap_eV"], 
                alpha=0.3, s=5)
axes[0].set_title("Raw Band Gap vs VBM")
axes[0].set_xlabel("VBM (eV)")
axes[0].set_ylabel("Band Gap (eV)")

axes[1].scatter(df_nonmetal["vbm_eV"], df_nonmetal["log_bandgap_eV"], 
                alpha=0.3, s=5)
axes[1].set_title("Log Band Gap vs VBM")
axes[1].set_xlabel("VBM (eV)")
axes[1].set_ylabel("log(Band Gap)")

plt.tight_layout()
plt.savefig("eda_plots/log_bandgap_vs_vbm.png", dpi = 150, bbox_inches="tight")
plt.close()

# Check what percentage of data I would've remove if I remove the materials with <0.1 eV

print((df_nonmetal["band_gap_eV"] < 0.1).sum())
print((df_nonmetal["band_gap_eV"] < 0.1).mean() * 100)

# Check if categorical values serve as predictors
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

sns.boxplot(ax=axes[0], data=df_nonmetal, x="is_direct_gap", y="band_gap_eV")
sns.boxplot(ax=axes[1], data=df_nonmetal, x="crystal_system", y="band_gap_eV")
plt.savefig("eda_plots/boxplot.png", dpi = 150, bbox_inches="tight")
plt.close()

# Check the distribuition of several features to see if it's worth logging when feeding them to the model

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16,8))

features = ["formation_energy_eV_atom", 
                     "density_g_cm3", "n_elements", "n_sites", 
                     "volume_A3", "total_magnetization", 
                     "cbm_eV", "vbm_eV"]

axes = axes.flatten()
for i, feature in enumerate(features):
    df_nonmetal[feature].hist(bins=100, ax=axes[i], linewidth=0.5, edgecolor="white")
    axes[i].set_title(feature)
    axes[i].set_ylabel("Count")
    axes[i].set_xlabel(feature)

plt.savefig("eda_plots/features_histogram.png", dpi = 150, bbox_inches="tight")
plt.close()

# Creating two separate features to tackle magnetization feature problem of beeing zero-inflated

df_nonmetal["is_magnetic"] = (df_nonmetal["total_magnetization"] > 0).astype(bool) # To tell if they are magnetic or not
print(df_nonmetal["is_magnetic"].value_counts()) # number of values in each class

# Check if there are any values below 0 in volume and n_sites, because we want to log those values 

print((df_nonmetal["volume_A3"] <= 0).sum())
print((df_nonmetal["n_sites"] <= 0).sum())

df_nonmetal["log_volume_A3"] = np.log(df_nonmetal["volume_A3"])
df_nonmetal["log_n_sites"] = np.log(df_nonmetal["n_sites"])

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,5))

df_nonmetal["log_volume_A3"].hist(ax=axes[0], bins=100, linewidth=0.5, edgecolor="white")
axes[0].set_xlabel("log_volume_A3")
df_nonmetal["log_n_sites"].hist(ax=axes[1], bins=100, linewidth=0.5, edgecolor="white")
axes[1].set_xlabel("log_n_sites")

plt.savefig("eda_plots/volume_sites_hist.png", dpi = 150, bbox_inches="tight")
plt.close()

# Check the scatter plots of all the features

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16,8))

features = ["log_n_sites", "log_volume_A3", "cbm_eV",
             "vbm_eV", "formation_energy_eV_atom", "density_g_cm3", "total_magnetization",
             "n_elements"]

axes = axes.flatten()

for i, feature in enumerate(features):
    axes[i].scatter(df_nonmetal[feature], df_nonmetal["band_gap_eV"], 
                alpha=0.3, s=5)
    axes[i].set_title(feature)
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel("Energy Bandgap (eV)")

plt.tight_layout()
plt.savefig("eda_plots/features_scatter.png", dpi = 150, bbox_inches="tight")
plt.close()

# Checking 2 minimums of log_n_sites if they are potential outleirs
idx = df_nonmetal["log_n_sites"].idxmin()
print(df_nonmetal.loc[idx, ["formula", "band_gap_eV", "n_sites", "log_n_sites", "crystal_system", "material_class"]])

# Checking potential total_magnetization outliers

print(df_nonmetal.nlargest(5, "total_magnetization")[["formula", "band_gap_eV", "total_magnetization", "material_class", "n_sites"]])

# Feature engeneering of magnetization per atom instead of total_magnetization

df_nonmetal["magnetization_per_atom"] = df_nonmetal["total_magnetization"] / df_nonmetal["n_sites"]
print(df_nonmetal["magnetization_per_atom"].corr(df_nonmetal["band_gap_eV"]))

# FEATURE ENGINEER

# dividing into groups - if the material is magnetic or not
df_magnetic = df_nonmetal[df_nonmetal["is_magnetic"] == True]
print(df_magnetic["total_magnetization"].corr(df_magnetic["n_sites"]))

print(df_nonmetal["total_magnetization"].corr(df_nonmetal["band_gap_eV"]))
print(df_nonmetal["magnetization_per_atom"].corr(df_nonmetal["band_gap_eV"]))
print(df_nonmetal["is_magnetic"].corr(df_nonmetal["band_gap_eV"]))

# BandGap position compared to the space 

df_nonmetal["bandgap_pos"] = df_nonmetal["vbm_eV"] + df_nonmetal["cbm_eV"]

print(df_nonmetal["bandgap_pos"])


# Matminer, extracting 145 different features
df_nonmetal["Composition"] = df_nonmetal["formula"].apply(Composition)

ep = ElementProperty.from_preset("magpie")
ep.set_n_jobs(1)

df_nonmetal = ep.featurize_dataframe(df_nonmetal, 
                                      col_id="Composition",
                                      ignore_errors=True)

print(df_nonmetal.shape)
print(df_nonmetal.head())

magpie_cols = [col for col in df_nonmetal.columns if "MagpieData" in col]
print(f"Magpie added this number of feautres {len(magpie_cols)}")
print(magpie_cols)

magpie_corr = df_nonmetal[magpie_cols + ["band_gap_eV"]].corr()["band_gap_eV"].drop("band_gap_eV")
print(magpie_corr.sort_values(ascending=True).head(10))

spacegroup_cols = [col for col in df_nonmetal.columns if "SpaceGroupNumber" in col]
df_nonmetal = df_nonmetal.drop(columns = spacegroup_cols)

magpie_features = [col for col in df_nonmetal.columns if "MagpieData" in col]

# Creating the Feature list to feed the model

engineered_columns = ["log_n_sites", "log_volume_A3", "is_direct_gap", "density_g_cm3", 
             "is_magnetic", "magnetization_per_atom", "n_elements"]

features_final = engineered_columns + magpie_features

metal = df_nonmetal["material_class"].value_counts()
print(metal)

group_stats = df_nonmetal.groupby("material_class")["band_gap_eV"].agg(["min", "max"])
print(group_stats.loc[["insulator", "semiconductor", "narrow_gap_semiconductor"]])

# X = df_nonmetal[features_final]
# Y = df_nonmetal["band_gap_eV"]

df_nonmetal[features_final + ["band_gap_eV"]].to_csv("model_dataset.csv", index=False)










