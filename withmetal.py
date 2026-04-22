"""Data with metal and non metal"""

import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import seaborn as sns
import os
from matminer.featurizers.composition import ElementProperty
from pymatgen.core.composition import Composition

#Display options
#pd.set_option("display.max_columns", None)

#I'll make another directory 
os.makedirs("eda_plots_with_metal", exist_ok=True)

#import the dataset
df = pd.read_csv("band_gap_dataset_dos.csv")


print(df.head())
print(df.shape)
print(df.dtypes)


print([df["crystal_system"].unique()])
print([df["material_class"].unique()])
print([df["spacegroup_number"].unique()])
print([df["energy_above_hull_eV"].unique()])
print([df["dos_integral_vb"].unique()])
print([df["dos_peak_count"].unique()])
print([df["dos_vbm_slope"].unique()])
print([df["dos_mean_density"].unique()])


print("--" * 50)
print(df["dos_peak_count"].isna().value_counts())
print("--" * 50)
print(df["dos_integral_vb"].isna().value_counts())
print("--" * 50)
print(df["dos_vbm_slope"].isna().value_counts())
print("--" * 50)
print(df["dos_mean_density"].isna().value_counts())

print("--/" * 50)
print(df.groupby("material_class")["dos_integral_vb"].mean())

df["dos_integral_vb"] = df["dos_integral_vb"].fillna(df.groupby("material_class")["dos_integral_vb"].transform("mean"))
df["dos_peak_count"] = df["dos_peak_count"].fillna(df.groupby("material_class")["dos_peak_count"].transform("mean"))
df["dos_vbm_slope"] = df["dos_vbm_slope"].fillna(df.groupby("material_class")["dos_vbm_slope"].transform("mean"))
df["dos_mean_density"] = df["dos_mean_density"].fillna(df.groupby("material_class")["dos_mean_density"].transform("mean"))



## Transforming the strings into floats to feed the model
mapping = {
    'Hexagonal': 1.0,
    'Cubic': 2.0,
    'Trigonal': 3.0,
    'Tetragonal': 4.0,
    'Monoclinic': 5.0,
    'Orthorhombic': 6.0,
    'Triclinic': 7.0
}

mapping_material = {
    'metal': 1.0,
    'insulator': 2.0,
    'semiconductor': 3.0,
    'narrow_gap_semiconductor': 4.0
}

df["crystal_system"] = df["crystal_system"].map(mapping)
df["material_class"] = df["material_class"].map(mapping_material)

# Debug print to check if the values are all right
print(df["crystal_system"].value_counts())
print(df["material_class"].value_counts())

# Divide into metal/non_metal in case I need it
metal = df[df["band_gap_eV"] <= 0.0].copy()
non_metal = df[df["band_gap_eV"] > 0.0].copy()


## Feature Engineer

# Remove columns that are useless
df=df.drop(["material_id", "formula", "elements"], axis=1)
metal=metal.drop(["material_id", "formula", "elements"], axis=1)
non_metal=non_metal.drop(["material_id", "formula", "elements"], axis=1)

# Some different/new features
df["mag_per_atom"] = df["total_magnetization"] / df["n_sites"]
df["volume_A3"] = np.log(df["volume_A3"])
df["n_sites"] = np.log(df["n_sites"])

df["dos_integral_vb"] = np.log1p(df["dos_integral_vb"])
df["dos_peak_count"] = np.log1p(df["dos_peak_count"])
df["dos_vbm_slope"] = np.log1p(df["dos_vbm_slope"])
df["dos_mean_density"] = np.log1p(df["dos_mean_density"])


## Plots for visualization

# Histogram of bandgaps
df["band_gap_eV"].hist(bins=80)
plt.savefig("eda_plots_with_metal/df_bandgap_hist")
plt.close()
metal["band_gap_eV"].hist(bins=80)
plt.savefig("eda_plots_with_metal/metal_bandgap_hist")
plt.close()
non_metal["band_gap_eV"].hist(bins=80)
plt.savefig("eda_plots_with_metal/nmetal_bandgap_hist")
plt.close()


# Correlation Matrices
corr_matrix = df.corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.tight_layout()
plt.savefig("eda_plots_with_metal/df_corr_matrix")
plt.close()
corr_matrix_metal = metal.corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix_metal, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.tight_layout()
plt.savefig("eda_plots_with_metal/metal_corr_matrix")
plt.close()
corr_matrix_non_metal = non_metal.corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix_non_metal, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.tight_layout()
plt.savefig("eda_plots_with_metal/nmetal_corr_matrix")
plt.close()

# Histogram of each Feature
df_numeric = df.select_dtypes(include=['number']) # only use the numeric features
n_features = len(df_numeric.columns)
n_cols = 4
n_rows = (n_features + n_cols - 1) // n_cols
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(16,4*n_rows))

axes = axes.flatten()

for i, feature in enumerate(df_numeric.columns):
    df_numeric[feature].hist(bins=100, ax=axes[i], linewidth=0.5, edgecolor="white")
    axes[i].set_title(feature)
    axes[i].set_ylabel("Count")
    axes[i].set_xlabel(feature)

for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

plt.savefig("eda_plots_with_metal/features_hist.png", dpi = 150, bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(16,4*n_rows))

axes = axes.flatten()

for i, feature in enumerate(df_numeric.columns):
    axes[i].scatter(df[feature], df["band_gap_eV"], alpha=0.3, s=5)
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel("Bandgap (eV)")

for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

plt.savefig("eda_plots_with_metal/features_scatter.png", dpi = 150, bbox_inches="tight")
plt.close()

# Removing outliers out of 1-99%

numerical_cols = df.select_dtypes(include=[np.number]).columns
outlier_indices = set()  # Track unique rows with outliers

for col in numerical_cols:
    threshold_up= df[col].quantile(0.999)
    threshold_down = df[col].quantile(0.001)
    outliers_down = df[df[col] < threshold_down]
    outliers_up = df[df[col] > threshold_up]
    
    if len(outliers_down) > 0 or len(outliers_up) > 0:
        print(f"\n{col}: {len(outliers_down) + len(outliers_up)} outliers")
        print(outliers_down[[col]])
        print(outliers_up[[col]])
        
        # Add indices to set (automatically handles duplicates)
        outlier_indices.update(outliers_down.index)
        outlier_indices.update(outliers_up.index)

print(f"\n{'='*50}")
print(f"Total unique rows with outliers across all features: {len(outlier_indices)}")
print(f"Percentage of outlier rows: {len(outlier_indices)/len(df)*100:.2f}%")

# Remove outlier rows from all dataframes
df = df.drop(outlier_indices, errors='ignore')



print(f"\nAfter removing outliers:")
print(f"df shape: {df.shape}")
print(f"df head:\n{df.head()}")
print(df.value_counts())


# Save with absolute path to ensure it saves
output_path = os.path.join(os.getcwd(), "df_model_dataset.csv")
print(f"\nSaving to: {output_path}")
df.to_csv(output_path, index=False)
print(f"File saved successfully! File size: {os.path.getsize(output_path)} bytes")


