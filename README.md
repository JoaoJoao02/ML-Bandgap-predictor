# ML Band Gap Predictor

Machine learning pipeline to predict electronic band gaps of inorganic crystals 
using Materials Project DFT data, featuring physics-motivated DOS feature 
engineering and XGBoost regression.

---

## Motivation

Solid State Physics is one of the most challenging and fascinating subjects in 
Engineering Physics. To deepen my understanding of concepts like Fermi energy, 
band theory, and crystal structure, I built this project around one of the 
field's most important material properties — the electronic band gap.

Beyond the personal motivation, there is a real-world problem this addresses: 
Density Functional Theory (DFT) is currently the most powerful computational 
method for predicting electronic properties of materials, but it is 
computationally expensive — calculating the band gap of a single material can 
take hours to days on a supercomputer. A fast ML model that predicts band gaps 
from compositional and structural descriptors could dramatically accelerate 
materials screening for applications like solar cells, LEDs, and semiconductors.

---

## Dataset

Data was fetched from the [Materials Project](https://materialsproject.org/) 
API, yielding **32,285 inorganic crystal structures** spanning metals, 
semiconductors, and insulators.

**Target variable**: `band_gap_eV` — the energy difference between the 
conduction band minimum (CBM) and valence band maximum (VBM), in electron volts 
(eV). This represents the minimum energy an electron needs to jump from the 
valence band to the conduction band. Materials are classified as:

| Class | Band Gap Range |
|---|---|
| Metal | 0 eV |
| Narrow gap semiconductor | 0 – 1.5 eV |
| Semiconductor | 1.5 – 3.0 eV |
| Insulator | > 3.0 eV |

> **Note**: Band gap values are computed at the PBE (aproximation used in DFT for the quantum mechanical interactions between electrons) level of DFT, which systematically underestimates experimental band gaps by approximately 40%. The model predicts DFT values, not experimental ones.

---

## Methodology

### Data Collection
Material properties and Density of States (DOS) curves were fetched directly 
from the Materials Project API. DOS curves encode how electronic states are 
distributed across energy levels and were used to engineer physics-motivated 
features.

### Feature Engineering
- **DOS features**: Extracted from the valence band region of each material's 
DOS curve — integrated density of states, number of Van Hove singularities, 
VBM edge slope, and mean density near the Fermi level.
- **Magnetization**: Converted total magnetization (extensive) to magnetization 
per atom (intensive) to make it size-independent.
- **Log transforms**: Applied to volume, number of sites, and DOS features to 
correct for heavy right skew.
- **Missing values**: DOS features had ~25% missing values due to API fetch 
failures. Missing values were imputed using the class mean (e.g. insulator NaNs 
were filled with the mean of all insulators) to preserve physical consistency.
- **Leakage prevention**: CBM and VBM energies were excluded as features since 
they are direct outputs of the same DFT calculation that produces the band gap.

### Model
- **Algorithm**: XGBoost Regressor
- **Hyperparameter tuning**: RandomizedSearchCV with 5-fold cross-validation
- **Best parameters**: `subsample=0.6`, `reg_lambda=1.0`, `reg_alpha=1.0`, 
`n_estimators=100`, `max_depth=10`, `learning_rate=0.1`, `colsample_bytree=0.8`
- **Train/test split**: 80/20, stratified by material class to ensure 
representative class distributions in both sets

---

## Results

| Metric | Train | Test |
|---|---|---|
| R² | 0.947 | 0.811 |
| RMSE (eV) | 0.375 | 0.708 |
| MAE (eV) | 0.222 | 0.408 |

The model shows some overfitting, reflected in the gap between train and test 
scores. The test MAE of ~0.41 eV is competitive given that PBE-DFT itself 
carries an intrinsic error of 0.5–1.0 eV relative to experiment.

### Feature Importances

| Feature | Importance |
|---|---|
| dos_peak_count | 0.277 |
| density_g_cm3 | 0.112 |
| dos_mean_density | 0.104 |
| is_direct_gap | 0.087 |
| n_elements | 0.081 |

---

## Limitations

- PBE band gaps are systematically underestimated relative to experiment
- DOS data was unavailable for ~25% of materials, requiring imputation
- No explicit structural features beyond crystal system and spacegroup number
- Overfitting suggests the model would benefit from additional regularization 
or more diverse training data

---

## How to Run

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download the dataset: `python data_download.py`
4. Run feature engineering and EDA: `python withmetal.py`
5. Train the model: `python model.py`

---

## Requirements

- mp-api
- pymatgen
- matminer
- scikit-learn
- xgboost
- pandas
- numpy
- matplotlib
- seaborn
- scipy

---

## References

- Jain et al., *The Materials Project: A materials genome approach*, 
APL Materials, 2013
- Zhuo et al., *Predicting the Band Gaps of Inorganic Solids by Machine 
Learning*, Journal of Physical Chemistry Letters, 2018
