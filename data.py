""" Band Gap Classifier"""

import os
import pandas as pd 
from mp_api.client import MPRester
import numpy as np
from scipy.integrate import simpson
from scipy.signal import find_peaks


MP_API_KEY = os.getenv("MP_API_KEY", "pXPTKGG9vhJ2SdTSWBNK6dYFis5YL1No")

# Band gap filter: only download materials in this range (eV)
# 0 = metals, 0–3 eV = semiconductors, >3 eV = insulators
BAND_GAP_MIN = 0.0   # set to 0.1 to exclude metals
BAND_GAP_MAX = 10.0  # set lower (e.g. 4.0) for semiconductors only
 
# Only include thermodynamically stable materials?
# True  → cleaner dataset, fewer entries (~30k)
# False → larger dataset, includes metastable phases (~150k)
STABLE_ONLY = True
 
# Maximum number of elements per compound (e.g. 4 = up to quaternary)
MAX_ELEMENTS = 4
 
# Output file
OUTPUT_CSV = "band_gap_dataset_dos.csv"
 
# ─────────────────────────────────────────────
# FIELDS TO DOWNLOAD
# ─────────────────────────────────────────────
# These are the properties we'll fetch from the API.
# Adding more fields slows down the download.
# See all available fields at: https://docs.materialsproject.org
 
FIELDS = [
    "material_id",          # Unique MP identifier (e.g. "mp-149" = Silicon)
    "formula_pretty",       # Human-readable formula (e.g. "Si", "GaAs")
    "band_gap",             # DFT band gap in eV (PBE functional, ~40% underestimated)
    "is_gap_direct",        # True if direct gap (important for optical applications!)
    "energy_above_hull",    # Thermodynamic stability (0 = on convex hull = stable)
    "formation_energy_per_atom",  # Formation energy in eV/atom
    "density",              # g/cm³
    "nelements",            # Number of distinct elements
    "elements",             # List of element symbols
    "nsites",               # Number of atoms in unit cell
    "volume",               # Unit cell volume in Å³
    "symmetry",             # Contains crystal_system, spacegroup symbol & number
    "total_magnetization",  # μB per unit cell (0 = non-magnetic)
    "cbm",                  # Conduction band minimum in eV
    "vbm",                  # Valence band maximum in eV
]

# ─────────────────────────────────────────────
# FEATURE ENGINEERING FUNCTION
# ─────────────────────────────────────────────
def engineer_dos_features(dos_obj):
    """Extracts physical descriptors from the DOS curve."""
    try:
        # Shift energies relative to Fermi
        energies = dos_obj.energies - dos_obj.efermi
        densities = dos_obj.get_densities()
        
        # Focus on the Valence Band (VB) near the gap (-4eV to 0eV)
        vb_mask = (energies >= -4) & (energies <= 0)
        vb_e, vb_d = energies[vb_mask], densities[vb_mask]
        
        # 1. Integrated States (Total electron density near Fermi)
        dos_int = simpson(y=vb_d, x=vb_e) if len(vb_e) > 1 else 0
        
        # 2. Peak Count (Van Hove Singularities)
        peaks, _ = find_peaks(vb_d, height=np.mean(vb_d))
        
        # 3. VBM Slope (Steepness of the band edge)
        # We take the last 5 data points before the Fermi level
        vbm_slope = np.polyfit(vb_e[-5:], vb_d[-5:], 1)[0] if len(vb_e) > 5 else 0
        
        return {
            "dos_integral_vb": dos_int,
            "dos_peak_count": len(peaks),
            "dos_vbm_slope": vbm_slope,
            "dos_mean_density": np.mean(vb_d) if len(vb_d) > 0 else 0
        }
    except Exception as e:
        print(f"The DOS of this material failed: {e}")
        return {"dos_integral_vb": 0, "dos_peak_count": 0, "dos_vbm_slope": 0, "dos_mean_density": 0}

 
# ─────────────────────────────────────────────
# DOWNLOAD
# ─────────────────────────────────────────────
 
def download_dataset():
    print("=" * 55)
    print("  Materials Project Band Gap Downloader")
    print("=" * 55)
    print(f"  Band gap range : {BAND_GAP_MIN} – {BAND_GAP_MAX} eV")
    print(f"  Stable only    : {STABLE_ONLY}")
    print(f"  Max elements   : {MAX_ELEMENTS}")
    print("=" * 55)
 
    with MPRester(MP_API_KEY) as mpr:
        print("\n⏳ Querying Materials Project API...")
 
        docs = mpr.materials.summary.search(
            band_gap=(BAND_GAP_MIN, BAND_GAP_MAX),
            is_stable=STABLE_ONLY,
            num_elements=(1, MAX_ELEMENTS),
            fields=FIELDS,
        )
 
        print(f"✅ Retrieved {len(docs)} materials.\n")
 
    # ─────────────────────────────────────────────
    # CONVERT TO DATAFRAME
    # ─────────────────────────────────────────────
    total_coutns = len(docs)
    records = []
    for i, doc in enumerate(docs):
        record = {
            "material_id":               doc.material_id,
            "formula":                   doc.formula_pretty,
            "band_gap_eV":               doc.band_gap,
            "is_direct_gap":             doc.is_gap_direct,
            "energy_above_hull_eV":      doc.energy_above_hull,
            "formation_energy_eV_atom":  doc.formation_energy_per_atom,
            "density_g_cm3":             doc.density,
            "n_elements":                doc.nelements,
            "elements":                  ", ".join(str(e) for e in doc.elements) if doc.elements else None,
            "n_sites":                   doc.nsites,
            "volume_A3":                 doc.volume,
            "crystal_system":            doc.symmetry.crystal_system.value if doc.symmetry else None,
            "spacegroup_number":         doc.symmetry.number if doc.symmetry else None,
            "total_magnetization":       doc.total_magnetization,
            "cbm_eV":                    doc.cbm,
            "vbm_eV":                    doc.vbm,
        }
        # Fetch and Engineer DOS
        print(f"[{i+1}/{total_coutns}] Fetching DOS for {doc.material_id}...", end="\r")
        try:
            dos_obj = mpr.get_dos_by_material_id(doc.material_id)
            if dos_obj:
                dos_feats = engineer_dos_features(dos_obj)
                record.update(dos_feats)
        except Exception:
            pass # Features default to 0/None via error handling in function

        if (i + 1) % 100 == 0:
            temp_df = pd.DataFrame(records)
            temp_df.to_csv(OUTPUT_CSV, index=False)


        records.append(record)
 
    df = pd.DataFrame(records)
 
    # ─────────────────────────────────────────────
    # BASIC CLEANING
    # ─────────────────────────────────────────────
 
    # Drop rows where the band gap itself is missing
    before = len(df)
    df = df.dropna(subset=["band_gap_eV"])
    print(f"🧹 Dropped {before - len(df)} rows with missing band gap.")
 
    # Classify into material type based on band gap
    # This is a useful target for classification tasks
    def classify(bg):
        if bg == 0.0:
            return "metal"
        elif bg < 1.5:
            return "narrow_gap_semiconductor"
        elif bg < 3.0:
            return "semiconductor"
        else:
            return "insulator"
 
    df["material_class"] = df["band_gap_eV"].apply(classify)
 
    # ─────────────────────────────────────────────
    # SAVE
    # ─────────────────────────────────────────────
 
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n💾 Saved {len(df)} materials to '{OUTPUT_CSV}'")
 
    # ─────────────────────────────────────────────
    # SUMMARY STATISTICS
    # ─────────────────────────────────────────────
 
    print("\n─── Dataset Summary ──────────────────────────────")
    print(f"  Total materials  : {len(df)}")
    print(f"  Band gap range   : {df['band_gap_eV'].min():.3f} – {df['band_gap_eV'].max():.3f} eV")
    print(f"  Mean band gap    : {df['band_gap_eV'].mean():.3f} eV")
    print(f"  Direct-gap count : {df['is_direct_gap'].sum()} ({100*df['is_direct_gap'].mean():.1f}%)")
    print(f"\n  Material classes:")
    print(df["material_class"].value_counts().to_string())
    print(f"\n  Crystal systems:")
    print(df["crystal_system"].value_counts().to_string())
    print("─────────────────────────────────────────────────\n")
 
    return df
 
 
if __name__ == "__main__":
    df = download_dataset()
 
    # Quick sanity check — print a few famous semiconductors
    known = ["Si", "GaAs", "GaN", "ZnO", "CdTe"]
    print("─── Sanity check: known semiconductors ───────────")
    subset = df[df["formula"].isin(known)][["formula", "band_gap_eV", "is_direct_gap", "crystal_system", "dos_integral_vb"]]
    print(subset.to_string(index=False))
    print("──────────────────────────────────────────────────")