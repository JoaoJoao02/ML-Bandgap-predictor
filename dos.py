import os
import pandas as pd
import numpy as np
from mp_api.client import MPRester
from scipy.signal import find_peaks
from scipy.integrate import simpson

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
MP_API_KEY = os.getenv("MP_API_KEY", "pXPTKGG9vhJ2SdTSWBNK6dYFis5YL1No")
OUTPUT_CSV = "band_gap_dos_engineered.csv"
LIMIT = 50  # Set a limit for testing; remove [:LIMIT] later for full run

# Filters
BAND_GAP_RANGE = (0.0, 10.0)
STABLE_ONLY = True
MAX_ELEMENTS = 4

FIELDS = [
    "material_id", "formula_pretty", "band_gap", "is_gap_direct", 
    "energy_above_hull", "formation_energy_per_atom", "density", 
    "nelements", "elements", "nsites", "volume", "symmetry", 
    "total_magnetization", "cbm", "vbm"
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
    except Exception:
        return {"dos_integral_vb": 0, "dos_peak_count": 0, "dos_vbm_slope": 0, "dos_mean_density": 0}

# ─────────────────────────────────────────────
# MAIN DOWNLOADER
# ─────────────────────────────────────────────
def download_and_featurize():
    print("⏳ Querying Summary Data...")
    with MPRester(MP_API_KEY) as mpr:
        docs = mpr.materials.summary.search(
            band_gap=BAND_GAP_RANGE,
            is_stable=STABLE_ONLY,
            num_elements=(1, MAX_ELEMENTS),
            fields=FIELDS
        )
        
        print(f"✅ Found {len(docs)} materials. Processing top {LIMIT}...")
        
        records = []
        for i, doc in enumerate(docs[:LIMIT]):
            # Base record
            record = {
                "material_id": doc.material_id,
                "formula": doc.formula_pretty,
                "band_gap_eV": doc.band_gap,
                "is_direct_gap": doc.is_gap_direct,
                "density_g_cm3": doc.density,
                "crystal_system": doc.symmetry.crystal_system.value if doc.symmetry else None,
                "spacegroup": doc.symmetry.number if doc.symmetry else None,
            }

            # Fetch and Engineer DOS
            print(f"[{i+1}/{LIMIT}] Fetching DOS for {doc.material_id}...", end="\r")
            try:
                dos_obj = mpr.get_dos_by_material_id(doc.material_id)
                if dos_obj:
                    dos_feats = engineer_dos_features(dos_obj)
                    record.update(dos_feats)
            except Exception:
                pass # Features default to 0/None via error handling in function

            records.append(record)

    # Save to CSV
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n💾 Saved to {OUTPUT_CSV}")
    return df

if __name__ == "__main__":
    df_final = download_and_featurize()
    print(df_final.head())
