import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cobra.io import read_sbml_model

# ============================================================
# Configuration
# ============================================================

# Directory where this script lives
BASE = os.path.dirname(os.path.abspath(__file__))

# Human-GEM model (same as earlier days)
MODEL_FILE = os.path.join(BASE, "../database/Human-GEM.xml")

# Day 21 (rich) essentiality results:
#   day21_essentiality_Blood.csv
#   day21_essentiality_TumorEdge.csv
#   day21_essentiality_TumorCore.csv
RICH_KO_TEMPLATE = os.path.join(BASE, "day21_essentiality_{sample}.csv")

# Day 24 (TME) essentiality results:
#   day24_KO_TME_Blood.csv
#   day24_KO_TME_TumorEdge.csv
#   day24_KO_TME_TumorCore.csv
TME_KO_TEMPLATE = os.path.join(BASE, "day24_KO_TME_{sample}.csv")

OUT_DIR = BASE
os.makedirs(OUT_DIR, exist_ok=True)

# Sample names you used everywhere
SAMPLES = ["Blood", "TumorEdge", "TumorCore"]


# ============================================================
# Helper: load and normalize KO tables
# ============================================================

def load_KO_table(sample: str, env: str) -> pd.DataFrame | None:
    """
    Load knockout (KO) result table for one sample and environment
    ("rich" for Day21, "TME" for Day24), and normalize column names.

    After this function, the returned DataFrame always has:
        reaction_id
        reaction_name
        biomass_KO_{env}
        baseline_biomass_{env}
        baseline_flux_{env}
        is_essential_{env}

    Parameters
    ----------
    sample : str
        "Blood", "TumorEdge", or "TumorCore"
    env : {"rich", "TME"}
        Which environment's KO table to load.

    Returns
    -------
    pandas.DataFrame or None
        Normalized KO table, or None if file does not exist.
    """
    # Choose path based on environment
    if env == "rich":
        path = RICH_KO_TEMPLATE.format(sample=sample)
    else:  # "TME"
        path = TME_KO_TEMPLATE.format(sample=sample)

    # If file missing, skip this sample
    if not os.path.exists(path):
        print(f"[Warning] KO file not found for {sample}, {env}: {path}")
        return None

    print(f"Loading {env} KO table for {sample}: {path}")
    df = pd.read_csv(path)

    # --- Normalize ID/name columns depending on source format -----------------
    # Day21 (rich) file uses 'rxn_id' and 'name'
    # Day24 (TME) file already uses 'reaction_id' and 'reaction_name'

    if "rxn_id" in df.columns:
        df = df.rename(columns={"rxn_id": "reaction_id"})
    if "name" in df.columns and "reaction_name" not in df.columns:
        df = df.rename(columns={"name": "reaction_name"})

    # Day21 rich file uses 'ko_biomass' for KO biomass
    # Day24 TME file uses 'biomass_KO'
    if "ko_biomass" in df.columns and "biomass_KO" not in df.columns:
        df = df.rename(columns={"ko_biomass": "biomass_KO"})

    # Basic sanity checks
    required_cols = ["reaction_id", "biomass_KO", "baseline_biomass", "baseline_flux"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"{path} missing required column '{col}'")

    # --- Essentiality flag ----------------------------------------------------
    # We want a boolean column:
    #   is_essential_rich   (for env == "rich")
    #   is_essential_TME    (for env == "TME")
    #
    # Day21 rich file already has 'essential' (True/False)
    # Day24 TME file has 'is_essential_TME' (from your Day24 script)
    # If not present, we recompute using biomass_KO < 1% of baseline.

    if env == "rich":
        if "is_essential_rich" in df.columns:
            # Already has proper name
            pass
        elif "essential" in df.columns:
            # Map Day21 'essential' to standard name
            df["is_essential_rich"] = df["essential"].astype(bool)
        else:
            # Fallback: recompute from biomass
            thr = 0.01 * df["baseline_biomass"]
            df["is_essential_rich"] = df["biomass_KO"] < thr
    else:  # env == "TME"
        if "is_essential_TME" in df.columns:
            # Already correct
            pass
        elif "essential" in df.columns:
            # If you ever stored TME essentials in 'essential'
            df["is_essential_TME"] = df["essential"].astype(bool)
        else:
            thr = 0.01 * df["baseline_biomass"]
            df["is_essential_TME"] = df["biomass_KO"] < thr

    # --- Rename numeric columns with environment suffix -----------------------
    if env == "rich":
        df = df.rename(columns={
            "biomass_KO": "biomass_KO_rich",
            "baseline_biomass": "baseline_biomass_rich",
            "baseline_flux": "baseline_flux_rich"
        })
    else:  # TME
        df = df.rename(columns={
            "biomass_KO": "biomass_KO_TME",
            "baseline_biomass": "baseline_biomass_TME",
            "baseline_flux": "baseline_flux_TME"
        })

    return df


# ============================================================
# Helper: annotate subsystems from Human-GEM
# ============================================================

def add_subsystem_annotation(model, df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'subsystem' column based on reaction IDs in the Human-GEM model.
    """
    rxn2sub = {
        rxn.id: (rxn.subsystem if rxn.subsystem else "Unknown")
        for rxn in model.reactions
    }
    df["subsystem"] = df["reaction_id"].map(lambda x: rxn2sub.get(x, "Unknown"))
    return df


# ============================================================
# Helper: classify essentiality class rich vs TME
# ============================================================

def classify_essentiality(row) -> str:
    """
    Assign a reaction to one of four classes:

        - "essential_in_both"
        - "essential_only_in_rich"
        - "essential_only_in_TME"
        - "non_essential"
    """
    r = bool(row["is_essential_rich"])
    t = bool(row["is_essential_TME"])

    if r and t:
        return "essential_in_both"
    elif r and not t:
        return "essential_only_in_rich"
    elif (not r) and t:
        return "essential_only_in_TME"
    else:
        return "non_essential"


# ============================================================
# Load Human-GEM model for subsystem annotations
# ============================================================

print(f"Loading model: {MODEL_FILE}")
model = read_sbml_model(MODEL_FILE)
print(f"Model loaded: {len(model.genes)} genes, {len(model.reactions)} reactions")


# ============================================================
# Main loop – per sample comparison rich vs TME
# ============================================================

for sample in SAMPLES:
    print("\n" + "=" * 70)
    print(f"Day 25 – Comparing rich vs TME essentiality for sample: {sample}")
    print("=" * 70)

    # Load KO tables for rich and TME
    df_rich = load_KO_table(sample, "rich")
    df_tme  = load_KO_table(sample, "TME")

    # Skip if either environment is missing
    if df_rich is None or df_tme is None:
        print(f"[Warning] Missing KO tables for {sample}; skipped.")
        continue

    # Merge on reaction_id + reaction_name so we have both environments together
    merged = df_rich.merge(
        df_tme,
        on=["reaction_id", "reaction_name"],
        how="inner"
    )

    # Compute essentiality class per reaction
    merged["essentiality_class"] = merged.apply(classify_essentiality, axis=1)

    # Add subsystem information from the model
    merged = add_subsystem_annotation(model, merged)

    # Save full comparison table
    out_csv = os.path.join(OUT_DIR, f"day25_essentiality_compare_{sample}.csv")
    merged.to_csv(out_csv, index=False)
    print(f"Saved essentiality comparison table: {out_csv}")

    # Extract TME-only essentials (the biologically interesting ones)
    tme_only = merged[merged["essentiality_class"] == "essential_only_in_TME"].copy()
    print(f"Number of TME-only essential reactions for {sample}: {len(tme_only)}")

    # Summarize TME-only essential reactions by subsystem (counts)
    sub_summary = (
        tme_only.groupby("subsystem")["reaction_id"]
        .nunique()
        .sort_values(ascending=False)
    )

    # Save subsystem summary (for plots / paper)
    sub_csv = os.path.join(OUT_DIR, f"day25_TME_only_subsystems_{sample}.csv")
    sub_summary.to_csv(sub_csv)
    print(f"Saved TME-only subsystem summary: {sub_csv}")

print("\nDay 25 rich vs TME essentiality comparison completed.")
