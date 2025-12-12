import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cobra.io import read_sbml_model

# ============================================================
# Day 27 – Realistic TME nutrient-bound E-Flux modeling
#
# Goal:
#   Replace the simplistic TME constraints from Day 22 with
#   literature-inspired nutrient bounds that better mimic
#   tumor microenvironment (TME) conditions.
#
#   1) Define three media:
#        - Rich/Blood      : nutrient-replete
#        - TME_edge        : moderately depleted tumor edge
#        - TME_core        : strongly depleted tumor core
#
#   2) Combine these media with reaction-level expression
#      (Day 17 output) using E-Flux.
#
#   3) Run FBA for each sample (Blood, TumorEdge, TumorCore),
#      save flux distributions and biomass summary.
#
# Inputs:
#   - Human-GEM model (SBML)
#   - day17_reaction_expression_Tcells.csv
#
# Outputs:
#   - day27_flux_realTME_{sample}.csv
#   - day27_flux_hist_realTME_{sample}.png
#   - day27_realTME_biomass_summary_Tcells.csv
#   - day27_realTME_medium_bounds.csv
#
# ============================================================

# Paths
BASE = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = os.path.join(BASE, "../database/Human-GEM.xml")
REACTION_EXPR_FILE = os.path.join(BASE, "day17_reaction_expression_Tcells.csv")

OUT_DIR = BASE
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Loading model: {MODEL_FILE}")
model = read_sbml_model(MODEL_FILE)
print(f"Model loaded: {len(model.genes)} genes, {len(model.reactions)} reactions")

print(f"Loading reaction-level expression: {REACTION_EXPR_FILE}")
rxn_expr = pd.read_csv(REACTION_EXPR_FILE, index_col=0)
samples = list(rxn_expr.columns)
print("Samples found:", samples)


# ============================================================
# Define key exchange reaction IDs (from your listing)
# ============================================================

# These IDs come from your printed exchange list for Human-GEM.
# If any of them do not exist in the model, the script will warn
# and skip that metabolite.

EX_GLC = "MAR09034"   # Exchange of glucose
EX_O2  = "MAR09048"   # Exchange of O2
EX_GLN = "MAR09063"   # Exchange of glutamine
EX_LAC_L = "MAR09135" # Exchange of L-lactate
EX_LAC_D = "MAR09136" # Exchange of D-lactate

# Optional extra nutrients – adjust IDs if you know them in your model.
# If you don't know them yet, you can leave them as None or comment out.
EX_SER = None   # serine
EX_ARG = None   # arginine
EX_TRP = None   # tryptophan


# ============================================================
# Define media: literature-inspired nutrient bounds
# ============================================================

# Convention:
#   - Lower bound < 0  → uptake allowed (mmol/gDW/h)
#   - Lower bound = 0  → no uptake
#
# Values below are approximate and meant to capture RELATIVE
# differences between rich and TME, not exact physiological numbers.

# ---- Rich / Blood (nutrient-replete) -----------------------
medium_rich = {
    EX_GLC:  -10.0,  # glucose ~ -8 to -15
    EX_GLN:  -10.0,  # glutamine abundant
    EX_O2:   -20.0,  # oxygen abundant
    EX_LAC_L:  0.0,  # no lactate uptake, only secretion
    EX_LAC_D:  0.0,
    # Optional extras if IDs are known:
    EX_SER:  -5.0 if EX_SER else None,
    EX_ARG:  -5.0 if EX_ARG else None,
    EX_TRP:  -2.0 if EX_TRP else None,
}

# ---- TME edge: moderate depletion, hypoxia, some lactate ----
medium_TME_edge = {
    EX_GLC:  -3.0,   # ~30% of rich
    EX_GLN:  -3.0,   # moderate glutamine depletion
    EX_O2:   -8.0,   # hypoxia
    EX_LAC_L: -5.0,  # allow significant lactate uptake (high extracellular lactate)
    EX_LAC_D: -5.0,
    EX_SER:  -1.5 if EX_SER else None,
    EX_ARG:  -1.5 if EX_ARG else None,
    EX_TRP:  -0.6 if EX_TRP else None,
}

# ---- TME core: severe depletion, strong hypoxia, high lactate ----
medium_TME_core = {
    EX_GLC:  -1.0,   # ~10% of rich
    EX_GLN:  -1.0,
    EX_O2:   -2.0,   # very low O2
    EX_LAC_L: -10.0, # very high lactate availability
    EX_LAC_D: -10.0,
    EX_SER:  -0.5 if EX_SER else None,
    EX_ARG:  -0.5 if EX_ARG else None,
    EX_TRP:  -0.2 if EX_TRP else None,
}


def clean_medium_dict(mdict):
    """
    Remove entries where the reaction ID is None.
    This allows us to keep optional nutrients without breaking the code.
    """
    return {k: v for k, v in mdict.items() if k is not None}


medium_rich = clean_medium_dict(medium_rich)
medium_TME_edge = clean_medium_dict(medium_TME_edge)
medium_TME_core = clean_medium_dict(medium_TME_core)

# For writing a summary CSV later
medium_records = []


def apply_medium_bounds(m, medium_dict, scenario_name):
    """
    Apply nutrient bounds to a model for a given scenario.

    Parameters
    ----------
    m : cobra.Model
        Model whose bounds will be modified in-place.
    medium_dict : dict
        Mapping {exchange_rxn_id: lower_bound}.
    scenario_name : str
        Name of the scenario (for logging and summary table).
    """
    def safe_set_lb(rxn_id, lb):
        if rxn_id in m.reactions:
            rxn = m.reactions.get_by_id(rxn_id)
            rxn.lower_bound = lb
            medium_records.append({
                "scenario": scenario_name,
                "reaction_id": rxn_id,
                "reaction_name": rxn.name,
                "new_lower_bound": lb,
            })
        else:
            print(f"[Warning] Exchange '{rxn_id}' not found in model; skipped.")

    print(f"Applying medium for scenario '{scenario_name}' ...")
    for rxn_id, lb in medium_dict.items():
        safe_set_lb(rxn_id, lb)


# ============================================================
# E-Flux helper (same logic as Day 18 / Day 22)
# ============================================================

def apply_eflux(model, expr_norm):
    """
    Apply E-Flux scaling to reaction bounds using normalized
    reaction-level expression.

    Parameters
    ----------
    model : cobra.Model
        Original metabolic model.
    expr_norm : pandas.Series
        Normalized expression (0–1) indexed by reaction ID.

    Returns
    -------
    cobra.Model
        A copy of the model with scaled bounds.
    """
    m = model.copy()

    for rxn in m.reactions:
        # Do not scale biomass reactions
        if "biomass" in rxn.id.lower():
            continue

        if rxn.id in expr_norm.index:
            scale = float(expr_norm[rxn.id])
        else:
            scale = 1e-6

        scale = max(scale, 1e-6)

        if rxn.upper_bound > 0:
            rxn.upper_bound *= scale
        if rxn.lower_bound < 0:
            rxn.lower_bound *= scale

    return m


# ============================================================
# Map sample → medium scenario
# ============================================================

def infer_scenario_from_sample(sample_name):
    """
    Map sample name to a medium scenario.

    Blood      → 'Rich'
    TumorEdge  → 'TME_edge'
    TumorCore  → 'TME_core'
    """
    s = sample_name.lower()
    if "blood" in s:
        return "Rich"
    elif "edge" in s:
        return "TME_edge"
    elif "core" in s:
        return "TME_core"
    else:
        print(f"[Warning] Could not infer scenario for '{sample_name}', using 'Rich'.")
        return "Rich"


# ============================================================
# Main loop: realistic TME E-Flux for each sample
# ============================================================

biomass_records = []

for sample in samples:
    print("\n==============================================")
    print(f"Day 27 – Realistic TME E-Flux for sample: {sample}")
    print("==============================================")

    expr_vec = rxn_expr[sample]
    max_val = expr_vec.max()
    if max_val <= 0:
        print(f"[Warning] Sample {sample} has non-positive max expression; skipping.")
        continue

    expr_norm = expr_vec / max_val
    print(f"Expression stats for {sample}: mean={expr_vec.mean():.3f}, std={expr_vec.std():.3f}")

    scenario = infer_scenario_from_sample(sample)
    print(f"Scenario for {sample}: {scenario}")

    # Start from a clean copy of the base model
    model_env = model.copy()

    # Apply medium bounds according to scenario
    if scenario == "Rich":
        apply_medium_bounds(model_env, medium_rich, scenario_name=scenario)
    elif scenario == "TME_edge":
        apply_medium_bounds(model_env, medium_TME_edge, scenario_name=scenario)
    elif scenario == "TME_core":
        apply_medium_bounds(model_env, medium_TME_core, scenario_name=scenario)
    else:
        # Fallback to Rich
        apply_medium_bounds(model_env, medium_rich, scenario_name=scenario)

    # Apply E-Flux scaling
    print("Applying E-Flux scaling ...")
    model_scaled = apply_eflux(model_env, expr_norm)

    # Optimize
    print("Running FBA under realistic TME constraints ...")
    sol = model_scaled.optimize()

    status = sol.status
    biomass = sol.objective_value if sol.status == "optimal" else 0.0

    print(f"FBA status: {status}")
    print(f"Biomass (objective value): {biomass}")

    biomass_records.append({
        "sample": sample,
        "scenario": scenario,
        "status": status,
        "biomass": biomass,
    })

    # Save flux vector
    flux_series = sol.fluxes
    flux_path = os.path.join(OUT_DIR, f"day27_flux_realTME_{sample}.csv")
    flux_series.to_csv(flux_path)
    print(f"Saved realistic-TME flux distribution: {flux_path}")

    # Plot histogram
    plt.figure(figsize=(6, 4))
    plt.hist(flux_series, bins=80)
    plt.xlabel("Flux value")
    plt.ylabel("Count")
    plt.title(f"Realistic TME E-Flux distribution — {sample} ({scenario})")
    plt.tight_layout()

    fig_out = os.path.join(OUT_DIR, f"day27_flux_hist_realTME_{sample}.png")
    plt.savefig(fig_out, dpi=150)
    plt.close()
    print(f"Saved flux histogram: {fig_out}")


# ============================================================
# Save biomass summary + medium bounds summary
# ============================================================

if biomass_records:
    df_biomass = pd.DataFrame(biomass_records)
    biomass_out = os.path.join(OUT_DIR, "day27_realTME_biomass_summary_Tcells.csv")
    df_biomass.to_csv(biomass_out, index=False)
    print(f"\nSaved biomass summary: {biomass_out}")

if medium_records:
    df_medium = pd.DataFrame(medium_records)
    medium_out = os.path.join(OUT_DIR, "day27_realTME_medium_bounds.csv")
    df_medium.to_csv(medium_out, index=False)
    print(f"Saved medium bounds summary: {medium_out}")

print("\nDay 27 realistic TME E-Flux analysis completed.")

