import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cobra.io import read_sbml_model

# ------------------------------------------------------------
# Day 22 – E-Flux under tumor microenvironment (TME) constraints
#
# Goal:
#   Combine expression-based E-Flux with realistic nutrient
#   constraints for Blood / TumorEdge / TumorCore to simulate
#   T-cell metabolism in different environments.
#
#   - Use reaction-level expression (Day17 output).
#   - Apply E-Flux scaling (like Day18).
#   - Add environment-specific exchange bounds:
#       * Blood: rich / blood-like
#       * Edge:  moderate limitation
#       * Core:  severe TME (low glucose, low oxygen, low glutamine)
#
#   Output:
#     - Flux distributions for each sample under its TME scenario
#     - Biomass summary table
# ------------------------------------------------------------

# Paths
BASE = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = os.path.join(BASE, "../database/Human-GEM.xml")
REACTION_EXPR_FILE = os.path.join(BASE, "day17_reaction_expression_Tcells.csv")

OUT_DIR = BASE
os.makedirs(OUT_DIR, exist_ok=True)

# 1) Load model and reaction-level expression

print(f"Loading model: {MODEL_FILE}")
model = read_sbml_model(MODEL_FILE)
print(f"Model loaded: {len(model.genes)} genes, {len(model.reactions)} reactions")

print(f"Loading reaction-level expression: {REACTION_EXPR_FILE}")
rxn_expr = pd.read_csv(REACTION_EXPR_FILE, index_col=0)
samples = list(rxn_expr.columns)
print("Samples found:", samples)



# 2) Environment helper: define TME-like nutrient constraints
# IMPORTANT:
#   You MUST adjust these exchange IDs to match your Human-GEM model.
#   A common check is:
#       for rxn in model.exchanges:
#           print(rxn.id, rxn.name)
#
# Below are example IDs; replace if needed.

GLC_EX_ID = "EX_glc_D"   # glucose exchange reaction ID
O2_EX_ID  = "EX_o2"      # oxygen exchange reaction ID
GLN_EX_ID = "EX_gln_L"   # glutamine exchange reaction ID
LAC_EX_ID = "EX_lac_L"   # lactate exchange (optional, for lactate uptake control)


def set_environment_TME(m, scenario):
    """
    Set nutrient / oxygen constraints to mimic different environments.

    Parameters
    ----------
    m : cobra.Model
        A model copy whose bounds will be modified in-place.
    scenario : {"Blood", "Edge", "Core"}
        Environment type:
          - "Blood": rich / blood-like, nutrient-replete.
          - "Edge":  moderate limitation (tumor edge).
          - "Core":  severe TME (tumor core).

    Returns
    -------
    cobra.Model
        The same model object, with modified exchange bounds.
    """
    # Default: do nothing if exchange reaction not found
    def safe_set_lb(rxn_id, lb):
        if rxn_id in m.reactions:
            m.reactions.get_by_id(rxn_id).lower_bound = lb
        else:
            print(f"[Warning] Exchange '{rxn_id}' not found in model; skipped.")

    # Recommended values (mmol/gDW/h), within typical literature ranges:
    #   Rich:    glucose ~ -8 to -15, oxygen ~ -15 to -25, glutamine ~ -8 to -12
    #   Edge:    ~30–50% of rich
    #   Core:    ~5–20% of rich (severe limitation)

    if scenario == "Blood":
        glc_lb = -10.0
        o2_lb  = -20.0
        gln_lb = -10.0
        lactate_lb = -10.0  # allow lactate uptake/export (or set to 0 if you prefer no uptake)
    elif scenario == "Edge":
        glc_lb = -3.0
        o2_lb  = -8.0
        gln_lb = -3.0
        lactate_lb = 0.0    # do not allow lactate uptake (only secretion)
    elif scenario == "Core":
        glc_lb = -1.0
        o2_lb  = -2.0
        gln_lb = -1.0
        lactate_lb = 0.0    # no lactate uptake, mimic lactate accumulation
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    print(f"Setting environment for {scenario}: "
          f"glc={glc_lb}, o2={o2_lb}, gln={gln_lb}, lac_lb={lactate_lb}")

    safe_set_lb(GLC_EX_ID, glc_lb)
    safe_set_lb(O2_EX_ID,  o2_lb)
    safe_set_lb(GLN_EX_ID, gln_lb)
    safe_set_lb(LAC_EX_ID, lactate_lb)

    return m

# 3) E-Flux helper (same logic as Day18)

def apply_eflux(model, expr_norm):
    """
    Apply E-Flux scaling to reaction bounds.

    Parameters
    ----------
    model : cobra.Model
        Original metabolic model.
    expr_norm : pandas.Series
        Normalized expression (0–1) indexed by reaction ID.

    Returns
    -------
    cobra.Model
        A model copy with scaled reaction bounds.
    """
    m = model.copy()

    for rxn in m.reactions:
        # Keep biomass reactions unscaled
        if "biomass" in rxn.id.lower():
            continue

        # Use expression value if available; otherwise assign a tiny value
        if rxn.id in expr_norm.index:
            scale = float(expr_norm[rxn.id])
        else:
            scale = 1e-6

        scale = max(scale, 1e-6)

        # Scale bounds
        if rxn.upper_bound > 0:
            rxn.upper_bound *= scale
        if rxn.lower_bound < 0:
            rxn.lower_bound *= scale

    return m


# 4) Map sample name → environment scenario

def infer_scenario_from_sample(sample_name):
    """
    Map sample name to environment scenario.
    Assumes sample names like "Blood", "TumorCore", "TumorEdge".
    Modify this mapping if your sample naming is different.
    """
    s = sample_name.lower()
    if "blood" in s:
        return "Blood"
    elif "core" in s:
        return "Core"
    elif "edge" in s:
        return "Edge"
    else:
        # Default: treat as Blood-like if not recognized
        print(f"[Warning] Cannot infer scenario from sample '{sample_name}', using 'Blood'.")
        return "Blood"

# 5) Run E-Flux under TME constraints for each sample

biomass_records = []

for sample in samples:
    print(f"\n=== Running TME-constrained E-Flux for sample: {sample} ===")

    expr_vec = rxn_expr[sample]
    scenario = infer_scenario_from_sample(sample)
    print(f"Environment scenario inferred: {scenario}")

    # Normalize expression to [0,1]
    max_val = expr_vec.max()
    if max_val <= 0:
        print(f"Warning: sample {sample} has non-positive max expression. Skipped.")
        continue

    expr_norm = expr_vec / max_val
    print(f"Expression stats for {sample}: mean={expr_vec.mean():.3f}, std={expr_vec.std():.3f}")

    # Start from the original model:
    #   1) Set TME environment (exchange bounds)
    #   2) Apply E-Flux scaling on top of that
    print("Setting TME environment bounds...")
    model_env = set_environment_TME(model.copy(), scenario)

    print("Applying E-Flux scaling...")
    model_scaled = apply_eflux(model_env, expr_norm)

    # Run FBA
    print("Running FBA under TME constraints...")
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
    flux_path = os.path.join(OUT_DIR, f"day22_flux_TME_{sample}.csv")
    flux_series.to_csv(flux_path)
    print(f"Saved TME-constrained flux distribution: {flux_path}")

    # Plot histogram of flux distribution
    plt.figure(figsize=(6, 4))
    plt.hist(flux_series, bins=80)
    plt.xlabel("Flux value")
    plt.ylabel("Count")
    plt.title(f"TME-constrained E-Flux distribution — {sample} ({scenario})")
    plt.tight_layout()

    fig_out = os.path.join(OUT_DIR, f"day22_flux_hist_TME_{sample}.png")
    plt.savefig(fig_out, dpi=150)
    plt.close()
    print(f"Saved flux histogram: {fig_out}")


# 6) Save biomass summary across samples

if biomass_records:
    df_biomass = pd.DataFrame(biomass_records)
    biomass_out = os.path.join(OUT_DIR, "day22_TME_biomass_summary_Tcells.csv")
    df_biomass.to_csv(biomass_out, index=False)
    print(f"\nSaved biomass summary: {biomass_out}")

print("\nDay22 TME-constrained E-Flux analysis completed.")

