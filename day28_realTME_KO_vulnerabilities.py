import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cobra.io import read_sbml_model

# ============================================================
# Day 28 – Realistic TME KO-based vulnerability screening
#
# Goal:
#   Under the same “realistic TME + E-Flux” conditions as Day 27,
#   perform reaction knock-out (KO) screening to identify
#   metabolically vulnerable reactions:
#
#      - essential_TME: biomass collapse (~0)
#      - strongly_sensitive_TME: biomass drops strongly, but not 100%
#
#   This is done separately for each sample:
#      Blood      → Rich medium
#      TumorEdge  → TME_edge medium
#      TumorCore  → TME_core medium
#
# Inputs:
#   - Human-GEM model (SBML)
#   - day17_reaction_expression_Tcells.csv   (reaction-level expression)
#
# Outputs:
#   - day28_KO_realTME_{sample}.csv
#       Columns:
#          reaction_id, reaction_name, subsystem,
#          baseline_biomass, baseline_flux,
#          ko_biomass, rel_drop,
#          is_essential_TME, is_strong_sensitive_TME
#
#   - day28_realTME_vulnerability_summary.csv
#       One line per sample, total counts of essential / strong-sensitive.
#
#   - day28_top_vulnerabilities_{sample}.csv
#       Top N (default 50) reactions ranked by rel_drop in TME.
#
# ============================================================

# -----------------------------
# Paths
# -----------------------------
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
# Exchange reaction IDs (same as Day 27)
# ============================================================

EX_GLC   = "MAR09034"   # Exchange of glucose
EX_O2    = "MAR09048"   # Exchange of O2
EX_GLN   = "MAR09063"   # Exchange of glutamine
EX_LAC_L = "MAR09135"   # Exchange of L-lactate
EX_LAC_D = "MAR09136"   # Exchange of D-lactate

# Optional: if you know IDs for these, fill them in; otherwise leave None
EX_SER = None
EX_ARG = None
EX_TRP = None


def clean_medium_dict(mdict):
    """Remove entries where rxn_id is None."""
    return {k: v for k, v in mdict.items() if k is not None}


# -----------------------------
# Rich / Edge / Core media (same pattern as Day 27)
# -----------------------------
medium_rich = clean_medium_dict({
    EX_GLC:   -10.0,
    EX_GLN:   -10.0,
    EX_O2:    -20.0,
    EX_LAC_L:  0.0,
    EX_LAC_D:  0.0,
    EX_SER:  -5.0 if EX_SER else None,
    EX_ARG:  -5.0 if EX_ARG else None,
    EX_TRP:  -2.0 if EX_TRP else None,
})

medium_TME_edge = clean_medium_dict({
    EX_GLC:   -3.0,
    EX_GLN:   -3.0,
    EX_O2:    -8.0,
    EX_LAC_L: -5.0,
    EX_LAC_D: -5.0,
    EX_SER:  -1.5 if EX_SER else None,
    EX_ARG:  -1.5 if EX_ARG else None,
    EX_TRP:  -0.6 if EX_TRP else None,
})

medium_TME_core = clean_medium_dict({
    EX_GLC:   -1.0,
    EX_GLN:   -1.0,
    EX_O2:    -2.0,
    EX_LAC_L: -10.0,
    EX_LAC_D: -10.0,
    EX_SER:  -0.5 if EX_SER else None,
    EX_ARG:  -0.5 if EX_ARG else None,
    EX_TRP:  -0.2 if EX_TRP else None,
})


medium_records = []  # optional, in case you want to log bounds again


def apply_medium_bounds(m, medium_dict, scenario_name):
    """
    Apply nutrient bounds to a model for a given scenario.
    Bounds are modified in-place.
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
            print(f"[Warning] Exchange '{rxn_id}' not found; skipped.")

    print(f"Applying medium for '{scenario_name}' ...")
    for rid, lb in medium_dict.items():
        safe_set_lb(rid, lb)


# ============================================================
# E-Flux helper
# ============================================================

def apply_eflux(model, expr_norm):
    """
    Apply E-Flux scaling to reaction bounds using normalized
    reaction-level expression.
    """
    m = model.copy()

    for rxn in m.reactions:
        # Keep biomass reactions unscaled
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
# Scenario mapping: sample → medium
# ============================================================

def infer_scenario_from_sample(sample_name: str) -> str:
    s = sample_name.lower()
    if "blood" in s:
        return "Rich"
    elif "edge" in s:
        return "TME_edge"
    elif "core" in s:
        return "TME_core"
    else:
        print(f"[Warning] Cannot infer scenario for '{sample_name}', using 'Rich'.")
        return "Rich"


# ============================================================
# KO screening helper
# ============================================================

def get_ko_candidate_reactions(m):
    """
    Select reactions to be knocked out:
      - Exclude boundary/exchange/demand/sink reactions
      - Exclude biomass/objective reactions
    """
    boundary_ids = {rxn.id for rxn in m.boundary}
    objective_ids = {rxn.id for rxn in m.reactions if abs(rxn.objective_coefficient) > 0}

    candidates = []
    for rxn in m.reactions:
        if rxn.id in boundary_ids:
            continue
        if rxn.id in objective_ids:
            continue
        candidates.append(rxn)

    print(f"KO candidates: {len(candidates)} reactions (excluding boundary & biomass).")
    return candidates


def run_KO_screen(model_scaled, baseline_sol, sample_name, scenario_name):
    """
    Run single-reaction KO screening on a given scaled model.
    Returns a DataFrame with KO results for this sample.
    """
    baseline_biomass = baseline_sol.objective_value
    if baseline_biomass is None:
        baseline_biomass = 0.0

    # To avoid division by zero
    denom = max(abs(baseline_biomass), 1e-9)

    baseline_flux = baseline_sol.fluxes

    candidates = get_ko_candidate_reactions(model_scaled)

    records = []

    for idx, rxn in enumerate(candidates, start=1):
        # Periodic progress report
        if idx % 500 == 0:
            print(f"[{sample_name}] KO progress: {idx}/{len(candidates)} reactions ...")

        # Store original bounds
        old_lb = rxn.lower_bound
        old_ub = rxn.upper_bound

        # KO: force flux to 0
        rxn.lower_bound = 0.0
        rxn.upper_bound = 0.0

        sol_ko = model_scaled.optimize()

        ko_biomass = sol_ko.objective_value if sol_ko.status == "optimal" else 0.0
        rel_drop = (baseline_biomass - ko_biomass) / denom

        # Restore bounds
        rxn.lower_bound = old_lb
        rxn.upper_bound = old_ub

        is_essential = rel_drop >= 0.9999   # essentially kills growth
        is_strong = (rel_drop >= 0.5) and (rel_drop < 0.9999)

        records.append({
            "sample": sample_name,
            "scenario": scenario_name,
            "reaction_id": rxn.id,
            "reaction_name": rxn.name,
            "subsystem": rxn.subsystem,
            "baseline_biomass": baseline_biomass,
            "baseline_flux": float(baseline_flux.get(rxn.id, 0.0)),
            "ko_biomass": ko_biomass,
            "rel_drop": rel_drop,
            "is_essential_TME": is_essential,
            "is_strong_sensitive_TME": is_strong,
        })

    df = pd.DataFrame(records)
    return df


# ============================================================
# Main loop over samples
# ============================================================

all_results = []
summary_records = []

for sample in samples:
    print("\n==================================================")
    print(f"Day 28 – Realistic TME KO screening for sample: {sample}")
    print("==================================================")

    expr_vec = rxn_expr[sample]
    max_val = expr_vec.max()
    if max_val <= 0:
        print(f"[Warning] Sample {sample} has non-positive max expression; skipping.")
        continue

    expr_norm = expr_vec / max_val
    print(f"Expression stats for {sample}: mean={expr_vec.mean():.3f}, std={expr_vec.std():.3f}")

    scenario = infer_scenario_from_sample(sample)
    print(f"Scenario for {sample}: {scenario}")

    # Build environment model
    model_env = model.copy()
    if scenario == "Rich":
        apply_medium_bounds(model_env, medium_rich, scenario_name=scenario)
    elif scenario == "TME_edge":
        apply_medium_bounds(model_env, medium_TME_edge, scenario_name=scenario)
    elif scenario == "TME_core":
        apply_medium_bounds(model_env, medium_TME_core, scenario_name=scenario)
    else:
        apply_medium_bounds(model_env, medium_rich, scenario_name=scenario)

    # Apply E-Flux
    print("Applying E-Flux scaling ...")
    model_scaled = apply_eflux(model_env, expr_norm)

    # Baseline FBA
    print("Running baseline FBA before KO ...")
    baseline_sol = model_scaled.optimize()
    print(f"Baseline status: {baseline_sol.status}, biomass={baseline_sol.objective_value}")

    if baseline_sol.status != "optimal":
        print(f"[Warning] Baseline FBA not optimal for {sample}; skipping KO.")
        continue

    # KO screen
    df_sample = run_KO_screen(model_scaled, baseline_sol, sample_name=sample, scenario_name=scenario)

    # Save per-sample KO table
    out_csv = os.path.join(OUT_DIR, f"day28_KO_realTME_{sample}.csv")
    df_sample.to_csv(out_csv, index=False)
    print(f"Saved KO table for {sample}: {out_csv}")

    # Summaries: how many essential / strong-sensitive reactions?
    n_ess = int(df_sample["is_essential_TME"].sum())
    n_strong = int(df_sample["is_strong_sensitive_TME"].sum())

    summary_records.append({
        "sample": sample,
        "scenario": scenario,
        "n_reactions": len(df_sample),
        "n_essential_TME": n_ess,
        "n_strong_sensitive_TME": n_strong,
    })

    # Save top vulnerabilities for quick inspection
    df_sorted = df_sample.sort_values("rel_drop", ascending=False)
    topN = 50
    out_top = os.path.join(OUT_DIR, f"day28_top_vulnerabilities_{sample}.csv")
    df_sorted.head(topN).to_csv(out_top, index=False)
    print(f"Saved top {topN} vulnerabilities for {sample}: {out_top}")

    all_results.append(df_sample)


# ============================================================
# Global summary
# ============================================================

if summary_records:
    df_summary = pd.DataFrame(summary_records)
    summary_out = os.path.join(OUT_DIR, "day28_realTME_vulnerability_summary.csv")
    df_summary.to_csv(summary_out, index=False)
    print(f"\nSaved vulnerability summary: {summary_out}")

print("\nDay 28 KO-based vulnerability screening completed.")

