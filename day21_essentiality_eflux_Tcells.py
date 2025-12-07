import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cobra.io import read_sbml_model

# ------------------------------------------------------------
# Day 21 – Reaction essentiality under expression-constrained
#          E-Flux models (Blood / TumorCore / TumorEdge)
#
# Goal:
#   For each condition (Blood/Core/Edge), apply E-Flux using
#   reaction-level expression and perform single-reaction
#   knockout analysis to identify condition-specific
#   essential reactions.
# ------------------------------------------------------------

# Paths
BASE = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = os.path.join(BASE, "../database/Human-GEM.xml")
REACTION_EXPR_FILE = os.path.join(BASE, "day17_reaction_expression_Tcells.csv")

OUT_DIR = BASE
os.makedirs(OUT_DIR, exist_ok=True)



# 1) Load Human-GEM model and reaction expression

print(f"Loading model: {MODEL_FILE}")
model = read_sbml_model(MODEL_FILE)
print(f"Model loaded: {len(model.genes)} genes, {len(model.reactions)} reactions")

print(f"Loading reaction-level expression: {REACTION_EXPR_FILE}")
rxn_expr = pd.read_csv(REACTION_EXPR_FILE, index_col=0)
samples = list(rxn_expr.columns)
print("Samples found in expression matrix:", samples)

# Reaction annotations
rxn_ids = [rxn.id for rxn in model.reactions]
rxn_name_map = {rxn.id: (rxn.name or "") for rxn in model.reactions}
rxn_subsystem_map = {rxn.id: (rxn.subsystem or "NA") for rxn in model.reactions}


# 2) E-Flux helper – same logic as Day18

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

# 3) Essentiality analysis per sample

def compute_reaction_essentiality(sample_name, expr_vec, model, eps_flux=1e-6):
    """
    For a given sample, build an expression-constrained model
    using E-Flux and perform single-reaction knockout analysis.

    Parameters
    ----------
    sample_name : str
        Name of the condition (e.g. Blood, TumorCore, TumorEdge).
    expr_vec : pandas.Series
        Raw reaction-level expression values for this sample.
    model : cobra.Model
        Base metabolic model.
    eps_flux : float
        Threshold to treat a reaction as carrying non-zero flux.

    Returns
    -------
    pandas.DataFrame
        Essentiality summary for reactions with non-zero baseline flux.
    """
    print(f"\n=== Essentiality analysis for sample: {sample_name} ===")

    # Normalize expression to [0,1]
    max_val = expr_vec.max()
    if max_val <= 0:
        print(f"Warning: sample {sample_name} has non-positive max expression. Skipped.")
        return None

    expr_norm = expr_vec / max_val

    # Build E-Flux scaled model
    print("Applying E-Flux scaling...")
    m = apply_eflux(model, expr_norm)

    # Baseline FBA
    print("Running baseline FBA...")
    sol = m.optimize()
    baseline_status = sol.status
    baseline_biomass = sol.objective_value
    print(f"Baseline status: {baseline_status}")
    print(f"Baseline biomass: {baseline_biomass}")

    if baseline_status != "optimal" or baseline_biomass is None:
        print(f"Sample {sample_name}: baseline solution not optimal. Skipping essentiality.")
        return None

    baseline_flux = sol.fluxes

    # Consider only reactions that carry non-zero flux at baseline
    candidate_rxns = [rid for rid in baseline_flux.index if abs(baseline_flux[rid]) > eps_flux]
    print(f"Number of reactions with |flux| > {eps_flux}: {len(candidate_rxns)}")

    records = []

    for rid in candidate_rxns:
        rxn = m.reactions.get_by_id(rid)

        # Save original bounds
        lb_orig = rxn.lower_bound
        ub_orig = rxn.upper_bound

        # Knock out reaction by forcing flux to 0
        rxn.lower_bound = 0.0
        rxn.upper_bound = 0.0

        sol_ko = m.optimize()
        ko_status = sol_ko.status
        ko_biomass = sol_ko.objective_value if ko_status == "optimal" else 0.0

        # Restore original bounds
        rxn.lower_bound = lb_orig
        rxn.upper_bound = ub_orig

        if baseline_biomass > 1e-9:
            rel_biomass = ko_biomass / baseline_biomass
            drop = 1.0 - rel_biomass
        else:
            rel_biomass = np.nan
            drop = np.nan

        essential_flag = (ko_biomass < 1e-6) or (drop > 0.9)

        records.append({
            "rxn_id": rid,
            "name": rxn_name_map.get(rid, ""),
            "subsystem": rxn_subsystem_map.get(rid, "NA"),
            "baseline_flux": baseline_flux[rid],
            "baseline_biomass": baseline_biomass,
            "ko_status": ko_status,
            "ko_biomass": ko_biomass,
            "rel_biomass": rel_biomass,
            "drop_in_biomass": drop,
            "essential": essential_flag,
        })

    df = pd.DataFrame(records)
    df = df.sort_values("drop_in_biomass", ascending=False)

    return df


# 4) Run essentiality for all samples and build summary

per_sample_results = {}
summary_rows = []

for sample in samples:
    expr_vec = rxn_expr[sample]

    df_sample = compute_reaction_essentiality(sample, expr_vec, model)

    if df_sample is None:
        continue

    # Save per-sample table
    out_csv = os.path.join(OUT_DIR, f"day21_essentiality_{sample}.csv")
    df_sample.to_csv(out_csv, index=False)
    print(f"Saved essentiality table for {sample}: {out_csv}")

    per_sample_results[sample] = df_sample

# Build a combined summary over all reactions
if per_sample_results:
    # Start with reaction annotations
    all_rxns = rxn_ids
    combined = pd.DataFrame({
        "rxn_id": all_rxns,
        "name": [rxn_name_map[r] for r in all_rxns],
        "subsystem": [rxn_subsystem_map[r] for r in all_rxns],
    })

    # For each sample, merge essentiality info
    for sample, df_sample in per_sample_results.items():
        df_small = df_sample[["rxn_id", "essential", "drop_in_biomass"]].copy()
        df_small = df_small.rename(columns={
            "essential": f"essential_{sample}",
            "drop_in_biomass": f"drop_{sample}",
        })
        combined = combined.merge(df_small, on="rxn_id", how="left")

    summary_out = os.path.join(OUT_DIR, "day21_essentiality_summary_all_samples.csv")
    combined.to_csv(summary_out, index=False)
    print(f"\nSaved combined essentiality summary: {summary_out}")

print("\nDay21 essentiality analysis completed.")

