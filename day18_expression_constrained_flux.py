import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cobra.io import read_sbml_model

# Paths (relative)

BASE = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = os.path.join(BASE, "../database/Human-GEM.xml")
REACTION_EXPR_FILE = os.path.join(BASE, "day17_reaction_expression_Tcells.csv")

OUT_DIR = BASE
os.makedirs(OUT_DIR, exist_ok=True)


# Load Human-GEM model

print(f"Loading model: {MODEL_FILE}")
model = read_sbml_model(MODEL_FILE)
print(f"Model loaded: {len(model.genes)} genes, {len(model.reactions)} reactions")


# Load reaction-level expression matrix (Day17 output)

print(f"Loading reaction-level expression: {REACTION_EXPR_FILE}")
rxn_expr = pd.read_csv(REACTION_EXPR_FILE, index_col=0)

samples = list(rxn_expr.columns)
print(f"Samples found: {samples}")


# E-Flux helper function


def apply_eflux(model, expr_norm):
    """
    Apply E-Flux scaling to reaction bounds.
    
    Parameters
    ----------
    model : cobra.Model
        The original metabolic model.
    expr_norm : pandas.Series
        Normalized expression (0–1) indexed by reaction ID.

    Returns
    -------
    cobra.Model
        A model copy with scaled reaction bounds.
    """
    m = model.copy()

    for rxn in m.reactions:

        # Keep biomass reaction unscaled
        if "biomass" in rxn.id.lower():
            continue

        # Use expression value if available; otherwise assign very small value
        if rxn.id in expr_norm.index:
            scale = float(expr_norm[rxn.id])
        else:
            scale = 1e-6

        # Prevent complete shutdown of reactions (minimum allowed)
        scale = max(scale, 1e-6)

        # Scale bounds
        if rxn.upper_bound > 0:
            rxn.upper_bound *= scale
        if rxn.lower_bound < 0:
            rxn.lower_bound *= scale

    return m

# Run E-Flux + FBA for all samples

for SAMPLE in samples:
    print(f"\n=== Running E-Flux for sample: {SAMPLE} ===")

    expr_vec = rxn_expr[SAMPLE]

    # Normalize expression to [0,1]
    max_val = expr_vec.max()
    if max_val <= 0:
        print(f"Warning: sample {SAMPLE} has non-positive max expression. Skipped.")
        continue

    expr_norm = expr_vec / max_val

    print(f"Expression stats for {SAMPLE}: mean={expr_vec.mean():.3f}, std={expr_vec.std():.3f}")

    print("Applying E-Flux scaling...")
    model_scaled = apply_eflux(model, expr_norm)

    # Run FBA
    print("Running FBA...")
    sol = model_scaled.optimize()

    print(f"FBA status: {sol.status}")
    print(f"Biomass (objective value): {sol.objective_value}")

    # Save fluxes
    flux_series = sol.fluxes
    flux_path = os.path.join(OUT_DIR, f"day18_flux_{SAMPLE}.csv")
    flux_series.to_csv(flux_path)
    print(f"Saved flux distribution: {flux_path}")

    # Plot histogram of flux distribution
    plt.figure(figsize=(6, 4))
    plt.hist(flux_series, bins=80)
    plt.xlabel("Flux value")
    plt.ylabel("Count")
    plt.title(f"E-Flux constrained flux distribution — {SAMPLE}")

    fig_out = os.path.join(OUT_DIR, f"day18_flux_hist_{SAMPLE}.png")
    plt.tight_layout()
    plt.savefig(fig_out, dpi=150)
    plt.close()

    print(f"Saved histogram: {fig_out}")

print("\nDay18 completed for all samples.")
