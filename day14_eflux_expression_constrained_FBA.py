import os
import cobra
from cobra.io import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Part 0 – Settings and output folders
DATA_DIR = "data"
FIG_DIR = "figures"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

BIOMASS_RXN_ID = "Biomass_Ecoli_core"

# File from Day13
RXN_EXPR_FILE = os.path.join(
    DATA_DIR, "day13_reaction_expression_from_GPR.csv")

# Small epsilon to avoid division by zero
EPS = 1e-9

# Helper – scale bounds using reaction expression (E-Flux style)


def map_expression_to_scale(expr_series, min_scale=0.01, max_scale=1.0, quantile_ref=0.95):
    """
    Map reaction_expression values to [min_scale, max_scale].
    We normalize by a reference quantile to avoid one or two extreme values
    dominating the whole scaling.

    expr_series : pd.Series (reaction_expression)
    """
    expr = expr_series.copy().astype(float)

    # Replace negative / NaN with 0
    expr = expr.fillna(0.0)
    expr[expr < 0] = 0.0

    # Use a high quantile as reference (e.g. 95% expression level)
    ref_val = np.quantile(expr, quantile_ref)
    if ref_val < EPS:
        ref_val = expr.max() if expr.max() > EPS else 1.0

    expr_norm = expr / ref_val
    expr_norm = expr_norm.clip(lower=0.0, upper=1.0)

    # Map to [min_scale, max_scale]
    scale = min_scale + (max_scale - min_scale) * expr_norm
    return scale


# Part 1 – Load model and WT reference solution
print("Loading model ...")
model = load_model("textbook")
print("Loaded model:", model)

assert BIOMASS_RXN_ID in model.reactions, f"{BIOMASS_RXN_ID} not in model."

print("\n[Part 1] Running WT FBA (no expression constraints) ...")
model.objective = BIOMASS_RXN_ID
wt_solution = model.optimize()

if wt_solution.status != "optimal":
    raise RuntimeError("WT optimization failed.")

wt_biomass = wt_solution.objective_value
print(f"WT biomass = {wt_biomass:.4f}")

wt_flux = wt_solution.fluxes  # Series (rxn_id -> flux)

# Part 2 – Load reaction expression from Day13
print("\n[Part 2] Loading reaction expression from Day13 ...")
df_rxn_expr = pd.read_csv(RXN_EXPR_FILE)

if "rxn_id" not in df_rxn_expr.columns or "reaction_expression" not in df_rxn_expr.columns:
    raise ValueError(
        "day13 CSV must contain 'rxn_id' and 'reaction_expression' columns.")

# Restrict to reactions that exist in the model
df_rxn_expr = df_rxn_expr[df_rxn_expr["rxn_id"].isin(
    [rxn.id for rxn in model.reactions])].copy()

print(
    f"Reaction rows in expression file (after filtering): {len(df_rxn_expr)}")

# Compute scale factors
df_rxn_expr["scale_factor"] = map_expression_to_scale(
    df_rxn_expr["reaction_expression"])

scale_dict = dict(zip(df_rxn_expr["rxn_id"], df_rxn_expr["scale_factor"]))

# Part 3 – Build an expression-constrained model (E-Flux)
print("\n[Part 3] Building expression-constrained model (E-Flux) ...")

# Use a context manager so we don't permanently modify the original model
with model as eflux_model:
    # We will adjust bounds for internal reactions (skip pure exchange/boundary reactions)
    for rxn in eflux_model.reactions:
        rid = rxn.id

        # Skip boundary / exchange reactions to keep environment the same
        # (You can refine this condition if needed)
        if rxn.boundary:    # cobra marks EX_/DM_/sink_ as boundary
            continue

        # If we don't have expression for this reaction, keep original bounds
        if rid not in scale_dict:
            continue

        scale = float(scale_dict[rid])

        lb_old = rxn.lower_bound
        ub_old = rxn.upper_bound

        # For reversible reactions, scale both bounds symmetrically
        # For irreversible, scale in the allowed direction
        lb_new = lb_old * scale
        ub_new = ub_old * scale

        rxn.lower_bound = lb_new
        rxn.upper_bound = ub_new

    # Set the biomass objective again (growth under expression constraints)
    eflux_model.objective = BIOMASS_RXN_ID
    eflux_solution = eflux_model.optimize()

    if eflux_solution.status != "optimal":
        print("WARNING: E-Flux optimization not optimal:", eflux_solution.status)

    eflux_biomass = eflux_solution.objective_value
    print(f"E-Flux biomass = {eflux_biomass:.4f}")

    eflux_flux = eflux_solution.fluxes  # Series

# Part 4 – Save flux comparison table
print("\n[Part 4] Saving WT vs E-Flux flux comparison ...")

rows = []
for rxn in model.reactions:
    rid = rxn.id
    name = rxn.name
    subsystem = rxn.subsystem if rxn.subsystem not in (None, "") else "NA"

    v_wt = wt_flux.get(rid, 0.0)
    v_ef = eflux_flux.get(rid, 0.0)

    delta = v_ef - v_wt
    abs_delta = abs(delta)
    log2_fc = np.log2((abs(v_ef) + EPS) / (abs(v_wt) + EPS))

    scale = scale_dict.get(rid, np.nan)

    rows.append(
        {
            "rxn_id": rid,
            "reaction_name": name,
            "subsystem": subsystem,
            "flux_WT": v_wt,
            "flux_EFlux": v_ef,
            "delta": delta,
            "abs_delta": abs_delta,
            "log2_FC_abs": log2_fc,
            "scale_factor": scale,
        }
    )

df_flux_comp = pd.DataFrame(rows)

out_csv = os.path.join(DATA_DIR, "day14_WT_vs_EFlux_fluxes.csv")
df_flux_comp.to_csv(out_csv, index=False)
print(f"Saved WT vs E-Flux flux comparison to: {out_csv}")

# Part 5 – Scatter plot: WT vs E-Flux fluxes
print("\n[Part 5] Plotting WT vs E-Flux flux scatter ...")

plt.figure(figsize=(6, 6))
plt.scatter(df_flux_comp["flux_WT"], df_flux_comp["flux_EFlux"], s=10)
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.xlabel("WT flux")
plt.ylabel("E-Flux flux")
plt.title("Day 14 – WT vs E-Flux fluxes")
plt.tight_layout()

scatter_fig = os.path.join(FIG_DIR, "day14_WT_vs_EFlux_scatter.png")
plt.savefig(scatter_fig, dpi=300)
plt.close()
print(f"Saved scatter plot to: {scatter_fig}")


# Part 6 – Volcano-style plot: delta vs log2FC
print("\n[Part 6] Plotting volcano-style flux rewiring (WT vs E-Flux) ...")

plt.figure(figsize=(7, 6))
plt.scatter(df_flux_comp["delta"], df_flux_comp["log2_FC_abs"], s=10)
plt.axhline(0.0, linestyle="--")
plt.xlabel("Flux change (E-Flux - WT)")
plt.ylabel("log2 fold-change (|E-Flux| / |WT|)")
plt.title("Day 14 – Flux rewiring under expression constraints")
plt.tight_layout()

volcano_fig = os.path.join(FIG_DIR, "day14_WT_vs_EFlux_volcano.png")
plt.savefig(volcano_fig, dpi=300)
plt.close()
print(f"Saved volcano plot to: {volcano_fig}")

print("\nDay 14 – Expression-constrained FBA (E-Flux) finished.")
