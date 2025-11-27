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

# Reaction IDs
BIOMASS_RXN_ID = "Biomass_Ecoli_core"
PRODUCT_RXN_ID = "EX_ac_e"          # same as Day11

# Choose one Pareto point to compare with WT
# e.g. biomass fixed at 30% of WT
TARGET_FRACTION = 0.30

# Small value to avoid division by zero in log2 fold-change
EPS = 1e-9

# Helper – run constrained model at a given biomass fraction


def run_pareto_point(base_model, fraction, wt_biomass):
    """
    Fix biomass to (fraction * wt_biomass) and maximize PRODUCT_RXN_ID.

    Returns:
        solution (cobra.Solution)
    """
    with base_model as model:
        biomass_rxn = model.reactions.get_by_id(BIOMASS_RXN_ID)
        product_rxn = model.reactions.get_by_id(PRODUCT_RXN_ID)

        target_bm = fraction * wt_biomass
        biomass_rxn.lower_bound = target_bm
        biomass_rxn.upper_bound = target_bm

        model.objective = product_rxn
        sol = model.optimize()
        return sol


# Part 1 – Load model and compute WT fluxes
print("Loading model ...")
model = load_model("textbook")
print("Loaded model:", model)

assert BIOMASS_RXN_ID in model.reactions, f"{BIOMASS_RXN_ID} not found."
assert PRODUCT_RXN_ID in model.reactions, f"{PRODUCT_RXN_ID} not found."

# WT: maximize biomass
print("\n[Part 1] Running WT FBA (maximize biomass) ...")
model.objective = BIOMASS_RXN_ID
wt_solution = model.optimize()

if wt_solution.status != "optimal":
    raise RuntimeError("WT optimization failed.")

wt_biomass = wt_solution.objective_value
print(f"WT biomass = {wt_biomass:.4f}")

wt_flux = wt_solution.fluxes  # pandas Series: index = rxn id


# Part 2 – Run Pareto point at TARGET_FRACTION
print(
    f"\n[Part 2] Running Pareto point at biomass fraction = {TARGET_FRACTION:.2f} ...")
pareto_solution = run_pareto_point(model, TARGET_FRACTION, wt_biomass)

if pareto_solution.status != "optimal":
    raise RuntimeError("Pareto-point optimization failed.")

pareto_flux = pareto_solution.fluxes

print(f"Pareto point biomass (should be ~{TARGET_FRACTION:.2f} * WT): "
      f"{pareto_flux[BIOMASS_RXN_ID]:.4f}")
print(f"Pareto point product flux ({PRODUCT_RXN_ID}): "
      f"{pareto_flux[PRODUCT_RXN_ID]:.4f}")


# Part 3 – Build reaction-wise comparison table
print("\n[Part 3] Building reaction-wise comparison table ...")

rows = []
for rxn in model.reactions:
    rid = rxn.id
    name = rxn.name
    subsystem = rxn.subsystem if rxn.subsystem not in (None, "") else "NA"

    v_wt = wt_flux.get(rid, 0.0)
    v_pt = pareto_flux.get(rid, 0.0)

    delta = v_pt - v_wt
    abs_delta = abs(delta)
    # log2 fold-change of |flux|
    log2_fc = np.log2((abs(v_pt) + EPS) / (abs(v_wt) + EPS))

    rows.append(
        {
            "rxn_id": rid,
            "reaction_name": name,
            "subsystem": subsystem,
            "flux_WT": v_wt,
            "flux_Pareto": v_pt,
            "delta": delta,
            "abs_delta": abs_delta,
            "log2_FC_abs": log2_fc,
        }
    )

df = pd.DataFrame(rows)

out_csv = os.path.join(
    DATA_DIR,
    f"day12_flux_comparison_bm{int(TARGET_FRACTION*100)}.csv"
)
df.to_csv(out_csv, index=False)
print(f"Saved flux comparison table to: {out_csv}")


# Part 4 – Volcano-style scatter plot: delta vs log2FC
print("\n[Part 4] Plotting volcano-style scatter ...")

plt.figure(figsize=(7, 6))
plt.scatter(df["delta"], df["log2_FC_abs"], s=10)
plt.axhline(0.0, linestyle="--")
plt.xlabel("Flux change (Pareto - WT)")
plt.ylabel("log2 fold-change (|Pareto| / |WT|)")
plt.title(
    f"Day 12 – Flux rewiring at biomass fraction = {TARGET_FRACTION:.2f}"
)
plt.tight_layout()

volcano_fig = os.path.join(
    FIG_DIR,
    f"day12_volcano_bm{int(TARGET_FRACTION*100)}.png"
)
plt.savefig(volcano_fig, dpi=300)
plt.close()
print(f"Saved volcano-style plot to: {volcano_fig}")


# Part 5 – Top up- and down-regulated reactions (bar plots)
print("\n[Part 5] Plotting top up/down reactions ...")

# Sort by delta (positive: increased in Pareto; negative: decreased)
df_sorted = df.sort_values("delta", ascending=False)

top_up = df_sorted.head(10).copy()
top_down = df_sorted.tail(10).copy()  # most negative deltas

# Barplot for top up-regulated reactions
plt.figure(figsize=(8, 5))
plt.barh(top_up["rxn_id"], top_up["delta"])
plt.gca().invert_yaxis()
plt.xlabel("Flux change (Pareto - WT)")
plt.title(
    f"Day 12 – Top 10 up-regulated reactions\n(biomass fraction = {TARGET_FRACTION:.2f})"
)
plt.tight_layout()

up_fig = os.path.join(
    FIG_DIR,
    f"day12_top_upregulated_bm{int(TARGET_FRACTION*100)}.png"
)
plt.savefig(up_fig, dpi=300)
plt.close()
print(f"Saved top-up barplot to: {up_fig}")

# Barplot for top down-regulated reactions
plt.figure(figsize=(8, 5))
plt.barh(top_down["rxn_id"], top_down["delta"])
plt.gca().invert_yaxis()
plt.xlabel("Flux change (Pareto - WT)")
plt.title(
    f"Day 12 – Top 10 down-regulated reactions\n(biomass fraction = {TARGET_FRACTION:.2f})"
)
plt.tight_layout()

down_fig = os.path.join(
    FIG_DIR,
    f"day12_top_downregulated_bm{int(TARGET_FRACTION*100)}.png"
)
plt.savefig(down_fig, dpi=300)
plt.close()
print(f"Saved top-down barplot to: {down_fig}")

print("\nDone. Day 12 flux rewiring analysis finished.")
