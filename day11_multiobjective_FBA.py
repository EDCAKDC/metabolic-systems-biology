import os
import cobra
from cobra.io import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Part 0 – Output folders and basic settings
DATA_DIR = "data"
FIG_DIR = "figures"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Reaction IDs in the textbook model
BIOMASS_RXN_ID = "Biomass_Ecoli_core"
PRODUCT_RXN_ID = "EX_ac_e"

# Biomass fraction range (relative to WT biomass)
BIOMASS_FRACTIONS = np.linspace(1.0, 0.1, 20)


# Helper function
def run_multiobjective_step(base_model, fraction, wt_biomass):
    """Fix biomass to a fraction of WT and maximize product."""
    with base_model as model:
        biomass_rxn = model.reactions.get_by_id(BIOMASS_RXN_ID)
        product_rxn = model.reactions.get_by_id(PRODUCT_RXN_ID)

        target_bm = fraction * wt_biomass
        biomass_rxn.lower_bound = target_bm
        biomass_rxn.upper_bound = target_bm

        model.objective = product_rxn
        sol = model.optimize()

        if sol.status != "optimal":
            return sol.status, np.nan, np.nan

        return (
            sol.status,
            sol.fluxes.get(BIOMASS_RXN_ID, np.nan),
            sol.fluxes.get(PRODUCT_RXN_ID, np.nan),
        )


# Part 1 – Load model and get WT biomass
print("Loading model ...")
model = load_model("textbook")
print("Loaded model.")

model.objective = BIOMASS_RXN_ID
wt_solution = model.optimize()

if wt_solution.status != "optimal":
    raise RuntimeError("WT FBA failed.")

wt_biomass = wt_solution.objective_value
print(f"WT biomass = {wt_biomass:.4f}")


# Part 2 – Multi-objective scan
print("\nScanning biomass fractions ...")
results = []

for frac in BIOMASS_FRACTIONS:
    status, biomass, product_flux = run_multiobjective_step(
        model, frac, wt_biomass)

    print(
        f"  fraction={frac:.2f}  status={status:8s}  "
        f"biomass={biomass:.4f}  product={product_flux:.4f}"
    )

    results.append(
        {
            "biomass_fraction": frac,
            "status": status,
            "biomass": biomass,
            "product_flux": product_flux,
        }
    )

df = pd.DataFrame(results)


# Part 3 – Save numeric table
csv_path = os.path.join(DATA_DIR, "day11_biomass_product_pareto.csv")
df.to_csv(csv_path, index=False)
print(f"\nSaved table to {csv_path}")


# Part 4 – Pareto frontier plot
df_ok = df[df["status"] == "optimal"].copy()

plt.figure(figsize=(6, 5))
plt.plot(df_ok["biomass_fraction"], df_ok["product_flux"], marker="o")
plt.xlabel("Biomass fraction (relative to WT)")
plt.ylabel(f"{PRODUCT_RXN_ID} flux")
plt.title("Day 11 – Biomass–product Pareto frontier")
plt.tight_layout()

fig1 = os.path.join(FIG_DIR, "day11_pareto_biomass_vs_product.png")
plt.savefig(fig1, dpi=300)
plt.close()
print(f"Saved Pareto figure to {fig1}")


# Part 5 – Scatter plot: actual biomass vs product
plt.figure(figsize=(6, 5))
plt.scatter(df_ok["biomass"], df_ok["product_flux"])
plt.xlabel("Biomass flux (absolute)")
plt.ylabel(f"{PRODUCT_RXN_ID} flux")
plt.title("Day 11 – Actual biomass vs product")
plt.tight_layout()

fig2 = os.path.join(FIG_DIR, "day11_scatter_biomass_vs_product.png")
plt.savefig(fig2, dpi=300)
plt.close()
print(f"Saved scatter figure to {fig2}")

print("\nDay 11 multi-objective analysis finished.")
