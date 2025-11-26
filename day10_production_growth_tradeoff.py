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

# IDs in the textbook model
BIOMASS_RXN_ID = "Biomass_Ecoli_core"   # growth objective
GLUCOSE_EX_ID  = "EX_glc__D_e"          # glucose exchange (uptake)
PRODUCT_RXN_ID = "EX_ac_e"              # product to track (acetate export, change if you want)

# Range of glucose lower bounds (negative = uptake)
# Example: from -5 to -20 mmol/gDW/h
GLC_LB_VALUES = np.linspace(-5.0, -20.0, 16)  # 16 steps between -5 and -20



# Part 1 – Helper: run FBA for a given glucose bound
def run_fba_with_glucose_limit(base_model, glc_lb):
    """
    Set the glucose uptake lower bound, run FBA, and return:
    - status (optimal / infeasible etc.)
    - biomass objective value
    - product flux
    """
    # Use context manager so we don't permanently change the original model
    with base_model as model:
        # Set objective to biomass explicitly (just to be safe)
        model.objective = BIOMASS_RXN_ID

        # Set glucose uptake lower bound
        rxn_glc = model.reactions.get_by_id(GLUCOSE_EX_ID)
        rxn_glc.lower_bound = glc_lb

        # Optimize model
        solution = model.optimize()

        if solution.status != "optimal":
            return solution.status, np.nan, np.nan

        biomass = solution.objective_value
        product_flux = solution.fluxes.get(PRODUCT_RXN_ID, np.nan)

        return solution.status, biomass, product_flux



# Part 2 – Main scan over glucose bounds
def main():
    print("Loading model ...")
    model = load_model("textbook")
    print("Loaded model:", model)

    # Check that reactions exist
    assert BIOMASS_RXN_ID in model.reactions, f"{BIOMASS_RXN_ID} not in model"
    assert GLUCOSE_EX_ID  in model.reactions, f"{GLUCOSE_EX_ID} not in model"
    assert PRODUCT_RXN_ID in model.reactions, f"{PRODUCT_RXN_ID} not in model"

    results = []

    print("\n[Part 1] Scanning glucose uptake limits ...")
    for glc_lb in GLC_LB_VALUES:
        status, biomass, product_flux = run_fba_with_glucose_limit(model, glc_lb)

        print(f"  glucose_lb = {glc_lb:6.2f}  ->  status = {status:8s}, "
              f"biomass = {biomass:.4f}, product = {product_flux:.4f}")

        results.append({
            "glucose_lb": glc_lb,
            "status": status,
            "biomass": biomass,
            "product_flux": product_flux
        })

    df = pd.DataFrame(results)


    # Part 3 – Save numeric results
    out_csv = os.path.join(DATA_DIR, "day10_glucose_tradeoff.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved trade-off table to: {out_csv}")

    # Filter feasible solutions (status == "optimal")
    df_ok = df[df["status"] == "optimal"].copy()


    # Part 4 – Plot biomass vs product (Pareto-style curve)
    plt.figure(figsize=(6, 5))
    plt.scatter(df_ok["biomass"], df_ok["product_flux"])
    plt.xlabel("Biomass growth rate")
    plt.ylabel(f"{PRODUCT_RXN_ID} flux")
    plt.title("Day 10 – Biomass vs product flux\n(glucose uptake scan)")
    plt.tight_layout()

    pareto_fig = os.path.join(FIG_DIR, "day10_pareto_biomass_vs_product.png")
    plt.savefig(pareto_fig, dpi=300)
    plt.close()
    print(f"Saved Pareto-style plot to: {pareto_fig}")

    # Part 5 – Plot glucose uptake vs biomass/product (optional)
    plt.figure(figsize=(6, 5))
    plt.plot(df_ok["glucose_lb"], df_ok["biomass"], marker="o", label="Biomass")
    plt.plot(df_ok["glucose_lb"], df_ok["product_flux"], marker="s", label=f"{PRODUCT_RXN_ID}")
    plt.xlabel("Glucose lower bound (uptake, negative)")
    plt.ylabel("Flux value")
    plt.title("Day 10 – Effect of glucose limit on growth and product")
    plt.legend()
    plt.tight_layout()

    tradeoff_fig = os.path.join(FIG_DIR, "day10_glucose_vs_fluxes.png")
    plt.savefig(tradeoff_fig, dpi=300)
    plt.close()
    print(f"Saved glucose vs flux plot to: {tradeoff_fig}")

    print("\nDone. Day 10 trade-off analysis finished.")


if __name__ == "__main__":
    main()

