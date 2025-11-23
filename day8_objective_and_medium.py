import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cobra
from cobra.io import load_model
# Output folders
DATA_DIR = "data"
FIG_DIR = "figures"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# 1. Load model and check wild-type biomass
print("Loading model ...")
model = load_model("textbook")

print("Running wild-type FBA ...")
wt_sol = model.optimize()
if wt_sol.status != "optimal":
    raise RuntimeError(f"WT solution not optimal (status={wt_sol.status})")

wt_growth = wt_sol.objective_value
print(f"WT biomass (default objective): {wt_growth:.4f}\n")


# Small helper to locate an exchange reaction safely
def get_rxn(model, rxn_id):
    """Return reaction by ID, raise a clear error if missing."""
    try:
        return model.reactions.get_by_id(rxn_id)
    except KeyError:
        raise KeyError(f"Reaction '{rxn_id}' not found in model. "
                       "Check the ID or change it in the script.")


# 2. Part A – Change objective function
print("[Part A] Comparing different objective functions ...")

objective_records = []

# A.1 – Default biomass objective
objective_records.append({
    "objective_type": "biomass_default",
    "objective_rxn": str(model.objective.expression),
    "max_value": wt_growth
})

# A.2 – Maximize ATP maintenance (ATPM)
with model:
    atpm_rxn = get_rxn(model, "ATPM")
    model.objective = atpm_rxn
    sol_atpm = model.optimize()
    objective_records.append({
        "objective_type": "ATPM",
        "objective_rxn": atpm_rxn.id,
        "max_value": sol_atpm.objective_value
    })
    print(f"Max ATPM flux: {sol_atpm.objective_value:.4f}")

# A.3 – Maximize lactate secretion (EX_lac__D_e)
with model:
    lac_ex = get_rxn(model, "EX_lac__D_e")
    model.objective = lac_ex
    sol_lac = model.optimize()
    objective_records.append({
        "objective_type": "lactate_secretion",
        "objective_rxn": lac_ex.id,
        "max_value": sol_lac.objective_value
    })
    print(f"Max lactate secretion: {sol_lac.objective_value:.4f}")

# A.4 – Maximize NADH dehydrogenase flux (NADH16)
with model:
    nadh_rxn = get_rxn(model, "NADH16")
    model.objective = nadh_rxn
    sol_nadh = model.optimize()
    objective_records.append({
        "objective_type": "NADH16",
        "objective_rxn": nadh_rxn.id,
        "max_value": sol_nadh.objective_value
    })
    print(f"Max NADH16 flux: {sol_nadh.objective_value:.4f}")

obj_df = pd.DataFrame(objective_records)
obj_path = os.path.join(DATA_DIR, "day8_objective_comparison.csv")
obj_df.to_csv(obj_path, index=False)
print(f"Saved objective comparison table to: {obj_path}\n")

# 3. Part B – Medium modification (environment changes)
print("[Part B] Testing different medium conditions ...")


def set_bound(rxn, lb=None, ub=None):
    """Set lower/upper bound of a reaction if values are provided."""
    if lb is not None:
        rxn.lower_bound = lb
    if ub is not None:
        rxn.upper_bound = ub


medium_records = []

# B.1 – Reference condition (copy WT biomass)
medium_records.append({
    "condition": "WT_reference",
    "EX_glc__D_e_lb": get_rxn(model, "EX_glc__D_e").lower_bound,
    "EX_o2_e_lb": get_rxn(model, "EX_o2_e").lower_bound,
    "EX_ac_e_lb": get_rxn(model, "EX_ac_e").lower_bound
    if "EX_ac_e" in model.reactions else np.nan,
    "growth": wt_growth
})

# B.2 – Anaerobic (no oxygen uptake)
with model:
    ex_o2 = get_rxn(model, "EX_o2_e")
    set_bound(ex_o2, lb=0.0)  # no O2 uptake
    sol_ana = model.optimize()
    medium_records.append({
        "condition": "anaerobic_no_O2",
        "EX_glc__D_e_lb": get_rxn(model, "EX_glc__D_e").lower_bound,
        "EX_o2_e_lb": ex_o2.lower_bound,
        "EX_ac_e_lb": get_rxn(model, "EX_ac_e").lower_bound
        if "EX_ac_e" in model.reactions else np.nan,
        "growth": sol_ana.objective_value
    })
    print(f"Anaerobic biomass: {sol_ana.objective_value:.4f}")

# B.3 – Glucose-limited (uptake -5)
with model:
    ex_glc = get_rxn(model, "EX_glc__D_e")
    set_bound(ex_glc, lb=-5.0)
    sol_glc5 = model.optimize()
    medium_records.append({
        "condition": "glucose_uptake_-5",
        "EX_glc__D_e_lb": ex_glc.lower_bound,
        "EX_o2_e_lb": get_rxn(model, "EX_o2_e").lower_bound,
        "EX_ac_e_lb": get_rxn(model, "EX_ac_e").lower_bound
        if "EX_ac_e" in model.reactions else np.nan,
        "growth": sol_glc5.objective_value
    })
    print(f"Biomass with glucose uptake -5: {sol_glc5.objective_value:.4f}")

# B.4 – Switch carbon source: no glucose, acetate as carbon source
with model:
    ex_glc = get_rxn(model, "EX_glc__D_e")
    ex_ac = get_rxn(model, "EX_ac_e")
    set_bound(ex_glc, lb=0.0)    # no glucose
    set_bound(ex_ac, lb=-10.0)   # allow acetate uptake
    sol_ac = model.optimize()
    medium_records.append({
        "condition": "acetate_only",
        "EX_glc__D_e_lb": ex_glc.lower_bound,
        "EX_o2_e_lb": get_rxn(model, "EX_o2_e").lower_bound,
        "EX_ac_e_lb": ex_ac.lower_bound,
        "growth": sol_ac.objective_value
    })
    print(f"Biomass on acetate: {sol_ac.objective_value:.4f}")

medium_df = pd.DataFrame(medium_records)
med_path = os.path.join(DATA_DIR, "day8_medium_conditions_growth.csv")
medium_df.to_csv(med_path, index=False)
print(f"Saved medium-condition growth table to: {med_path}\n")

# 4. Part C – Phenotypic curve: growth vs glucose uptake
print("[Part C] Scanning glucose uptake vs growth ...")

scan_vals = np.linspace(-20, -1, 20)  # from -20 to -1 mmol/gDW/hr
growth_vals = []

for g in scan_vals:
    with model:
        ex_glc = get_rxn(model, "EX_glc__D_e")
        set_bound(ex_glc, lb=g)
        sol = model.optimize()
        growth_vals.append(sol.objective_value)

curve_df = pd.DataFrame({
    "glucose_uptake": scan_vals,
    "growth": growth_vals
})
curve_path = os.path.join(DATA_DIR, "day8_growth_vs_glucose_curve.csv")
curve_df.to_csv(curve_path, index=False)
print(f"Saved growth vs glucose curve data to: {curve_path}")

plt.figure(figsize=(6, 4))
plt.plot(scan_vals, growth_vals, marker="o")
plt.xlabel("Glucose uptake (EX_glc__D_e lower bound)")
plt.ylabel("Growth rate (biomass)")
plt.title("Day 8: Growth vs glucose uptake")
plt.tight_layout()
fig_path = os.path.join(FIG_DIR, "day8_growth_vs_glucose.png")
plt.savefig(fig_path, dpi=300)
plt.close()
print(f"Saved phenotypic curve figure to: {fig_path}")

print("\nDay 8 – objective & medium analysis completed.")
