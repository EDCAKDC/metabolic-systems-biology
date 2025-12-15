import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from cobra.io import read_sbml_model

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE, "../database/Human-GEM.xml")
REACTION_EXPR_FILE = os.path.join(BASE, "day17_reaction_expression_Tcells.csv")
OUT_DIR = BASE
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- medium (same as before) ----------
EX_GLC   = "MAR09034"
EX_O2    = "MAR09048"
EX_GLN   = "MAR09063"
EX_LAC_L = "MAR09135"
EX_LAC_D = "MAR09136"

medium = {
    "Rich": {
        EX_GLC: -10, EX_GLN: -10, EX_O2: -20, EX_LAC_L: 0, EX_LAC_D: 0
    },
    "TME_edge": {
        EX_GLC: -3, EX_GLN: -3, EX_O2: -8, EX_LAC_L: -5, EX_LAC_D: -5
    },
    "TME_core": {
        EX_GLC: -1, EX_GLN: -1, EX_O2: -2, EX_LAC_L: -10, EX_LAC_D: -10
    }
}

def apply_medium(m, tag):
    for rid, lb in medium[tag].items():
        if rid in m.reactions:
            m.reactions.get_by_id(rid).lower_bound = lb

def infer_scenario(sample):
    s = sample.lower()
    if "blood" in s: return "Rich"
    if "edge" in s: return "TME_edge"
    if "core" in s: return "TME_core"
    return "Rich"

def eflux_scale(m, expr):
    mm = m.copy()
    for r in mm.reactions:
        if "biomass" in r.id.lower(): continue
        scale = max(expr.get(r.id, 1e-6), 1e-6)
        if r.upper_bound > 0: r.upper_bound *= scale
        if r.lower_bound < 0: r.lower_bound *= scale
    return mm

def get_objective_rxn(m):
    return [r.id for r in m.reactions if r.objective_coefficient != 0][0]

def find_ATP_demand(m):
    for rxn in m.boundary:
        mets = list(rxn.metabolites)
        if len(mets) == 1 and "ATP" in (mets[0].id + mets[0].name).upper():
            return rxn.id
    return None

# ---------- main ----------
def main():
    model = read_sbml_model(MODEL_FILE)
    rxn_expr = pd.read_csv(REACTION_EXPR_FILE, index_col=0)

    biomass_id = get_objective_rxn(model)
    atp_rxn = find_ATP_demand(model)

    if atp_rxn is None:
        raise RuntimeError("ATP demand reaction not found")

    biomass_fracs = [0.9, 0.8, 0.7, 0.6]
    rows = []

    for sample in rxn_expr.columns:
        scenario = infer_scenario(sample)
        expr = (rxn_expr[sample] / rxn_expr[sample].max()).to_dict()

        m0 = model.copy()
        apply_medium(m0, scenario)
        m_scaled = eflux_scale(m0, expr)

        # baseline biomass
        m_scaled.objective = biomass_id
        base = m_scaled.optimize()
        if base.status != "optimal":
            continue

        base_biomass = base.objective_value

        for frac in biomass_fracs:
            m = m_scaled.copy()
            bio = m.reactions.get_by_id(biomass_id)
            bio.lower_bound = frac * base_biomass

            m.objective = atp_rxn
            sol = m.optimize()

            rows.append({
                "sample": sample,
                "scenario": scenario,
                "biomass_frac": frac,
                "ATP_capacity": sol.objective_value if sol.status == "optimal" else 0.0
            })

    df = pd.DataFrame(rows)
    out = os.path.join(OUT_DIR, "day31_ATP_capacity_vs_biomass_frac.csv")
    df.to_csv(out, index=False)
    print("Saved:", out)

    piv = df.pivot_table(
        index="biomass_frac",
        columns="scenario",
        values="ATP_capacity",
        aggfunc="mean"
    ).reset_index()

    out2 = os.path.join(OUT_DIR, "day31_ATP_capacity_pivot.csv")
    piv.to_csv(out2, index=False)
    print("Saved:", out2)

    print("\nPreview:")
    print(piv.to_string(index=False))

if __name__ == "__main__":
    main()

