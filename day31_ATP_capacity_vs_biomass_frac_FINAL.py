import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from cobra.io import read_sbml_model
from cobra import Reaction

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE, "../database/Human-GEM.xml")
REACTION_EXPR_FILE = os.path.join(BASE, "day17_reaction_expression_Tcells.csv")
OUT_DIR = BASE
os.makedirs(OUT_DIR, exist_ok=True)

# exchange IDs (your current set)
EX_GLC   = "MAR09034"
EX_O2    = "MAR09048"
EX_GLN   = "MAR09063"
EX_LAC_L = "MAR09135"
EX_LAC_D = "MAR09136"

medium = {
    "Rich":     {EX_GLC:-10, EX_GLN:-10, EX_O2:-20, EX_LAC_L:0,   EX_LAC_D:0},
    "TME_edge": {EX_GLC:-3,  EX_GLN:-3,  EX_O2:-8,  EX_LAC_L:-5,  EX_LAC_D:-5},
    "TME_core": {EX_GLC:-1,  EX_GLN:-1,  EX_O2:-2,  EX_LAC_L:-10, EX_LAC_D:-10},
}

def apply_medium(m, tag: str):
    for rid, lb in medium[tag].items():
        if rid in m.reactions:
            m.reactions.get_by_id(rid).lower_bound = float(lb)

def infer_scenario(sample: str) -> str:
    s = sample.lower()
    if "blood" in s: return "Rich"
    if "edge" in s:  return "TME_edge"
    if "core" in s:  return "TME_core"
    return "Rich"

def get_objective_rxn(m):
    ids = [r.id for r in m.reactions if abs(r.objective_coefficient) > 0]
    if not ids:
        raise RuntimeError("No objective reaction found in model.")
    if len(ids) > 1:
        # still pick the first; but warn
        print("[warn] multiple objective reactions found, picking first:", ids[:5])
    return ids[0]

def ensure_dm_atp(mm, dm_id="DM_ATP_c_custom", ub=1e6):
    """
    Add a cytosolic ATP hydrolysis demand:
      ATP[c] + H2O[c] -> ADP[c] + Pi[c] + H+[c]
    """
    ATP_C = "MAM01371c"
    ADP_C = "MAM01285c"
    PI_C  = "MAM02751c"
    H2O_C = "MAM02040c"
    H_C   = "MAM02039c"

    # remove existing (to avoid duplicates / different stoich)
    if dm_id in mm.reactions:
        mm.remove_reactions([mm.reactions.get_by_id(dm_id)])

    atp = mm.metabolites.get_by_id(ATP_C)
    adp = mm.metabolites.get_by_id(ADP_C)
    pi  = mm.metabolites.get_by_id(PI_C)
    h2o = mm.metabolites.get_by_id(H2O_C)
    h   = mm.metabolites.get_by_id(H_C)

    dm = Reaction(dm_id)
    dm.name = "ATP hydrolysis demand (cytosol)"
    dm.lower_bound = 0.0
    dm.upper_bound = float(ub)
    dm.add_metabolites({atp: -1, h2o: -1, adp: 1, pi: 1, h: 1})

    mm.add_reactions([dm])
    return dm_id

def eflux_scale_skip_boundary_and_dm(m, expr_norm):
    """
    E-Flux scaling for internal reactions only.
    We skip boundary and DM/sink and biomass.
    """
    mm = m.copy()
    boundary_ids = set(r.id for r in mm.boundary)

    for rxn in mm.reactions:
        if rxn.id in boundary_ids:
            continue
        if rxn.id.startswith("DM_") or rxn.id.startswith("sink_"):
            continue
        if "biomass" in rxn.id.lower():
            continue

        scale = float(expr_norm.get(rxn.id, 1e-6))
        if scale < 1e-6:
            scale = 1e-6

        if rxn.upper_bound > 0:
            rxn.upper_bound *= scale
        if rxn.lower_bound < 0:
            rxn.lower_bound *= scale

    return mm

def lock_biomass_exact(m, biomass_id: str, target: float, tol: float = 1e-9):
    """
    Lock biomass flux to an exact value:
      lb = ub = target
    (add a tiny tolerance if you ever hit numerical infeasible issues)
    """
    bio = m.reactions.get_by_id(biomass_id)
    # exact lock
    bio.lower_bound = float(target)
    bio.upper_bound = float(target)
    # If numerical issues happen, uncomment:
    # bio.lower_bound = float(max(target - tol, 0.0))
    # bio.upper_bound = float(target + tol)

def main():
    model = read_sbml_model(MODEL_FILE)
    rxn_expr = pd.read_csv(REACTION_EXPR_FILE, index_col=0)

    biomass_id = get_objective_rxn(model)
    print("[biomass objective]", biomass_id)

    biomass_fracs = np.round(np.arange(0.1, 1.0, 0.1), 2).tolist() + [0.95, 0.99]
    biomass_fracs = sorted(set(biomass_fracs))

    rows = []

    for sample in rxn_expr.columns:
        scenario = infer_scenario(sample)

        expr = rxn_expr[sample]
        mx = float(expr.max())
        if mx <= 0:
            print("[warn] skip sample with nonpositive expr max:", sample)
            continue
        expr_norm = (expr / mx).to_dict()

        # Build env-specific model
        m0 = model.copy()
        apply_medium(m0, scenario)

        # Add DM ATP on the *unscaled* env model, then scale internals
        dm_id = ensure_dm_atp(m0, ub=1e6)

        m_scaled = eflux_scale_skip_boundary_and_dm(m0, expr_norm)

        # baseline biomass (under this scenario + eflux)
        m_scaled.objective = biomass_id
        base = m_scaled.optimize()
        if base.status != "optimal" or base.objective_value is None or base.objective_value <= 0:
            print("[warn] baseline biomass failed:", sample, scenario, base.status, base.objective_value)
            continue

        base_biomass = float(base.objective_value)
        print(f"\n[{scenario}] sample={sample} baseline_biomass={base_biomass:.6g}")

        # also sanity-check DM max without biomass lock (optional)
        # m_tmp = m_scaled.copy()
        # m_tmp.objective = dm_id
        # print("[debug] max DM without biomass lock:", m_tmp.optimize().objective_value)

        for frac in biomass_fracs:
            target = frac * base_biomass

            m = m_scaled.copy()

            lock_biomass_exact(m, biomass_id, target)

            # maximize ATP hydrolysis demand
            dm = m.reactions.get_by_id(dm_id)
            dm.lower_bound = 0.0
            dm.upper_bound = 1e6
            m.objective = dm_id

            sol = m.optimize()
            atp_cap = np.nan
            if sol.status == "optimal" and sol.objective_value is not None:
                atp_cap = float(sol.objective_value)

            rows.append({
                "sample": sample,
                "scenario": scenario,
                "biomass_frac": float(frac),
                "biomass_target": float(target),
                "baseline_biomass": float(base_biomass),
                "ATP_capacity": atp_cap,
                "status": sol.status
            })

    df = pd.DataFrame(rows)
    out = os.path.join(OUT_DIR, "day31_ATP_capacity_vs_biomass_frac_FINAL.csv")
    df.to_csv(out, index=False)
    print("\nSaved:", out)

    # pivot (mean over samples per scenario; you can change aggfunc)
    if len(df) > 0:
        piv = df.pivot_table(
            index="biomass_frac",
            columns="scenario",
            values="ATP_capacity",
            aggfunc="mean"
        ).reset_index()
        out2 = os.path.join(OUT_DIR, "day31_ATP_capacity_pivot_FINAL.csv")
        piv.to_csv(out2, index=False)
        print("Saved:", out2)
        print("\nPreview pivot:")
        print(piv.to_string(index=False))

if __name__ == "__main__":
    main()
