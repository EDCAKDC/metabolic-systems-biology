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

# --- exchange IDs (same as Day27/28/29) ---
EX_GLC   = "MAR09034"
EX_O2    = "MAR09048"
EX_GLN   = "MAR09063"
EX_LAC_L = "MAR09135"
EX_LAC_D = "MAR09136"

EX_SER = None
EX_ARG = None
EX_TRP = None


def clean_medium(d):
    return {k: v for k, v in d.items() if k is not None}


medium_rich = clean_medium({
    EX_GLC: -10.0,
    EX_GLN: -10.0,
    EX_O2:  -20.0,
    EX_LAC_L: 0.0,
    EX_LAC_D: 0.0,
    EX_SER: -5.0 if EX_SER else None,
    EX_ARG: -5.0 if EX_ARG else None,
    EX_TRP: -2.0 if EX_TRP else None,
})

medium_edge = clean_medium({
    EX_GLC: -3.0,
    EX_GLN: -3.0,
    EX_O2:  -8.0,
    EX_LAC_L: -5.0,
    EX_LAC_D: -5.0,
    EX_SER: -1.5 if EX_SER else None,
    EX_ARG: -1.5 if EX_ARG else None,
    EX_TRP: -0.6 if EX_TRP else None,
})

medium_core = clean_medium({
    EX_GLC: -1.0,
    EX_GLN: -1.0,
    EX_O2:  -2.0,
    EX_LAC_L: -10.0,
    EX_LAC_D: -10.0,
    EX_SER: -0.5 if EX_SER else None,
    EX_ARG: -0.5 if EX_ARG else None,
    EX_TRP: -0.2 if EX_TRP else None,
})


def set_medium(m, medium_dict, tag):
    print("Applying medium:", tag)
    for rid, lb in medium_dict.items():
        if rid in m.reactions:
            m.reactions.get_by_id(rid).lower_bound = lb
        else:
            print("[warn] missing exchange:", rid)


def pick_scenario(sample_name):
    s = sample_name.lower()
    if "blood" in s:
        return "Rich"
    if "edge" in s:
        return "TME_edge"
    if "core" in s:
        return "TME_core"
    print("[warn] can't infer scenario for", sample_name, "-> Rich")
    return "Rich"


def get_objective_rxn_id(m):
    ids = [r.id for r in m.reactions if abs(r.objective_coefficient) > 0]
    return ids[0] if ids else None


def eflux_scale(m, expr_norm):
    mm = m.copy()
    for rxn in mm.reactions:
        if "biomass" in rxn.id.lower():
            continue
        scale = float(expr_norm.get(rxn.id, 1e-6))
        scale = max(scale, 1e-6)
        if rxn.upper_bound > 0:
            rxn.upper_bound *= scale
        if rxn.lower_bound < 0:
            rxn.lower_bound *= scale
    return mm


# ---------- functional objective helpers ----------

def find_metabolites_by_name(m, keyword_upper):
    hits = []
    for met in m.metabolites:
        nm = (met.name or "").upper()
        mid = (met.id or "").upper()
        if keyword_upper in nm or keyword_upper in mid:
            hits.append(met)
    return hits


def find_single_met_boundary_rxn(m, met):
    """
    Find boundary/demand-like reactions that contain only this metabolite.
    This is the most robust way to locate DM_* equivalents even if IDs are MARxxxxx.
    """
    cand = []
    for rxn in m.boundary:
        mets = list(rxn.metabolites.keys())
        if len(mets) == 1 and mets[0].id == met.id:
            cand.append(rxn)
    return cand


def pick_best_boundary_rxn(boundary_rxns):
    """
    If multiple exist, pick one with the widest upper bound.
    """
    if not boundary_rxns:
        return None
    boundary_rxns = sorted(boundary_rxns, key=lambda r: r.upper_bound, reverse=True)
    return boundary_rxns[0]


def find_demand_rxn_for_metabolite(m, met_keyword):
    """
    Try to find a demand-like boundary reaction for a metabolite keyword (ATP/AMP/IMP...).
    Returns: (met_id, rxn_id) or (None, None)
    """
    mets = find_metabolites_by_name(m, met_keyword.upper())
    for met in mets:
        bxs = find_single_met_boundary_rxn(m, met)
        rxn = pick_best_boundary_rxn(bxs)
        if rxn is not None:
            return met.id, rxn.id
    return None, None


def run_functional_capacity(m_scaled, biomass_id, biomass_frac, demand_rxn_id):
    """
    Constrain biomass >= frac * baseline and maximize demand reaction flux.
    Returns (baseline_biomass, functional_value, status)
    """
    # baseline
    m1 = m_scaled.copy()
    m1.objective = biomass_id
    base = m1.optimize()
    if base.status != "optimal" or base.objective_value is None:
        return 0.0, 0.0, "baseline_failed"

    baseline_biomass = float(base.objective_value)
    if baseline_biomass <= 0:
        return baseline_biomass, 0.0, "baseline_zero"

    # functional objective with biomass floor
    m2 = m_scaled.copy()
    m2.objective = demand_rxn_id

    # add biomass minimum
    bio = m2.reactions.get_by_id(biomass_id)
    old_lb = bio.lower_bound
    bio.lower_bound = max(old_lb, biomass_frac * baseline_biomass)

    sol = m2.optimize()
    func_val = float(sol.objective_value) if sol.status == "optimal" and sol.objective_value is not None else 0.0

    return baseline_biomass, func_val, sol.status


def main(biomass_frac=0.80):
    print("Loading model:", MODEL_FILE)
    model = read_sbml_model(MODEL_FILE)
    rxn_expr = pd.read_csv(REACTION_EXPR_FILE, index_col=0)
    samples = list(rxn_expr.columns)

    biomass_id = get_objective_rxn_id(model)
    print("Samples:", samples)
    print("Biomass objective:", biomass_id)
    print("Biomass floor fraction:", biomass_frac)

    # targets: ATP + nucleotides (we'll run the ones we can detect)
    target_keywords = ["ATP", "AMP", "IMP", "GMP", "UMP", "CTP"]

    rows_all = []

    for sample in samples:
        print("\n--- Day30:", sample, "---")
        expr_vec = rxn_expr[sample]
        mx = expr_vec.max()
        if mx <= 0:
            print("[warn] bad expression max, skip:", sample)
            continue
        expr_norm = (expr_vec / mx).to_dict()

        scenario = pick_scenario(sample)
        m_env = model.copy()
        if scenario == "Rich":
            set_medium(m_env, medium_rich, "Rich")
        elif scenario == "TME_edge":
            set_medium(m_env, medium_edge, "TME_edge")
        elif scenario == "TME_core":
            set_medium(m_env, medium_core, "TME_core")
        else:
            set_medium(m_env, medium_rich, "Rich")

        m_scaled = eflux_scale(m_env, expr_norm)

        # find demand rxns for targets in THIS scaled model
        targets_found = []
        for kw in target_keywords:
            met_id, rxn_id = find_demand_rxn_for_metabolite(m_scaled, kw)
            if rxn_id is not None:
                targets_found.append((kw, met_id, rxn_id))

        if not targets_found:
            print("[warn] no demand reactions detected (ATP/nucleotides).")
            print("      If Human-GEM has different naming, we can expand the search.")
            continue

        print("Functional targets found:")
        for kw, met_id, rxn_id in targets_found:
            rxn = m_scaled.reactions.get_by_id(rxn_id)
            print(f"  {kw}: met={met_id}  rxn={rxn_id}  name={rxn.name}")

        # run capacities
        per_sample = []
        for kw, met_id, rxn_id in targets_found:
            base_bio, func_val, status = run_functional_capacity(
                m_scaled=m_scaled,
                biomass_id=biomass_id,
                biomass_frac=biomass_frac,
                demand_rxn_id=rxn_id
            )
            per_sample.append({
                "sample": sample,
                "scenario": scenario,
                "biomass_frac": biomass_frac,
                "baseline_biomass": base_bio,
                "target": kw,
                "metabolite_id": met_id,
                "demand_rxn_id": rxn_id,
                "functional_value": func_val,
                "status": status
            })

        df_sample = pd.DataFrame(per_sample)
        out_csv = os.path.join(OUT_DIR, f"day30_functional_capacity_{sample}.csv")
        df_sample.to_csv(out_csv, index=False)
        print("Saved:", out_csv)

        rows_all.extend(per_sample)

    if rows_all:
        df_all = pd.DataFrame(rows_all)
        out_all = os.path.join(OUT_DIR, "day30_functional_capacity_all.csv")
        df_all.to_csv(out_all, index=False)
        print("\nSaved:", out_all)

        # quick pivot for readability
        piv = df_all.pivot_table(
            index=["target"],
            columns=["scenario"],
            values="functional_value",
            aggfunc="mean"
        ).reset_index()
        out_piv = os.path.join(OUT_DIR, "day30_functional_capacity_pivot.csv")
        piv.to_csv(out_piv, index=False)
        print("Saved:", out_piv)

        print("\nPreview (functional_value by scenario):")
        print(piv.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main(biomass_frac=0.80)
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

# --- exchange IDs (same as Day27/28/29) ---
EX_GLC   = "MAR09034"
EX_O2    = "MAR09048"
EX_GLN   = "MAR09063"
EX_LAC_L = "MAR09135"
EX_LAC_D = "MAR09136"

EX_SER = None
EX_ARG = None
EX_TRP = None


def clean_medium(d):
    return {k: v for k, v in d.items() if k is not None}


medium_rich = clean_medium({
    EX_GLC: -10.0,
    EX_GLN: -10.0,
    EX_O2:  -20.0,
    EX_LAC_L: 0.0,
    EX_LAC_D: 0.0,
    EX_SER: -5.0 if EX_SER else None,
    EX_ARG: -5.0 if EX_ARG else None,
    EX_TRP: -2.0 if EX_TRP else None,
})

medium_edge = clean_medium({
    EX_GLC: -3.0,
    EX_GLN: -3.0,
    EX_O2:  -8.0,
    EX_LAC_L: -5.0,
    EX_LAC_D: -5.0,
    EX_SER: -1.5 if EX_SER else None,
    EX_ARG: -1.5 if EX_ARG else None,
    EX_TRP: -0.6 if EX_TRP else None,
})

medium_core = clean_medium({
    EX_GLC: -1.0,
    EX_GLN: -1.0,
    EX_O2:  -2.0,
    EX_LAC_L: -10.0,
    EX_LAC_D: -10.0,
    EX_SER: -0.5 if EX_SER else None,
    EX_ARG: -0.5 if EX_ARG else None,
    EX_TRP: -0.2 if EX_TRP else None,
})


def set_medium(m, medium_dict, tag):
    print("Applying medium:", tag)
    for rid, lb in medium_dict.items():
        if rid in m.reactions:
            m.reactions.get_by_id(rid).lower_bound = lb
        else:
            print("[warn] missing exchange:", rid)


def pick_scenario(sample_name):
    s = sample_name.lower()
    if "blood" in s:
        return "Rich"
    if "edge" in s:
        return "TME_edge"
    if "core" in s:
        return "TME_core"
    print("[warn] can't infer scenario for", sample_name, "-> Rich")
    return "Rich"


def get_objective_rxn_id(m):
    ids = [r.id for r in m.reactions if abs(r.objective_coefficient) > 0]
    return ids[0] if ids else None


def eflux_scale(m, expr_norm):
    mm = m.copy()
    for rxn in mm.reactions:
        if "biomass" in rxn.id.lower():
            continue
        scale = float(expr_norm.get(rxn.id, 1e-6))
        scale = max(scale, 1e-6)
        if rxn.upper_bound > 0:
            rxn.upper_bound *= scale
        if rxn.lower_bound < 0:
            rxn.lower_bound *= scale
    return mm


# ---------- functional objective helpers ----------

def find_metabolites_by_name(m, keyword_upper):
    hits = []
    for met in m.metabolites:
        nm = (met.name or "").upper()
        mid = (met.id or "").upper()
        if keyword_upper in nm or keyword_upper in mid:
            hits.append(met)
    return hits


def find_single_met_boundary_rxn(m, met):
    """
    Find boundary/demand-like reactions that contain only this metabolite.
    This is the most robust way to locate DM_* equivalents even if IDs are MARxxxxx.
    """
    cand = []
    for rxn in m.boundary:
        mets = list(rxn.metabolites.keys())
        if len(mets) == 1 and mets[0].id == met.id:
            cand.append(rxn)
    return cand


def pick_best_boundary_rxn(boundary_rxns):
    """
    If multiple exist, pick one with the widest upper bound.
    """
    if not boundary_rxns:
        return None
    boundary_rxns = sorted(boundary_rxns, key=lambda r: r.upper_bound, reverse=True)
    return boundary_rxns[0]


def find_demand_rxn_for_metabolite(m, met_keyword):
    """
    Try to find a demand-like boundary reaction for a metabolite keyword (ATP/AMP/IMP...).
    Returns: (met_id, rxn_id) or (None, None)
    """
    mets = find_metabolites_by_name(m, met_keyword.upper())
    for met in mets:
        bxs = find_single_met_boundary_rxn(m, met)
        rxn = pick_best_boundary_rxn(bxs)
        if rxn is not None:
            return met.id, rxn.id
    return None, None


def run_functional_capacity(m_scaled, biomass_id, biomass_frac, demand_rxn_id):
    """
    Constrain biomass >= frac * baseline and maximize demand reaction flux.
    Returns (baseline_biomass, functional_value, status)
    """
    # baseline
    m1 = m_scaled.copy()
    m1.objective = biomass_id
    base = m1.optimize()
    if base.status != "optimal" or base.objective_value is None:
        return 0.0, 0.0, "baseline_failed"

    baseline_biomass = float(base.objective_value)
    if baseline_biomass <= 0:
        return baseline_biomass, 0.0, "baseline_zero"

    # functional objective with biomass floor
    m2 = m_scaled.copy()
    m2.objective = demand_rxn_id

    # add biomass minimum
    bio = m2.reactions.get_by_id(biomass_id)
    old_lb = bio.lower_bound
    bio.lower_bound = max(old_lb, biomass_frac * baseline_biomass)

    sol = m2.optimize()
    func_val = float(sol.objective_value) if sol.status == "optimal" and sol.objective_value is not None else 0.0

    return baseline_biomass, func_val, sol.status


def main(biomass_frac=0.80):
    print("Loading model:", MODEL_FILE)
    model = read_sbml_model(MODEL_FILE)
    rxn_expr = pd.read_csv(REACTION_EXPR_FILE, index_col=0)
    samples = list(rxn_expr.columns)

    biomass_id = get_objective_rxn_id(model)
    print("Samples:", samples)
    print("Biomass objective:", biomass_id)
    print("Biomass floor fraction:", biomass_frac)

    # targets: ATP + nucleotides (we'll run the ones we can detect)
    target_keywords = ["ATP", "AMP", "IMP", "GMP", "UMP", "CTP"]

    rows_all = []

    for sample in samples:
        print("\n--- Day30:", sample, "---")
        expr_vec = rxn_expr[sample]
        mx = expr_vec.max()
        if mx <= 0:
            print("[warn] bad expression max, skip:", sample)
            continue
        expr_norm = (expr_vec / mx).to_dict()

        scenario = pick_scenario(sample)
        m_env = model.copy()
        if scenario == "Rich":
            set_medium(m_env, medium_rich, "Rich")
        elif scenario == "TME_edge":
            set_medium(m_env, medium_edge, "TME_edge")
        elif scenario == "TME_core":
            set_medium(m_env, medium_core, "TME_core")
        else:
            set_medium(m_env, medium_rich, "Rich")

        m_scaled = eflux_scale(m_env, expr_norm)

        # find demand rxns for targets in THIS scaled model
        targets_found = []
        for kw in target_keywords:
            met_id, rxn_id = find_demand_rxn_for_metabolite(m_scaled, kw)
            if rxn_id is not None:
                targets_found.append((kw, met_id, rxn_id))

        if not targets_found:
            print("[warn] no demand reactions detected (ATP/nucleotides).")
            print("      If Human-GEM has different naming, we can expand the search.")
            continue

        print("Functional targets found:")
        for kw, met_id, rxn_id in targets_found:
            rxn = m_scaled.reactions.get_by_id(rxn_id)
            print(f"  {kw}: met={met_id}  rxn={rxn_id}  name={rxn.name}")

        # run capacities
        per_sample = []
        for kw, met_id, rxn_id in targets_found:
            base_bio, func_val, status = run_functional_capacity(
                m_scaled=m_scaled,
                biomass_id=biomass_id,
                biomass_frac=biomass_frac,
                demand_rxn_id=rxn_id
            )
            per_sample.append({
                "sample": sample,
                "scenario": scenario,
                "biomass_frac": biomass_frac,
                "baseline_biomass": base_bio,
                "target": kw,
                "metabolite_id": met_id,
                "demand_rxn_id": rxn_id,
                "functional_value": func_val,
                "status": status
            })

        df_sample = pd.DataFrame(per_sample)
        out_csv = os.path.join(OUT_DIR, f"day30_functional_capacity_{sample}.csv")
        df_sample.to_csv(out_csv, index=False)
        print("Saved:", out_csv)

        rows_all.extend(per_sample)

    if rows_all:
        df_all = pd.DataFrame(rows_all)
        out_all = os.path.join(OUT_DIR, "day30_functional_capacity_all.csv")
        df_all.to_csv(out_all, index=False)
        print("\nSaved:", out_all)

        # quick pivot for readability
        piv = df_all.pivot_table(
            index=["target"],
            columns=["scenario"],
            values="functional_value",
            aggfunc="mean"
        ).reset_index()
        out_piv = os.path.join(OUT_DIR, "day30_functional_capacity_pivot.csv")
        piv.to_csv(out_piv, index=False)
        print("Saved:", out_piv)

        print("\nPreview (functional_value by scenario):")
        print(piv.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main(biomass_frac=0.80)

