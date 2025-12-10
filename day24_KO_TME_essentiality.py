import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
import numpy as np
import pandas as pd
from cobra.io import read_sbml_model

# ------------------------------------------------------------
# Day 24 – Reaction knockout (KO) under TME constraints
#
# Goal:
#   For each T-cell condition (Blood / TumorEdge / TumorCore),
#   1) Build a TME-constrained, expression-scaled E-Flux model
#      (same logic as Day 22).
#   2) Run single-reaction knockout (KO) screening.
#   3) Classify reactions as essential / non-essential under TME.
#
# Output (per sample):
#   - day24_KO_TME_{sample}.csv
#       columns:
#           reaction_id
#           reaction_name
#           biomass_KO
#           is_essential_TME
#           baseline_biomass
#           baseline_flux
#
#   Essentiality threshold:
#       biomass_KO < 0.01 * baseline_biomass  → essential
# ------------------------------------------------------------

# ---------- Paths ----------
BASE = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = os.path.join(BASE, "../database/Human-GEM.xml")
REACTION_EXPR_FILE = os.path.join(BASE, "day17_reaction_expression_Tcells.csv")

OUT_DIR = BASE
os.makedirs(OUT_DIR, exist_ok=True)

# Sample names (as in your Day17 reaction-expression columns)
SAMPLES = ["Blood", "TumorEdge", "TumorCore"]

# ---------- Correct exchange IDs for Human-GEM in your model ----------
# (From your search output)
GLC_EX_ID = "MAR09034"   # Exchange of glucose
O2_EX_ID  = "MAR09048"   # Exchange of O2
GLN_EX_ID = "MAR09063"   # Exchange of glutamine
LAC_EX_ID = "MAR09135"   # Exchange of L-lactate


# ---------- Helper functions ----------

def set_environment_TME(m, scenario):
    """
    Apply TME-like nutrient bounds on a model copy.

    scenario in {"Blood", "Edge", "Core"}:
      - Blood: nutrient-replete (rich-ish)
      - Edge:  moderate limitation
      - Core:  severe limitation
    """
    def safe_set_lb(rxn_id, lb):
        if rxn_id in m.reactions:
            m.reactions.get_by_id(rxn_id).lower_bound = lb
        else:
            print(f"[Warning] Exchange '{rxn_id}' not found in model; skipped.")

    # Values in mmol/gDW/h (you can tune these)
    if scenario == "Blood":
        glc_lb = -10.0
        o2_lb  = -20.0
        gln_lb = -10.0
        lactate_lb = -10.0   # allow some lactate exchange
    elif scenario == "Edge":
        glc_lb = -3.0
        o2_lb  = -8.0
        gln_lb = -3.0
        lactate_lb = 0.0     # no lactate uptake
    elif scenario == "Core":
        glc_lb = -1.0
        o2_lb  = -2.0
        gln_lb = -1.0
        lactate_lb = 0.0     # no lactate uptake (accumulation)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    print(f"Setting TME environment for {scenario}: "
          f"glc={glc_lb}, o2={o2_lb}, gln={gln_lb}, lac_lb={lactate_lb}")

    safe_set_lb(GLC_EX_ID, glc_lb)
    safe_set_lb(O2_EX_ID,  o2_lb)
    safe_set_lb(GLN_EX_ID, gln_lb)
    safe_set_lb(LAC_EX_ID, lactate_lb)

    return m


def apply_eflux(model, expr_norm):
    """
    Apply E-Flux scaling to a model based on normalized
    reaction-level expression (0–1).

    expr_norm: pandas.Series
        index = reaction IDs, values = normalized expression.
    """
    m = model.copy()
    for rxn in m.reactions:
        # Do not scale biomass reactions
        if "biomass" in rxn.id.lower():
            continue

        if rxn.id in expr_norm.index:
            scale = float(expr_norm[rxn.id])
        else:
            scale = 1e-6
        scale = max(scale, 1e-6)

        if rxn.upper_bound > 0:
            rxn.upper_bound *= scale
        if rxn.lower_bound < 0:
            rxn.lower_bound *= scale

    return m


def infer_scenario_from_sample(sample_name):
    """
    Map sample name to TME scenario.
    Assumes names like "Blood", "TumorCore", "TumorEdge".
    """
    s = sample_name.lower()
    if "blood" in s:
        return "Blood"
    elif "core" in s:
        return "Core"
    elif "edge" in s:
        return "Edge"
    else:
        print(f"[Warning] Cannot infer scenario from sample '{sample_name}', using 'Blood'.")
        return "Blood"


# ---------- Load model and reaction expression ----------

print(f"Loading model: {MODEL_FILE}")
model = read_sbml_model(MODEL_FILE)
print(f"Model loaded: {len(model.genes)} genes, {len(model.reactions)} reactions")

print(f"Loading reaction-level expression: {REACTION_EXPR_FILE}")
rxn_expr = pd.read_csv(REACTION_EXPR_FILE, index_col=0)
print("Samples found in expression table:", list(rxn_expr.columns))


# ---------- KO screening under TME for each sample ----------

for sample in SAMPLES:
    if sample not in rxn_expr.columns:
        print(f"[Warning] Sample '{sample}' not found in expression table; skipping.")
        continue

    print("\n" + "=" * 70)
    print(f"Day 24 – TME KO screening for sample: {sample}")
    print("=" * 70)

    scenario = infer_scenario_from_sample(sample)
    expr_vec = rxn_expr[sample]

    max_val = expr_vec.max()
    if max_val <= 0:
        print(f"[Warning] Sample {sample} has non-positive max expression; skipping.")
        continue

    expr_norm = expr_vec / max_val
    print(f"Expression stats for {sample}: mean={expr_vec.mean():.3f}, std={expr_vec.std():.3f}")

    # 1) Build TME + E-Flux model
    print("Building TME-constrained E-Flux model...")
    model_env = set_environment_TME(model.copy(), scenario)
    model_scaled = apply_eflux(model_env, expr_norm)

    # 2) Baseline FBA (no KO)
    print("Running baseline FBA under TME constraints...")
    sol0 = model_scaled.optimize()
    if sol0.status != "optimal":
        print(f"[Warning] Baseline FBA not optimal for {sample} ({sol0.status}); skipping.")
        continue

    baseline_biomass = float(sol0.objective_value)
    baseline_flux = sol0.fluxes
    print(f"Baseline biomass: {baseline_biomass:.4f}")

    # Essentiality threshold: 1% of baseline
    thr = 0.01 * baseline_biomass
    print(f"Essentiality threshold: biomass_KO < {thr:.6g}")

    # 3) Decide which reactions to test
    #    Here we only KO reactions that carry flux in baseline solution
    active_rxn_ids = [
        rxn_id for rxn_id, v in baseline_flux.items()
        if abs(v) > 1e-9
    ]
    print(f"Number of reactions with non-zero baseline flux: {len(active_rxn_ids)}")

    # 4) KO loop
    records = []
    n_total = len(active_rxn_ids)
    t_start = time.time()

    for i, rxn_id in enumerate(active_rxn_ids, start=1):
        rxn = model_scaled.reactions.get_by_id(rxn_id)

        # Skip biomass reactions (we don't knock them out)
        if "biomass" in rxn.id.lower():
            continue

        # Use cobra's context manager so changes are temporary
        with model_scaled as m:
            # Knock out reaction: set both bounds to 0
            r = m.reactions.get_by_id(rxn_id)
            old_lb, old_ub = r.lower_bound, r.upper_bound
            r.lower_bound = 0.0
            r.upper_bound = 0.0

            sol = m.optimize()
            if sol.status == "optimal":
                biomass_ko = float(sol.objective_value)
            else:
                biomass_ko = 0.0

        is_essential = biomass_ko < thr

        records.append({
            "reaction_id": rxn_id,
            "reaction_name": rxn.name,
            "biomass_KO": biomass_ko,
            "is_essential_TME": bool(is_essential),
            "baseline_biomass": baseline_biomass,
            "baseline_flux": float(baseline_flux[rxn_id]),
        })

        if i % 200 == 0 or i == n_total:
            elapsed = time.time() - t_start
            print(f"  KO {i}/{n_total} reactions done "
                  f"({elapsed/60:.1f} min elapsed)")

    # 5) Save results
    df_ko = pd.DataFrame(records)
    out_path = os.path.join(OUT_DIR, f"day24_KO_TME_{sample}.csv")
    df_ko.to_csv(out_path, index=False)
    print(f"\nSaved KO results for {sample} to: {out_path}")

    # Simple summary
    n_essential = df_ko["is_essential_TME"].sum()
    print(f"Number of essential reactions under TME ({sample}): {n_essential}")
    print("Top 10 essential reactions (sorted by baseline flux):")
    print(df_ko[df_ko["is_essential_TME"]]
          .sort_values("baseline_flux", key=lambda s: np.abs(s), ascending=False)
          .head(10)[["reaction_id", "reaction_name", "baseline_flux", "biomass_KO"]]
          .to_string(index=False))

print("\nDay 24 TME KO screening completed.")

