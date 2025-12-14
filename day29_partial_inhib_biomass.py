import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import math
import numpy as np
import pandas as pd
from cobra.io import read_sbml_model, save_json_model, load_json_model
from multiprocessing import Pool, cpu_count

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE, "../database/Human-GEM.xml")
REACTION_EXPR_FILE = os.path.join(BASE, "day17_reaction_expression_Tcells.csv")
OUT_DIR = BASE
os.makedirs(OUT_DIR, exist_ok=True)

# --- exchange IDs (same as your Day27/28) ---
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


def candidates_for_inhib(m, biomass_id):
    boundary = set(r.id for r in m.boundary)
    out = []
    for r in m.reactions:
        if r.id in boundary:
            continue
        if r.id == biomass_id:
            continue
        if r.lower_bound == 0 and r.upper_bound == 0:
            continue
        out.append(r.id)
    return out


def chunk_list(xs, n_chunks):
    if n_chunks <= 1:
        return [xs]
    k = math.ceil(len(xs) / n_chunks)
    return [xs[i:i+k] for i in range(0, len(xs), k)]


# ---------------- multiprocess worker state ----------------
_G = {}

def _init_worker(model_json_path, biomass_id, baseline_biomass, denom, alphas, sample, scenario):
    # Each process loads its own model copy once
    m = load_json_model(model_json_path)
    m.objective = biomass_id
    _G["m"] = m
    _G["biomass_id"] = biomass_id
    _G["baseline"] = baseline_biomass
    _G["denom"] = denom
    _G["alphas"] = alphas
    _G["sample"] = sample
    _G["scenario"] = scenario


def _scale_bounds(rxn, alpha):
    old_lb, old_ub = rxn.lower_bound, rxn.upper_bound
    rxn.lower_bound = min(old_lb * alpha, old_ub * alpha)
    rxn.upper_bound = max(old_lb * alpha, old_ub * alpha)
    return old_lb, old_ub


def _restore_bounds(rxn, old_lb, old_ub):
    rxn.lower_bound, rxn.upper_bound = old_lb, old_ub


def _worker_run(rxn_ids):
    m = _G["m"]
    baseline = _G["baseline"]
    denom = _G["denom"]
    alphas = _G["alphas"]
    sample = _G["sample"]
    scenario = _G["scenario"]

    rows = []
    for rid in rxn_ids:
        rxn = m.reactions.get_by_id(rid)
        for a in alphas:
            old_lb, old_ub = _scale_bounds(rxn, a)

            sol = m.optimize()
            bio = sol.objective_value if sol.status == "optimal" and sol.objective_value is not None else 0.0
            drop = (baseline - bio) / denom

            _restore_bounds(rxn, old_lb, old_ub)

            rows.append({
                "sample": sample,
                "scenario": scenario,
                "reaction_id": rid,
                "reaction_name": rxn.name,
                "subsystem": rxn.subsystem,
                "alpha": a,
                "baseline_biomass": float(baseline),
                "inhibited_biomass": float(bio),
                "rel_drop": float(drop),
                "is_near_essential": drop >= 0.9999,
                "is_strong_sensitive": (drop >= 0.5) and (drop < 0.9999),
            })
    return rows


def get_nproc(user_nproc=None):
    if user_nproc is not None and user_nproc > 0:
        return user_nproc
    # common HPC env vars
    for k in ["SLURM_CPUS_PER_TASK", "NSLOTS", "OMP_NUM_THREADS"]:
        v = os.environ.get(k, None)
        if v and str(v).isdigit() and int(v) > 0:
            return int(v)
    return max(1, cpu_count() // 2)


# ---------------- main ----------------
def main(nproc=None, alphas=(0.8, 0.5, 0.2)):
    print("Loading model:", MODEL_FILE)
    model = read_sbml_model(MODEL_FILE)

    rxn_expr = pd.read_csv(REACTION_EXPR_FILE, index_col=0)
    samples = list(rxn_expr.columns)
    print("Samples:", samples)

    BIOMASS_ID = get_objective_rxn_id(model)
    print("Biomass objective:", BIOMASS_ID)

    nproc = get_nproc(nproc)
    print("Using processes:", nproc)

    all_dfs = []
    summary = []

    for sample in samples:
        print("\n--- Day29(MP):", sample, "---")
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
        m_scaled.objective = BIOMASS_ID
        base = m_scaled.optimize()
        print("Baseline:", base.status, "biomass =", base.objective_value)

        if base.status != "optimal" or base.objective_value is None or base.objective_value <= 0:
            print("[warn] baseline failed, skip:", sample)
            continue

        baseline_biomass = float(base.objective_value)
        denom = max(abs(baseline_biomass), 1e-9)

        # candidates
        cand_ids = candidates_for_inhib(m_scaled, BIOMASS_ID)
        print("Candidates:", len(cand_ids))

        # save this sample's scaled model to json once
        model_json = os.path.join(OUT_DIR, f"tmp_day29_scaled_{sample}.json")
        save_json_model(m_scaled, model_json)

        # split work into chunks ~ nproc*4 (better load balancing)
        n_chunks = max(1, nproc * 4)
        chunks = chunk_list(cand_ids, n_chunks)

        with Pool(
            processes=nproc,
            initializer=_init_worker,
            initargs=(model_json, BIOMASS_ID, baseline_biomass, denom, alphas, sample, scenario)
        ) as pool:
            all_rows = []
            done = 0
            for rows in pool.imap_unordered(_worker_run, chunks, chunksize=1):
                all_rows.extend(rows)
                done += 1
                if done % 10 == 0 or done == len(chunks):
                    print(f"[{sample}] finished chunks: {done}/{len(chunks)}")

        # remove tmp json (optional)
        try:
            os.remove(model_json)
        except Exception:
            pass

        df = pd.DataFrame(all_rows)
        out_csv = os.path.join(OUT_DIR, f"day29_partialInhib_{sample}.csv")
        df.to_csv(out_csv, index=False)
        print("Saved:", out_csv)

        a05 = df[df["alpha"] == 0.5]
        summary.append({
            "sample": sample,
            "scenario": scenario,
            "alpha": 0.5,
            "n_records": len(a05),
            "n_near_essential": int(a05["is_near_essential"].sum()),
            "n_strong_sensitive": int(a05["is_strong_sensitive"].sum()),
        })

        all_dfs.append(df)

    if all_dfs:
        df_all = pd.concat(all_dfs, ignore_index=True)
        out_all = os.path.join(OUT_DIR, "day29_partialInhib_all_samples.csv")
        df_all.to_csv(out_all, index=False)
        print("\nSaved:", out_all)

        df_a05 = df_all[df_all["alpha"] == 0.5].copy()
        piv = df_a05.pivot_table(
            index=["reaction_id", "reaction_name", "subsystem"],
            columns="scenario",
            values="rel_drop",
            aggfunc="mean"
        ).reset_index()

        for col in ["Rich", "TME_edge", "TME_core"]:
            if col not in piv.columns:
                piv[col] = np.nan

        piv["S_edge_vs_rich"] = piv["TME_edge"] - piv["Rich"]
        piv["S_core_vs_rich"] = piv["TME_core"] - piv["Rich"]

        out_sel = os.path.join(OUT_DIR, "day29_selectivity_alpha0.5.csv")
        piv.to_csv(out_sel, index=False)
        print("Saved:", out_sel)

        print("\nTop Edge-selective:")
        print(piv.sort_values("S_edge_vs_rich", ascending=False)
                .head(10)[["reaction_id","S_edge_vs_rich","Rich","TME_edge","subsystem"]]
                .to_string(index=False))

        print("\nTop Core-selective:")
        print(piv.sort_values("S_core_vs_rich", ascending=False)
                .head(10)[["reaction_id","S_core_vs_rich","Rich","TME_core","subsystem"]]
                .to_string(index=False))

    if summary:
        df_sum = pd.DataFrame(summary)
        out_sum = os.path.join(OUT_DIR, "day29_summary.csv")
        df_sum.to_csv(out_sum, index=False)
        print("\nSaved:", out_sum)

    print("\nDone.")


if __name__ == "__main__":
    main(nproc=16)
