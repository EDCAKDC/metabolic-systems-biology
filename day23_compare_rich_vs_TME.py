import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cobra.io import read_sbml_model

# ------------------------------------------------------------
# Day 23 – Compare rich vs TME E-Flux (TME effect)
#
# Goal:
#   For each T-cell condition (Blood / TumorEdge / TumorCore):
#
#   1) Compare reaction-level fluxes:
#        - rich (Day18) vs TME (Day22)
#        - compute delta and log2FC
#   2) Aggregate to subsystem level:
#        - mean |flux| in rich vs TME
#        - subsystem log2FC
#   3) Compare biomass (growth) between rich and TME.
#
#   This answers: "How does adding TME constraints
#   change metabolism compared to the rich medium?"
# ------------------------------------------------------------

# ---------- Paths ----------
BASE = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = os.path.join(BASE, "../database/Human-GEM.xml")

# Flux files
RICH_FLUX_TEMPLATE = os.path.join(BASE, "day18_flux_{sample}.csv")
TME_FLUX_TEMPLATE  = os.path.join(BASE, "day22_flux_TME_{sample}.csv")

# Biomass summary files (from Day18, Day22)
RICH_BIOMASS_FILE = os.path.join(BASE, "day18_biomass_summary_Tcells.csv")
TME_BIOMASS_FILE  = os.path.join(BASE, "day22_TME_biomass_summary_Tcells.csv")

OUT_DIR = BASE
os.makedirs(OUT_DIR, exist_ok=True)

# Sample names (adjust if your naming differs)
SAMPLES = ["Blood", "TumorEdge", "TumorCore"]

# ---------- Helper functions ----------

def load_flux(sample: str, env: str) -> pd.DataFrame:
    """
    Load flux table for a given sample and environment.

    env:
        - "rich" -> Day18 result
        - "TME"  -> Day22 result

    Returns DataFrame with columns: reaction, flux
    """
    if env == "rich":
        path = RICH_FLUX_TEMPLATE.format(sample=sample)
    elif env == "TME":
        path = TME_FLUX_TEMPLATE.format(sample=sample)
    else:
        raise ValueError(f"Unknown env: {env}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Flux file not found: {path}")

    df = pd.read_csv(path)

    # Try to standardize column names
    df = df.rename(columns={
        "reaction_id": "reaction",
        "rxn_id": "reaction",
        "v": "flux"
    })

    # If saved from Series.to_csv() (index,label) format
    if "flux" not in df.columns and df.shape[1] == 2:
        df.columns = ["reaction", "flux"]

    if "reaction" not in df.columns or "flux" not in df.columns:
        raise ValueError(f"File {path} must contain 'reaction' and 'flux' columns.")

    return df[["reaction", "flux"]]


def add_subsystem(model, df: pd.DataFrame) -> pd.DataFrame:
    """
    Add subsystem annotation to a reaction-level DataFrame.
    """
    rxn2sub = {
        rxn.id: (rxn.subsystem if rxn.subsystem else "Unknown")
        for rxn in model.reactions
    }
    df["subsystem"] = df["reaction"].map(lambda r: rxn2sub.get(r, "Unknown"))
    return df


def aggregate_subsystem(flux_cmp: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate reaction-level comparison to subsystem-level.

    Input columns:
        - subsystem
        - flux_rich
        - flux_TME
    """
    grp = flux_cmp.groupby("subsystem")

    agg = pd.DataFrame({
        "mean_abs_flux_rich": grp["flux_rich"].apply(lambda x: np.mean(np.abs(x))),
        "mean_abs_flux_TME":  grp["flux_TME"].apply(lambda x: np.mean(np.abs(x)))
    }).reset_index()

    agg["delta_mean_abs_flux"] = agg["mean_abs_flux_TME"] - agg["mean_abs_flux_rich"]

    eps = 1e-9
    agg["log2FC_TME_vs_rich"] = np.log2(
        (agg["mean_abs_flux_TME"] + eps) /
        (agg["mean_abs_flux_rich"] + eps)
    )

    # Sort by |log2FC|
    agg = agg.sort_values("log2FC_TME_vs_rich", key=lambda s: np.abs(s), ascending=False)
    return agg


def compare_biomass():
    """
    Compare biomass between rich (Day18) and TME (Day22).
    """
    if not (os.path.exists(RICH_BIOMASS_FILE) and os.path.exists(TME_BIOMASS_FILE)):
        print("[Warning] Biomass summary files not found; skip biomass comparison.")
        return

    rich = pd.read_csv(RICH_BIOMASS_FILE)
    tme  = pd.read_csv(TME_BIOMASS_FILE)

    # Keep only columns we need
    rich = rich[["sample", "biomass"]].rename(columns={"biomass": "biomass_rich"})
    tme  = tme[["sample", "biomass"]].rename(columns={"biomass": "biomass_TME"})

    merged = rich.merge(tme, on="sample", how="inner")

    merged["delta_biomass"] = merged["biomass_TME"] - merged["biomass_rich"]

    eps = 1e-9
    merged["log2FC_TME_vs_rich"] = np.log2(
        (merged["biomass_TME"] + eps) /
        (merged["biomass_rich"] + eps)
    )

    out_csv = os.path.join(OUT_DIR, "day23_biomass_compare_Tcells.csv")
    merged.to_csv(out_csv, index=False)
    print(f"\n[Biomass] Saved comparison table to: {out_csv}")
    print("[Biomass] Summary:")
    print(merged.to_string(index=False))


# ---------- Main analysis ----------

print(f"Loading model: {MODEL_FILE}")
model = read_sbml_model(MODEL_FILE)
print(f"Model loaded: {len(model.genes)} genes, {len(model.reactions)} reactions")

for sample in SAMPLES:
    print("\n" + "=" * 70)
    print(f"Sample: {sample} – rich vs TME flux comparison")
    print("=" * 70)

    # 1) Load rich & TME flux
    df_rich = load_flux(sample, "rich")
    df_tme  = load_flux(sample, "TME")

    # 2) Merge and compute differences
    merged = df_rich.merge(df_tme, on="reaction", suffixes=("_rich", "_TME"))

    merged["delta_flux"] = merged["flux_TME"] - merged["flux_rich"]

    eps = 1e-9
    merged["log2FC_TME_vs_rich"] = np.log2(
        (np.abs(merged["flux_TME"]) + eps) /
        (np.abs(merged["flux_rich"]) + eps)
    )

    # Add subsystem annotation
    merged = add_subsystem(model, merged)

    # Save full reaction-level comparison
    out_flux_csv = os.path.join(OUT_DIR, f"day23_flux_compare_{sample}.csv")
    merged.to_csv(out_flux_csv, index=False)
    print(f"Saved reaction-level comparison: {out_flux_csv}")

    # Quick summary
    corr = np.corrcoef(merged["flux_rich"], merged["flux_TME"])[0, 1]
    print(f"Pearson correlation (rich vs TME flux): {corr:.3f}")
    print("Top 5 reactions with largest |log2FC|:")
    print(
        merged.reindex(
            np.argsort(np.abs(merged["log2FC_TME_vs_rich"]))[::-1]
        ).head(5)[["reaction", "flux_rich", "flux_TME", "log2FC_TME_vs_rich"]]
    )

    # 3) Scatter plot: rich vs TME flux
    plt.figure(figsize=(5, 5))
    plt.scatter(merged["flux_rich"], merged["flux_TME"], alpha=0.4)
    plt.xlabel("Flux in rich medium (Day18)")
    plt.ylabel("Flux in TME (Day22)")
    plt.title(f"Day23 – Flux: rich vs TME ({sample})")
    plt.axhline(0, linewidth=0.5)
    plt.axvline(0, linewidth=0.5)
    plt.tight_layout()
    scatter_path = os.path.join(OUT_DIR, f"day23_flux_scatter_{sample}.png")
    plt.savefig(scatter_path, dpi=150)
    plt.close()
    print(f"Saved scatter plot: {scatter_path}")

    # 4) Histogram of log2FC
    plt.figure(figsize=(6, 4))
    plt.hist(merged["log2FC_TME_vs_rich"], bins=60)
    plt.xlabel("log2(|flux_TME| / |flux_rich|)")
    plt.ylabel("Reaction count")
    plt.title(f"Day23 – Flux log2FC distribution ({sample})")
    plt.tight_layout()
    hist_path = os.path.join(OUT_DIR, f"day23_flux_hist_log2FC_{sample}.png")
    plt.savefig(hist_path, dpi=150)
    plt.close()
    print(f"Saved log2FC histogram: {hist_path}")

    # 5) Subsystem-level comparison
    sub_agg = aggregate_subsystem(merged)
    out_sub_csv = os.path.join(OUT_DIR, f"day23_subsystem_compare_{sample}.csv")
    sub_agg.to_csv(out_sub_csv, index=False)
    print(f"Saved subsystem-level comparison: {out_sub_csv}")

    # Barplot: top 15 subsystems by |log2FC|
    top_n = 15
    top_sub = sub_agg.head(top_n)

    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(top_sub))
    plt.barh(y_pos, top_sub["log2FC_TME_vs_rich"])
    plt.yticks(y_pos, top_sub["subsystem"])
    plt.xlabel("log2( mean |flux| TME / rich )")
    plt.title(f"Day23 – Top {top_n} subsystems (TME vs rich) – {sample}")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    bar_path = os.path.join(OUT_DIR, f"day23_subsystem_bar_{sample}.png")
    plt.savefig(bar_path, dpi=150)
    plt.close()
    print(f"Saved subsystem bar plot: {bar_path}")

# 6) Biomass comparison (rich vs TME)
compare_biomass()

print("\nDay23 rich vs TME comparison completed.")

