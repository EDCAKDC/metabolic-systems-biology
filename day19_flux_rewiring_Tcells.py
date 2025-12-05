import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cobra.io import read_sbml_model

# Day 19 – Flux rewiring analysis between E-Flux solutions

# Paths
BASE = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = os.path.join(BASE, "../database/Human-GEM.xml")
OUT_DIR = BASE

# Prefix for Day18 E-Flux output files
PREFIX = "day18_flux_"
os.makedirs(OUT_DIR, exist_ok=True)


# 1) Load model and collect reaction annotations

print(f"Loading model: {MODEL_FILE}")
model = read_sbml_model(MODEL_FILE)
print(f"Model loaded: {len(model.genes)} genes, {len(model.reactions)} reactions")

# Reaction metadata
rxn_ids = [rxn.id for rxn in model.reactions]
rxn_name_map = {rxn.id: (rxn.name or "") for rxn in model.reactions}
rxn_subsystem_map = {rxn.id: (rxn.subsystem or "NA") for rxn in model.reactions}


# 2) Read all Day18 E-Flux solutions and build a flux matrix

# Find all Day18 flux files
flux_files = [f for f in os.listdir(OUT_DIR) if f.startswith(PREFIX) and f.endswith(".csv")]
if not flux_files:
    raise FileNotFoundError(
        f"No '{PREFIX}*.csv' files found in {OUT_DIR}. "
        "Run Day18 first to generate E-Flux solutions."
    )

samples = []
flux_mat = pd.DataFrame(index=rxn_ids)

for f in sorted(flux_files):
    # Filename format: day18_flux_<Sample>.csv
    sample = f[len(PREFIX):-4]
    path = os.path.join(OUT_DIR, f)

    # Load flux vector
    s = pd.read_csv(path, index_col=0).iloc[:, 0]
    s.name = sample
    samples.append(sample)

    # Align reaction order and fill missing reactions with zero
    flux_mat[sample] = s.reindex(rxn_ids).fillna(0.0)

print("Samples found from Day18 flux files:", samples)

# Save the full flux matrix
flux_mat_out = os.path.join(OUT_DIR, "day19_flux_matrix_Tcells.csv")
flux_mat.to_csv(flux_mat_out)
print(f"Saved combined flux matrix: {flux_mat_out}")

# 3) Flux rewiring: compare each E-Flux condition vs Blood

REF_SAMPLE = "Blood"
if REF_SAMPLE not in samples:
    REF_SAMPLE = samples[0]
    print(f"'Blood' not found; using '{REF_SAMPLE}' as the reference sample.")

eps = 1e-9  # Prevent division by zero

for sample in samples:
    if sample == REF_SAMPLE:
        continue

    print(f"\n=== Comparing {sample} vs {REF_SAMPLE} ===")

    # Assemble comparison dataframe
    df = pd.DataFrame({
        "rxn_id": rxn_ids,
        f"flux_{REF_SAMPLE}": flux_mat[REF_SAMPLE].values,
        f"flux_{sample}": flux_mat[sample].values,
    })

    df["name"] = df["rxn_id"].map(rxn_name_map)
    df["subsystem"] = df["rxn_id"].map(rxn_subsystem_map)

    # Compute |flux| log2 fold-change to avoid sign issues
    ref_abs = df[f"flux_{REF_SAMPLE}"].abs() + eps
    samp_abs = df[f"flux_{sample}"].abs() + eps

    df["log2FC_abs_flux"] = np.log2(samp_abs / ref_abs)
    df["abs_log2FC"] = df["log2FC_abs_flux"].abs()
    df["mean_abs_flux"] = (ref_abs + samp_abs) / 2.0

    # Rank reactions by magnitude of change
    df_sorted = df.sort_values("abs_log2FC", ascending=False)

    out_csv = os.path.join(OUT_DIR, f"day19_flux_rewiring_{sample}_vs_{REF_SAMPLE}.csv")
    df_sorted.to_csv(out_csv, index=False)
    print(f"Saved reaction-level rewiring table: {out_csv}")

    # 3a) Scatter plot: flux_ref vs flux_sample
    plt.figure(figsize=(6, 5))
    x = df[f"flux_{REF_SAMPLE}"]
    y = df[f"flux_{sample}"]

    plt.scatter(x, y, s=5, alpha=0.4)

    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    plt.xlabel(f"{REF_SAMPLE} flux")
    plt.ylabel(f"{sample} flux")
    plt.title(f"Flux comparison: {sample} vs {REF_SAMPLE}")
    plt.tight_layout()

    scatter_out = os.path.join(OUT_DIR, f"day19_scatter_flux_{sample}_vs_{REF_SAMPLE}.png")
    plt.savefig(scatter_out, dpi=150)
    plt.close()
    print(f"Saved flux scatter plot: {scatter_out}")

    # 3b) Subsystem-level rewiring: mean |log2FC|
    subsys_change = (
        df.groupby("subsystem")["abs_log2FC"]
        .mean()
        .sort_values(ascending=False)
    )

    top_n = 20
    subsys_top = subsys_change.head(top_n)

    plt.figure(figsize=(14, 12))
    subsys_top[::-1].plot(kind="barh")

    plt.xlabel("Mean |log2FC| of absolute flux")
    plt.title(f"Subsystem-level flux rewiring: {sample} vs {REF_SAMPLE}")
    plt.tight_layout()

    bar_out = os.path.join(OUT_DIR, f"day19_subsystem_rewiring_{sample}_vs_{REF_SAMPLE}.png")
    plt.savefig(bar_out, dpi=150)
    plt.close()
    print(f"Saved subsystem rewiring plot: {bar_out}")

print("\nDay19 flux rewiring analysis completed.")
