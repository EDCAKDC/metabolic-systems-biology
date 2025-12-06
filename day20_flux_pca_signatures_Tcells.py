import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cobra.io import read_sbml_model
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Day 20 – PCA and subsystem-level flux signatures
# Goal:
#   (1) Perform PCA on E-Flux solutions (Blood/Core/Edge).
#   (2) Build subsystem-level metabolic signatures
#       (mean absolute flux per subsystem per sample).
#   (3) Visualize sample separation and top rewired subsystems.

# Paths
BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE, "../database/Human-GEM.xml")
OUT_DIR = BASE
os.makedirs(OUT_DIR, exist_ok=True)

# Filenames from previous days
FLUX_MATRIX_FILE = os.path.join(OUT_DIR, "day19_flux_matrix_Tcells.csv")
DAY18_PREFIX = "day18_flux_"


# 1) Load flux matrix (or reconstruct from Day18 outputs)

if os.path.exists(FLUX_MATRIX_FILE):
    print(f"Loading flux matrix from: {FLUX_MATRIX_FILE}")
    flux_mat = pd.read_csv(FLUX_MATRIX_FILE, index_col=0)
else:
    # Fallback: reconstruct the matrix from Day18 flux CSV files
    print("day19_flux_matrix_Tcells.csv not found.")
    print("Reconstructing flux matrix from Day18 flux files...")

    flux_files = [
        f for f in os.listdir(OUT_DIR)
        if f.startswith(DAY18_PREFIX) and f.endswith(".csv")
    ]
    if not flux_files:
        raise FileNotFoundError(
            f"No '{DAY18_PREFIX}*.csv' files found in {OUT_DIR}. "
            "Run Day18/Day19 before Day20."
        )

    # Load model to get reaction order
    print(f"Loading model: {MODEL_FILE}")
    model = read_sbml_model(MODEL_FILE)
    rxn_ids = [rxn.id for rxn in model.reactions]

    flux_mat = pd.DataFrame(index=rxn_ids)
    samples = []

    for f in sorted(flux_files):
        sample = f[len(DAY18_PREFIX):-4]
        path = os.path.join(OUT_DIR, f)
        s = pd.read_csv(path, index_col=0).iloc[:, 0]
        s.name = sample
        samples.append(sample)
        flux_mat[sample] = s.reindex(rxn_ids).fillna(0.0)

    flux_mat.to_csv(FLUX_MATRIX_FILE)
    print(f"Saved reconstructed flux matrix: {FLUX_MATRIX_FILE}")

# Now we have: flux_mat (rows = reactions, columns = samples)
print("Flux matrix shape (reactions x samples):", flux_mat.shape)
samples = list(flux_mat.columns)
print("Samples:", samples)


# 2) Load model to get reaction annotations (again, in case)

print(f"Loading model for annotations: {MODEL_FILE}")
model = read_sbml_model(MODEL_FILE)
rxn_ids = [rxn.id for rxn in model.reactions]
rxn_subsystem_map = {rxn.id: (rxn.subsystem or "NA") for rxn in model.reactions}

# Ensure flux matrix is aligned with model reactions
flux_mat = flux_mat.reindex(index=rxn_ids).fillna(0.0)


# 3) PCA on whole-flux profiles (samples in flux space)
# We treat each sample as a point in reaction-flux space.
# Rows: samples, Columns: reactions
X = flux_mat.T.values

# Optionally standardize (mean=0, var=1) across reactions
scaler = StandardScaler(with_mean=True, with_std=True)
X_scaled = scaler.fit_transform(X)

# With N samples, PCA can have at most N-1 components.
n_components = min(3, X_scaled.shape[0])
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

explained = pca.explained_variance_ratio_ * 100
print("PCA explained variance (%):", explained)

# Save PCA coordinates
pca_df = pd.DataFrame(
    X_pca,
    index=samples,
    columns=[f"PC{i+1}" for i in range(n_components)]
)
pca_out = os.path.join(OUT_DIR, "day20_flux_PCA_coordinates.csv")
pca_df.to_csv(pca_out)
print(f"Saved PCA coordinates: {pca_out}")

# 2D PCA plot (PC1 vs PC2)
if n_components >= 2:
    plt.figure(figsize=(6, 5))
    for i, sample in enumerate(samples):
        plt.scatter(
            pca_df.loc[sample, "PC1"],
            pca_df.loc[sample, "PC2"],
            s=80
        )
        plt.text(
            pca_df.loc[sample, "PC1"],
            pca_df.loc[sample, "PC2"],
            sample,
            fontsize=10,
            ha="center",
            va="bottom"
        )

    plt.xlabel(f"PC1 ({explained[0]:.1f}% var)")
    plt.ylabel(f"PC2 ({explained[1]:.1f}% var)")
    plt.title("PCA of E-Flux solutions (flux space)")
    plt.tight_layout()

    pca_plot_out = os.path.join(OUT_DIR, "day20_flux_PCA_samples_PC1_PC2.png")
    plt.savefig(pca_plot_out, dpi=150)
    plt.close()
    print(f"Saved PCA plot: {pca_plot_out}")


# 4) Subsystem-level flux signatures

# Build a dataframe with reaction → subsystem
annot = pd.DataFrame({
    "rxn_id": rxn_ids,
    "subsystem": [rxn_subsystem_map[r] for r in rxn_ids]
})
annot.index = rxn_ids

# Compute mean absolute flux per subsystem per sample
subsys_signature = []

for subsys, rxn_idx in annot.groupby("subsystem").groups.items():
    # rxn_idx is an index of reaction IDs belonging to this subsystem
    sub_flux = flux_mat.loc[list(rxn_idx), :]  # reactions in this subsystem
    # Mean absolute flux across reactions for each sample
    mean_abs = sub_flux.abs().mean(axis=0)
    mean_abs.name = subsys
    subsys_signature.append(mean_abs)

subsys_sig_df = pd.DataFrame(subsys_signature)
subsys_sig_df.index.name = "subsystem"
subsys_sig_df = subsys_sig_df.sort_index()

sig_out = os.path.join(OUT_DIR, "day20_subsystem_flux_signatures_meanAbs.csv")
subsys_sig_df.to_csv(sig_out)
print(f"Saved subsystem-level flux signatures: {sig_out}")


# 5) Identify most variable subsystems and plot heatmap

# Variability of each subsystem across samples
subsys_var = subsys_sig_df.var(axis=1)
subsys_var = subsys_var.sort_values(ascending=False)

top_n = min(30, subsys_var.shape[0])  # up to 30 subsystems
top_subsystems = subsys_var.head(top_n).index

subsys_top_df = subsys_sig_df.loc[top_subsystems]

# Z-score across samples (per subsystem) for visualization
subsys_top_z = (subsys_top_df - subsys_top_df.mean(axis=1).values[:, None])
subsys_top_z = subsys_top_z.divide(
    subsys_top_df.std(axis=1).replace(0, np.nan).values[:, None]
)
subsys_top_z = subsys_top_z.fillna(0.0)

heatmap_out_csv = os.path.join(OUT_DIR, "day20_subsystem_flux_signatures_topVar_zscore.csv")
subsys_top_z.to_csv(heatmap_out_csv)
print(f"Saved z-scored top subsystem signatures: {heatmap_out_csv}")

# Heatmap (subsystems x samples)
plt.figure(figsize=(6, 0.4 * top_n + 2))
im = plt.imshow(subsys_top_z.values, aspect="auto")

plt.colorbar(im, label="Z-scored mean |flux|")

plt.yticks(
    np.arange(top_n),
    subsys_top_z.index,
    fontsize=6
)
plt.xticks(
    np.arange(len(samples)),
    samples,
    rotation=45,
    ha="right"
)

plt.title("Top variable subsystems (mean |flux|, z-scored)")
plt.tight_layout()

heatmap_out = os.path.join(OUT_DIR, "day20_subsystem_flux_signatures_heatmap.png")
plt.savefig(heatmap_out, dpi=150)
plt.close()
print(f"Saved subsystem heatmap: {heatmap_out}")


print("\nDay20 PCA and subsystem signature analysis completed.")

