import os
import cobra
from cobra.io import load_model
from cobra.sampling import sample
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Output folders
DATA_DIR = "data"
FIG_DIR = "figures"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# Part 1 – Load model and run a reference FBA
print("Loading model ...")
model = load_model("textbook")

print("Running reference FBA (wild-type) ...")
solution = model.optimize()
wt_obj = solution.objective_value
print(f"WT objective (biomass) = {wt_obj:.4f}")
# Part 2 – Randomized flux sampling
# Number of samples from the feasible flux space
N_SAMPLES = 1000
RANDOM_SEED = 123

print(f"\nSampling {N_SAMPLES} feasible flux distributions ...")
# ACHR is the default method in cobra.sampling.sample
samples = sample(model, N_SAMPLES, seed=RANDOM_SEED)

print("Sampling finished.")
print("Sampled flux matrix shape:", samples.shape)  # (n_samples, n_reactions)

# Save the sampled flux table
out_csv = os.path.join(DATA_DIR, "day9_flux_samples.csv")
samples.to_csv(out_csv)
print(f"Saved flux samples to: {out_csv}")

# Part 3 – Visualize flux distributions for selected reactions
# Choose a few reactions that are biologically interesting in the core model.
# You can adjust this list depending on the model content.
candidate_rxns = [
    "PFK",   # phosphofructokinase
    "PYK",   # pyruvate kinase
    "ATPM",  # non-growth associated maintenance
    "BIOMASS_Ecoli_core_w_GAM"  # biomass reaction in the textbook model
]

for rxn_id in candidate_rxns:
    if rxn_id not in samples.columns:
        print(
            f"[Warning] Reaction '{rxn_id}' not found in sampled table. Skipping.")
        continue

    values = samples[rxn_id].values

    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=40)
    plt.xlabel("Flux value")
    plt.ylabel("Count")
    plt.title(f"Flux distribution from sampling: {rxn_id}")
    plt.tight_layout()

    fig_path = os.path.join(FIG_DIR, f"day9_flux_hist_{rxn_id}.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved histogram for {rxn_id} to: {fig_path}")

# Part 4 – Project flux space with PCA
print("\nRunning PCA on sampled flux space ...")

X = samples.values  # shape: (n_samples, n_reactions)

# Standard PCA to 2 components
pca = PCA(n_components=2, random_state=RANDOM_SEED)
X_pca = pca.fit_transform(X)

explained = pca.explained_variance_ratio_
print("Explained variance by PC1 and PC2:", explained)

# Scatter plot in PC1–PC2 space
plt.figure(figsize=(5, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], s=5)
plt.xlabel(f"PC1 ({explained[0] * 100:.1f}% var)")
plt.ylabel(f"PC2 ({explained[1] * 100:.1f}% var)")
plt.title("Flux space projection (PCA of sampled flux distributions)")
plt.tight_layout()

pca_fig = os.path.join(FIG_DIR, "day9_flux_space_pca.png")
plt.savefig(pca_fig, dpi=300)
plt.close()
print(f"Saved PCA projection figure to: {pca_fig}")

# Part 5 – Short textual summary
print("\nSummary:")
print(f"- WT biomass (objective): {wt_obj:.4f}")
print(f"- Number of sampled flux distributions: {N_SAMPLES}")
print(f"- Sampled flux table saved to: {out_csv}")
print(f"- PCA projection figure saved to: {pca_fig}")
print("Day 9 completed: flux space exploration via randomized sampling.")
