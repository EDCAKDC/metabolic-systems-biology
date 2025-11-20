import itertools
import cobra
from cobra.io import load_model
from cobra.flux_analysis import single_reaction_deletion
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load model and compute WT growth
model = load_model("textbook")
print("Loaded model:", model)

wt_solution = model.optimize()
wt_growth = wt_solution.objective_value
print("WT biomass (objective value):", wt_growth)

# Single-reaction KO robustness
ess_rxn = single_reaction_deletion(model)

# Add normalized growth ratio
ess_rxn["growth_ratio"] = ess_rxn["growth"] / wt_growth

# Save full table
ess_rxn.to_csv("day5_single_reaction_KO_robustness.csv")
print(ess_rxn.head())


# Plot robustness curve: sort reactions by growth_ratio ---
sorted_gr = ess_rxn["growth_ratio"].sort_values(
    ascending=True).reset_index(drop=True)

plt.figure(figsize=(6, 4))
plt.plot(sorted_gr.values, marker=".", linestyle="-")
plt.xlabel("Reaction knockout (sorted)")
plt.ylabel("Growth ratio (KO / WT)")
plt.title("Day 5: Single-reaction KO robustness curve")
plt.tight_layout()
plt.savefig("day5_single_KO_robustness_curve.png", dpi=300)
plt.close()

# Also a histogram if you like
plt.figure(figsize=(6, 4))
ess_rxn["growth_ratio"].hist(bins=30)
plt.xlabel("Growth ratio (KO / WT)")
plt.ylabel("Number of reactions")
plt.title("Day 5: Distribution of single-KO growth ratios")
plt.tight_layout()
plt.savefig("day5_single_KO_growth_ratio_hist.png", dpi=300)
plt.close()

# Double-reaction KO (synthetic lethality)

# 1 Choose candidate reactions for double KO
#     Strategy: choose reactions that are:
#       - solution status is "optimal"
#       - single KO is not lethal (growth_ratio > single_lethal_cutoff)
#     To keep runtime reasonable, we only take the first N candidates.

single_lethal_cutoff = 0.2   # <0.2 ~ strongly essential
max_candidates = 30          # you can increase if your machine is fast enough

candidates = ess_rxn[
    (ess_rxn["status"] == "optimal") &
    (ess_rxn["growth_ratio"] > single_lethal_cutoff)
].sort_values("growth_ratio")

candidate_ids = list(candidates.index[:max_candidates])
print(
    f"Selected {len(candidate_ids)} candidate reactions for double KO analysis.")
print("Example candidates:", candidate_ids[:10])


# 2 Prepare containers for double KO results
n = len(candidate_ids)
growth_mat = pd.DataFrame(
    np.nan,
    index=candidate_ids,
    columns=candidate_ids,
    dtype=float
)

synthetic_pairs = []  # store synthetic lethal pairs as dict


# thresholds to define synthetic lethality
single_safe_cutoff = 0.5   # both single KOs must have growth_ratio > 0.5
double_lethal_cutoff = 0.05  # double KO growth_ratio < 0.05 considered lethal

print(f"\nSynthetic lethal criteria:")
print(f"  Single KO growth_ratio > {single_safe_cutoff}")
print(f"  Double KO growth_ratio < {double_lethal_cutoff}\n")


# 3 Loop over all pairs (i, j) with i <= j
for i, rxn_i in enumerate(candidate_ids):
    for j, rxn_j in enumerate(candidate_ids):
        if j < i:  # fill symmetric matrix only once
            continue

        with model:
            # knock out both reactions in the temporary model
            model.reactions.get_by_id(rxn_i).knock_out()
            model.reactions.get_by_id(rxn_j).knock_out()

            sol_ij = model.optimize()
            growth_ij = sol_ij.objective_value if sol_ij.status == "optimal" else 0.0

        if wt_growth > 0:
            gr_ij = growth_ij / wt_growth
        else:
            gr_ij = 0.0

        # fill matrix (symmetric)
        growth_mat.loc[rxn_i, rxn_j] = gr_ij
        growth_mat.loc[rxn_j, rxn_i] = gr_ij

        # check synthetic lethality
        gr_i = ess_rxn.loc[rxn_i, "growth_ratio"]
        gr_j = ess_rxn.loc[rxn_j, "growth_ratio"]

        if (gr_i > single_safe_cutoff) and (gr_j > single_safe_cutoff) and (gr_ij < double_lethal_cutoff):
            synthetic_pairs.append({
                "rxn_i": rxn_i,
                "rxn_j": rxn_j,
                "single_gr_i": gr_i,
                "single_gr_j": gr_j,
                "double_gr_ij": gr_ij
            })

    print(f"Finished row {i+1}/{n} ({rxn_i})")

print("\nDouble KO simulation finished.")


# 4 Save double KO growth matrix
growth_mat.to_csv("day5_double_KO_growth_matrix.csv")

# 5 Save synthetic lethal pairs (if any)
if synthetic_pairs:
    sl_df = pd.DataFrame(synthetic_pairs)
    sl_df = sl_df.sort_values("double_gr_ij")
    sl_df.to_csv("day5_synthetic_lethal_pairs.csv", index=False)
    print(f"Found {len(sl_df)} synthetic lethal pairs.")
    print("Saved: day5_synthetic_lethal_pairs.csv")
    print(sl_df.head())
else:
    print("No synthetic lethal pairs found under current thresholds.")
    sl_df = pd.DataFrame(
        columns=["rxn_i", "rxn_j", "single_gr_i", "single_gr_j", "double_gr_ij"])


# 6 Plot heatmap of double KO growth ratios
plt.figure(figsize=(8, 6))
plt.imshow(growth_mat.values, origin="lower", aspect="auto")
plt.colorbar(label="Growth ratio (double KO / WT)")
plt.xticks(range(n), candidate_ids, rotation=90)
plt.yticks(range(n), candidate_ids)
plt.title("Day 5: Double KO growth ratio matrix (candidate reactions)")
plt.tight_layout()
plt.savefig("day5_double_KO_growth_heatmap.png", dpi=300)
plt.close()
