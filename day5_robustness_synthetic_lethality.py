import cobra
from cobra.io import load_model
from cobra.flux_analysis import single_reaction_deletion
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Load model and wild-type growth
# ------------------------------------------------------------
model = load_model("textbook")
wt_sol = model.optimize()
wt_growth = wt_sol.objective_value
print("WT biomass:", wt_growth)

# ------------------------------------------------------------
# Part 1 — Single-reaction KO robustness
# ------------------------------------------------------------
ess_rxn = single_reaction_deletion(model)

# Backup the original index for safety
ess_rxn = ess_rxn.copy()
idx = ess_rxn.index.astype(str)

# If the index already matches reaction IDs, use it
if all(i in model.reactions for i in idx):
    ess_rxn["reaction_id"] = idx
else:
    # Otherwise map by order to the model reaction list
    rxn_ids = [r.id for r in model.reactions]
    ess_rxn["reaction_id"] = rxn_ids[: len(ess_rxn)]

# Normalize growth ratio
ess_rxn["growth_ratio"] = ess_rxn["growth"] / wt_growth

# Set reaction IDs as the index (clean and safe)
ess_rxn = ess_rxn.set_index("reaction_id")

# Save single-KO results
ess_rxn.to_csv("day5_single_reaction_KO_robustness.csv")
print(ess_rxn.head())

# Robustness curve
sorted_gr = ess_rxn["growth_ratio"].sort_values().reset_index(drop=True)

plt.figure(figsize=(6, 4))
plt.plot(sorted_gr.values, marker=".", linestyle="-")
plt.xlabel("Reactions (sorted KO)")
plt.ylabel("Growth ratio (KO / WT)")
plt.title("Day 5: Single-reaction KO robustness")
plt.tight_layout()
plt.savefig("day5_single_KO_robustness_curve.png", dpi=300)
plt.close()

# Histogram of single-KO growth ratios
plt.figure(figsize=(6, 4))
ess_rxn["growth_ratio"].hist(bins=30)
plt.xlabel("Growth ratio (KO / WT)")
plt.ylabel("Count")
plt.title("Day 5: Single-KO growth ratio distribution")
plt.tight_layout()
plt.savefig("day5_single_KO_growth_ratio_hist.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# Part 2 — Double KO & Synthetic Lethality
# ------------------------------------------------------------
print("\n[Part 2] Selecting candidates for double KO...")

# Pick reactions with near-WT growth and excluding exchange reactions
candidate = ess_rxn[
    (ess_rxn["status"] == "optimal")
    & (ess_rxn["growth_ratio"] >= 0.95)
    & (ess_rxn["growth_ratio"] <= 1.05)
    & (~ess_rxn.index.str.startswith("EX_"))
].sort_values("growth_ratio", ascending=False)

max_candidates = 20
candidate_ids = list(candidate.index[:max_candidates])

print(f"Selected {len(candidate_ids)} candidate reactions.")
print("Example:", candidate_ids[:10])

# Sanity check: confirm all IDs exist in model
invalid = [rid for rid in candidate_ids if rid not in model.reactions]
if invalid:
    print("Invalid reaction IDs:", invalid)
    raise RuntimeError("Candidate list contains invalid reaction IDs.")

# Prepare results matrix
n = len(candidate_ids)
growth_mat = pd.DataFrame(
    np.nan, index=candidate_ids, columns=candidate_ids, dtype=float
)

synthetic_pairs = []

# Thresholds
single_safe_cutoff = 0.5      # single KO must retain growth
double_lethal_cutoff = 0.05   # double KO considered lethal

print("\nSynthetic lethal criteria:")
print("  single KO growth_ratio >", single_safe_cutoff)
print("  double KO growth_ratio <", double_lethal_cutoff, "\n")

# Double KO loop
for i, rxn_i in enumerate(candidate_ids):
    for j, rxn_j in enumerate(candidate_ids):
        if j < i:
            continue

        with model:
            model.reactions.get_by_id(rxn_i).knock_out()
            model.reactions.get_by_id(rxn_j).knock_out()
            sol = model.optimize()

            if sol.status == "optimal" and wt_growth > 0:
                gr_ij = sol.objective_value / wt_growth
            else:
                gr_ij = 0.0

        growth_mat.loc[rxn_i, rxn_j] = gr_ij
        growth_mat.loc[rxn_j, rxn_i] = gr_ij

        # Check for synthetic lethality
        gr_i = ess_rxn.loc[rxn_i, "growth_ratio"]
        gr_j = ess_rxn.loc[rxn_j, "growth_ratio"]

        if (gr_i > single_safe_cutoff) and (gr_j > single_safe_cutoff) and (
            gr_ij < double_lethal_cutoff
        ):
            synthetic_pairs.append(
                {
                    "rxn_i": rxn_i,
                    "rxn_j": rxn_j,
                    "single_gr_i": gr_i,
                    "single_gr_j": gr_j,
                    "double_gr_ij": gr_ij,
                }
            )

    print(f"Finished {i+1}/{n}: {rxn_i}")

print("\nDouble KO simulation finished.")

# Save results
growth_mat.to_csv("day5_double_KO_growth_matrix.csv")

if synthetic_pairs:
    sl_df = pd.DataFrame(synthetic_pairs).sort_values("double_gr_ij")
    sl_df.to_csv("day5_synthetic_lethal_pairs.csv", index=False)
    print(f"Found {len(sl_df)} synthetic lethal pairs.")
    print(sl_df.head())
else:
    sl_df = pd.DataFrame(
        columns=["rxn_i", "rxn_j", "single_gr_i", "single_gr_j", "double_gr_ij"]
    )
    print("No synthetic lethal pairs found with current thresholds.")

# Heatmap
plt.figure(figsize=(8, 6))
plt.imshow(growth_mat.values, origin="lower", aspect="auto")
plt.colorbar(label="Growth ratio (double KO / WT)")
plt.xticks(range(n), candidate_ids, rotation=90)
plt.yticks(range(n), candidate_ids)
plt.title("Day 5: Double KO growth ratio matrix")
plt.tight_layout()
plt.savefig("day5_double_KO_growth_heatmap.png", dpi=300)
plt.close()


