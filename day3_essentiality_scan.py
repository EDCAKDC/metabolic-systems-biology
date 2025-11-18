import matplotlib.pyplot as plt
import numpy as np
import cobra
import pandas as pd
from cobra.flux_analysis import single_reaction_deletion, single_gene_deletion
from cobra.io import load_model

model = load_model("textbook")

# Reaction essentiality
ess_rxn = single_reaction_deletion(model)
ess_rxn.to_csv("day3_reaction_essentiality.csv")

# Gene essentiality
ess_gene = single_gene_deletion(model)
ess_gene.to_csv("day3_gene_essentiality.csv")

for rxn in model.exchanges:
    print(rxn.id, rxn.lower_bound, rxn.upper_bound)

# Wild-type growth (for normalization)
wt_solution = model.optimize()
wt_growth = wt_solution.objective_value
print("WT biomass:", wt_growth)

# Add normalized growth (= KO / WT)
ess_rxn["growth_ratio"] = ess_rxn["growth"] / wt_growth
ess_gene["growth_ratio"] = ess_gene["growth"] / wt_growth


def classify_growth(ratio, status):
    if status != "optimal":
        return "infeasible"
    if ratio >= 0.95:
        return "non-essential"
    elif 0.5 <= ratio < 0.95:
        return "partially-essential"
    elif 0.001 <= ratio < 0.5:
        return "strongly-essential"
    else:  # ratio < 0.001
        return "lethal"


ess_rxn["class"] = [classify_growth(r, s) for r, s in zip(
    ess_rxn["growth_ratio"], ess_rxn["status"])]
ess_gene["class"] = [classify_growth(r, s) for r, s in zip(
    ess_gene["growth_ratio"], ess_gene["status"])]

# Save updated tables
ess_rxn.to_csv("day3_reaction_essentiality_annotated.csv")
ess_gene.to_csv("day3_gene_essentiality_annotated.csv")


# Reaction KO histogram
plt.figure(figsize=(6, 4))
ess_rxn["growth_ratio"].hist(bins=30)
plt.xlabel("Growth ratio (KO / WT)")
plt.ylabel("Number of reactions")
plt.title("Reaction essentiality distribution")
plt.tight_layout()
plt.savefig("day3_rxn_growth_ratio_hist.png", dpi=300)
plt.close()

# Gene KO histogram
plt.figure(figsize=(6, 4))
ess_gene["growth_ratio"].hist(bins=30)
plt.xlabel("Growth ratio (KO / WT)")
plt.ylabel("Number of genes")
plt.title("Gene essentiality distribution")
plt.tight_layout()
plt.savefig("day3_gene_growth_ratio_hist.png", dpi=300)
plt.close()

# Top 20 most essential reactions (smallest growth_ratio among optimal)
top_rxn = (
    ess_rxn[ess_rxn["status"] == "optimal"]
    .sort_values("growth_ratio")
    .head(20)
)

plt.figure(figsize=(8, 5))
plt.barh(top_rxn.index, top_rxn["growth_ratio"])
plt.xlabel("Growth ratio (KO / WT)")
plt.ylabel("Reaction")
plt.title("Top 20 essential reactions")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("day3_rxn_top20_essential.png", dpi=300)
plt.close()

# Top 20 most essential genes
top_gene = (
    ess_gene[ess_gene["status"] == "optimal"]
    .sort_values("growth_ratio")
    .head(20)
)

plt.figure(figsize=(8, 5))
plt.barh(top_gene.index, top_gene["growth_ratio"])
plt.xlabel("Growth ratio (KO / WT)")
plt.ylabel("Gene")
plt.title("Top 20 essential genes")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("day3_gene_top20_essential.png", dpi=300)
plt.close()

print("Saved plots for essentiality analysis.")
