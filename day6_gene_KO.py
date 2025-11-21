import os
import cobra
from cobra.io import load_model
from cobra.flux_analysis import single_gene_deletion
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Utility: ensure output folders exist
DATA_DIR = "data"
FIG_DIR = "figures"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Load model and compute wild-type growth
print("Loading model ...")
model = load_model("textbook")

print("Running wild-type FBA ...")
wt_solution = model.optimize()

if wt_solution.status != "optimal":
    raise RuntimeError(
        f"Wild-type model is not optimal (status={wt_solution.status})")

wt_growth = wt_solution.objective_value
print(f"WT biomass (objective value): {wt_growth:.4f}\n")

# Part 1 – Inspect gene–reaction mapping (GPR)
print("[Part 1] Extracting gene–reaction (GPR) mapping ...")

# Build a simple gene → reactions mapping table
gene_to_rxn_records = []

for gene in model.genes:
    rxn_ids = [rxn.id for rxn in gene.reactions]
    rxn_subsystems = list(
        {rxn.subsystem for rxn in gene.reactions if rxn.subsystem})
    gene_to_rxn_records.append(
        {
            "gene_id": gene.id,
            "gene_name": gene.name,
            "n_reactions": len(rxn_ids),
            "reactions": ";".join(sorted(rxn_ids)),
            "subsystems": ";".join(sorted(rxn_subsystems)),
        }
    )

gene_rxn_df = pd.DataFrame(gene_to_rxn_records).sort_values("gene_id")
gene_rxn_path = os.path.join(DATA_DIR, "day6_gene_reaction_mapping.csv")
gene_rxn_df.to_csv(gene_rxn_path, index=False)
print(f"Saved gene–reaction mapping to: {gene_rxn_path}")
print(f"Total genes in model: {len(gene_rxn_df)}\n")
# Part 2 – Single-gene deletion (gene essentiality scan)

print("[Part 2] Running single-gene deletion scan ...")

# Run COBRApy's single_gene_deletion function
# This returns a DataFrame with growth and status for each gene KO
gko_results = single_gene_deletion(model)

# Backup the original index
gko_results = gko_results.copy()
gene_idx = gko_results.index.astype(str)

# If the index already corresponds to gene IDs, keep it
if all(gid in [g.id for g in model.genes] for gid in gene_idx):
    gko_results["gene_id"] = gene_idx
else:
    # Fallback: map by order (should rarely be needed)
    gene_ids = [g.id for g in model.genes]
    gko_results["gene_id"] = gene_ids[: len(gko_results)]

# Normalize growth to WT and merge basic gene info
gko_results["growth_ratio"] = gko_results["growth"] / wt_growth

# Attach gene name and number of affected reactions from mapping table
gko_results = gko_results.merge(
    gene_rxn_df[["gene_id", "gene_name", "n_reactions"]],
    on="gene_id",
    how="left",
)

# Set gene_id as clean index
gko_results = gko_results.set_index("gene_id")

# Classify gene essentiality based on growth_ratio


def classify_gene(row, essential_cutoff=0.05, partial_cutoff=0.5):
    """
    Simple rule-based gene essentiality classification.

    - essential: growth_ratio < essential_cutoff
    - partially_essential: essential_cutoff <= growth_ratio < partial_cutoff
    - non_essential: growth_ratio >= partial_cutoff
    """
    gr = row["growth_ratio"]

    if pd.isna(gr):
        return "unknown"

    if gr < essential_cutoff:
        return "essential"
    elif gr < partial_cutoff:
        return "partially_essential"
    else:
        return "non_essential"


gko_results["essentiality_class"] = gko_results.apply(classify_gene, axis=1)

# Save full gene KO table
gko_path = os.path.join(DATA_DIR, "day6_single_gene_KO_results.csv")
gko_results.to_csv(gko_path)
print(f"Saved single-gene KO results to: {gko_path}")

# Print a small summary
print("\nGene essentiality summary:")
print(gko_results["essentiality_class"].value_counts())
print()

# Part 3 – Gene-level robustness visualization
print("[Part 3] Plotting gene-level robustness curves and distributions ...")

# --- 3.1 Sorted robustness curve for gene KO ---
sorted_gr = (
    gko_results["growth_ratio"]
    .dropna()
    .sort_values()
    .reset_index(drop=True)
)

plt.figure(figsize=(6, 4))
plt.plot(sorted_gr.values, marker=".", linestyle="-")
plt.xlabel("Genes (sorted by KO impact)")
plt.ylabel("Growth ratio (KO / WT)")
plt.title("Gene robustness curve (single-gene KO)")
plt.tight_layout()
curve_path = os.path.join(FIG_DIR, "day6_gene_KO_robustness_curve.png")
plt.savefig(curve_path, dpi=300)
plt.close()
print(f"Saved robustness curve to: {curve_path}")

# --- 3.2 Histogram of growth ratios ---
plt.figure(figsize=(6, 4))
gko_results["growth_ratio"].dropna().hist(bins=30)
plt.xlabel("Growth ratio (KO / WT)")
plt.ylabel("Number of genes")
plt.title("Gene essentiality distribution")
plt.tight_layout()
hist_path = os.path.join(FIG_DIR, "day6_gene_KO_growth_ratio_hist.png")
plt.savefig(hist_path, dpi=300)
plt.close()
print(f"Saved growth-ratio histogram to: {hist_path}")

# --- 3.3 Barplot of essentiality classes ---
class_counts = (
    gko_results["essentiality_class"]
    .value_counts()
    .reindex(["essential", "partially_essential", "non_essential", "unknown"])
    .fillna(0)
)

plt.figure(figsize=(6, 4))
class_counts.plot(kind="bar")
plt.ylabel("Number of genes")
plt.title("Gene essentiality classes")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
bar_path = os.path.join(FIG_DIR, "day6_gene_essentiality_classes.png")
plt.savefig(bar_path, dpi=300)
plt.close()
print(f"Saved essentiality-class barplot to: {bar_path}\n")

# Part 4 – Optional: pathway-level gene KO using subsystems
print("[Part 4] Optional pathway-level gene KO (subsystem-based) ...")

# Helper: collect all subsystems to inspect what exists
all_subsystems = sorted(
    {rxn.subsystem for rxn in model.reactions if rxn.subsystem})
print("Found subsystems in model:")
for s in all_subsystems:
    print("  -", s)
print()


def get_genes_for_subsystem(m, keyword):
    """
    Collect all gene IDs that are associated with reactions whose
    subsystem string contains the given keyword (case-insensitive).
    """
    keyword_lower = keyword.lower()
    genes = set()

    for rxn in m.reactions:
        if rxn.subsystem and keyword_lower in rxn.subsystem.lower():
            for g in rxn.genes:
                genes.add(g.id)

    return sorted(genes)


# Define a few example pathway keywords (will be matched against reaction.subsystem)
pathway_keywords = [
    "Glycolysis",
    "TCA",
    "Pentose phosphate",
]

pathway_records = []

for kw in pathway_keywords:
    genes = get_genes_for_subsystem(model, kw)
    if not genes:
        # Skip if this subsystem keyword is not present in this model
        print(f"  [Skip] No genes found for subsystem keyword: '{kw}'")
        continue

    # Count how many reactions are affected
    affected_rxns = set()
    for gid in genes:
        gene = model.genes.get_by_id(gid)
        for rxn in gene.reactions:
            affected_rxns.add(rxn.id)

    # Perform pathway-level gene KO (knock out all genes simultaneously)
    with model:
        for gid in genes:
            model.genes.get_by_id(gid).knock_out()

        sol = model.optimize()
        if sol.status == "optimal" and wt_growth > 0:
            gr = sol.objective_value / wt_growth
        else:
            gr = 0.0

    pathway_records.append(
        {
            "pathway_keyword": kw,
            "n_genes": len(genes),
            "n_reactions": len(affected_rxns),
            "growth_ratio": gr,
        }
    )

# Save pathway KO results if any
if pathway_records:
    pathway_df = pd.DataFrame(pathway_records)
    pathway_path = os.path.join(DATA_DIR, "day6_pathway_gene_KO_results.csv")
    pathway_df.to_csv(pathway_path, index=False)
    print("Pathway-level KO results:")
    print(pathway_df)
    print(f"\nSaved pathway KO table to: {pathway_path}")

    # Simple barplot for pathway-level growth ratios
    plt.figure(figsize=(6, 4))
    plt.bar(pathway_df["pathway_keyword"], pathway_df["growth_ratio"])
    plt.ylabel("Growth ratio (KO / WT)")
    plt.title("Pathway-level gene KO (subsystem-based)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path_fig = os.path.join(FIG_DIR, "day6_pathway_gene_KO_barplot.png")
    plt.savefig(path_fig, dpi=300)
    plt.close()
    print(f"Saved pathway-level KO barplot to: {path_fig}")
else:
    print("No pathway KO results (keywords did not match any subsystems).")

print("\nDay 6 – gene knockout analysis completed.")
