import os
import cobra
from cobra.io import load_model
from cobra.flux_analysis import single_reaction_deletion, single_gene_deletion
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# Output folders
DATA_DIR = "data"
FIG_DIR = "figures"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Load model and compute WT growth
print("Loading model ...")
model = load_model("textbook")

print("Running wild-type FBA ...")
wt_sol = model.optimize()
if wt_sol.status != "optimal":
    raise RuntimeError(f"WT solution not optimal (status={wt_sol.status})")

wt_growth = wt_sol.objective_value
print(f"WT biomass: {wt_growth:.4f}\n")

# Helper: classification of essentiality based on growth ratio


def classify_essentiality(gr, essential_cutoff=0.05, partial_cutoff=0.5):
    """
    Classify essentiality based on growth ratio.

    Parameters
    ----------
    gr : float or NaN
        Growth ratio (KO / WT).
    essential_cutoff : float
        Threshold below which KO is considered lethal.
    partial_cutoff : float
        Threshold below which KO is considered partially essential.

    Returns
    -------
    str
        One of: 'essential', 'partially_essential', 'non_essential', 'unknown'.
    """
    if pd.isna(gr):
        return "unknown"

    if gr < essential_cutoff:
        return "essential"
    elif gr < partial_cutoff:
        return "partially_essential"
    else:
        return "non_essential"


# Part 1 – Reaction and gene essentiality (for network annotation)
print("[Part 1] Running reaction and gene single-deletion scans ...")

# ----- 1.1 Reaction deletion -----
rxn_del = single_reaction_deletion(model)
rxn_del = rxn_del.copy()
rxn_del["reaction_id"] = rxn_del.index.astype(str)
rxn_del["growth_ratio"] = rxn_del["growth"] / wt_growth
rxn_del["essentiality_class"] = rxn_del["growth_ratio"].apply(
    classify_essentiality
)

# Small summary
print("Reaction essentiality summary:")
print(rxn_del["essentiality_class"].value_counts())
print()

rxn_ess_path = os.path.join(
    DATA_DIR, "day7_reaction_essentiality_for_network.csv")
rxn_del.to_csv(rxn_ess_path, index=False)
print(f"Saved reaction essentiality table to: {rxn_ess_path}\n")

# Map: reaction_id -> essentiality_class
rxn_ess_map = rxn_del.set_index("reaction_id")["essentiality_class"].to_dict()

# ----- 1.2 Gene deletion -----
g_del = single_gene_deletion(model)
g_del = g_del.copy()
g_del["gene_id"] = g_del.index.astype(str)
g_del["growth_ratio"] = g_del["growth"] / wt_growth
g_del["essentiality_class"] = g_del["growth_ratio"].apply(
    classify_essentiality
)

print("Gene essentiality summary:")
print(g_del["essentiality_class"].value_counts())
print()

gene_ess_path = os.path.join(
    DATA_DIR, "day7_gene_essentiality_for_network.csv")
g_del.to_csv(gene_ess_path, index=False)
print(f"Saved gene essentiality table to: {gene_ess_path}\n")

# Map: gene_id -> essentiality_class
gene_ess_map = g_del.set_index("gene_id")["essentiality_class"].to_dict()

# Part 2 – Build gene–reaction bipartite network
print("[Part 2] Building gene–reaction bipartite network ...")

B = nx.Graph()

edge_records = []

for gene in model.genes:
    gene_id = gene.id
    gene_name = gene.name

    # Add gene node (bipartite set "gene")
    if gene_id not in B:
        B.add_node(
            gene_id,
            node_type="gene",
            bipartite="gene",
            label=gene_name if gene_name else gene_id,
            essentiality=gene_ess_map.get(gene_id, "unknown"),
        )

    for rxn in gene.reactions:
        rxn_id = rxn.id
        subsystem = rxn.subsystem if rxn.subsystem else ""

        # Add reaction node (bipartite set "reaction")
        if rxn_id not in B:
            B.add_node(
                rxn_id,
                node_type="reaction",
                bipartite="reaction",
                subsystem=subsystem,
                label=rxn_id,
                essentiality=rxn_ess_map.get(rxn_id, "unknown"),
            )

        # Add edge gene–reaction
        B.add_edge(gene_id, rxn_id)

        edge_records.append(
            {
                "gene_id": gene_id,
                "gene_name": gene_name,
                "reaction_id": rxn_id,
                "subsystem": subsystem,
            }
        )

# Save edge list
edge_df = pd.DataFrame(edge_records).drop_duplicates()
edges_path = os.path.join(DATA_DIR, "day7_gene_reaction_edges.csv")
edge_df.to_csv(edges_path, index=False)
print(f"Saved gene–reaction edge table to: {edges_path}")
print(
    f"Total nodes: {B.number_of_nodes()}, total edges: {B.number_of_edges()}\n")

# Part 3 – Network centrality and node metrics
print("[Part 3] Computing network centrality metrics ...")

# Degree (number of neighbors)
degree_dict = dict(B.degree())

# Degree centrality and betweenness centrality
deg_centrality = nx.degree_centrality(B)
btw_centrality = nx.betweenness_centrality(B)

node_records = []

for node_id, data in B.nodes(data=True):
    node_type = data.get("node_type", "unknown")
    essentiality = data.get("essentiality", "unknown")
    subsystem = data.get("subsystem", "") if node_type == "reaction" else ""

    node_records.append(
        {
            "node_id": node_id,
            "node_type": node_type,  # gene or reaction
            "degree": degree_dict.get(node_id, 0),
            "degree_centrality": deg_centrality.get(node_id, np.nan),
            "betweenness_centrality": btw_centrality.get(node_id, np.nan),
            "essentiality_class": essentiality,
            "subsystem": subsystem,
        }
    )

node_df = pd.DataFrame(node_records)

node_metrics_path = os.path.join(DATA_DIR, "day7_network_node_metrics.csv")
node_df.to_csv(node_metrics_path, index=False)
print(f"Saved node-level metrics to: {node_metrics_path}\n")

# Part 4 – Identify articulation points (vulnerability nodes)
print("[Part 4] Identifying articulation points (vulnerability nodes) ...")

articulation_nodes = list(nx.articulation_points(B))

art_records = []
for n in articulation_nodes:
    data = B.nodes[n]
    art_records.append(
        {
            "node_id": n,
            "node_type": data.get("node_type", "unknown"),
            "essentiality_class": data.get("essentiality", "unknown"),
            "degree": degree_dict.get(n, 0),
        }
    )

if art_records:
    art_df = pd.DataFrame(art_records)
    art_path = os.path.join(DATA_DIR, "day7_articulation_points.csv")
    art_df.to_csv(art_path, index=False)
    print(f"Found {len(art_df)} articulation points.")
    print(f"Saved articulation-point table to: {art_path}\n")
else:
    print("No articulation points found (graph has no critical single nodes).\n")


# Part 5 – Visualizations
print("[Part 5] Plotting network and distributions ...")

# --- 5.1 Degree distribution (genes vs reactions) ---
gene_degrees = node_df.loc[node_df["node_type"] == "gene", "degree"]
rxn_degrees = node_df.loc[node_df["node_type"] == "reaction", "degree"]

plt.figure(figsize=(6, 4))
plt.hist(gene_degrees, bins=20, alpha=0.7, label="genes")
plt.hist(rxn_degrees, bins=20, alpha=0.7, label="reactions")
plt.xlabel("Degree (number of neighbors)")
plt.ylabel("Count")
plt.title("Degree distribution (gene–reaction network)")
plt.legend()
plt.tight_layout()
deg_hist_path = os.path.join(
    FIG_DIR, "day7_degree_distribution_genes_vs_reactions.png")
plt.savefig(deg_hist_path, dpi=300)
plt.close()
print(f"Saved degree distribution histogram to: {deg_hist_path}")

# --- 5.2 Betweenness centrality histogram ---
plt.figure(figsize=(6, 4))
node_df["betweenness_centrality"].hist(bins=30)
plt.xlabel("Betweenness centrality")
plt.ylabel("Number of nodes")
plt.title("Betweenness centrality distribution")
plt.tight_layout()
btw_hist_path = os.path.join(FIG_DIR, "day7_betweenness_centrality_hist.png")
plt.savefig(btw_hist_path, dpi=300)
plt.close()
print(f"Saved betweenness centrality histogram to: {btw_hist_path}")

# --- 5.3 Bipartite network visualization ---
# To keep the figure readable, we can optionally restrict to a subgraph
# of nodes with degree >= min_degree_threshold
min_degree_threshold = 1  # set >1 if the network is too dense to visualize
sub_nodes = [n for n, d in degree_dict.items() if d >= min_degree_threshold]
H = B.subgraph(sub_nodes).copy()

print(
    f"Drawing bipartite network (subgraph with {H.number_of_nodes()} nodes) ...")

# Use a spring layout for visualization
pos = nx.spring_layout(H, seed=0)

# Prepare node colors: essential vs non-essential
color_map = {
    "essential": "#d62728",
    "partially_essential": "#ff7f0e",
    "non_essential": "#2ca02c",
    "unknown": "#7f7f7f",
}

node_colors = []
node_sizes = []

for n, data in H.nodes(data=True):
    ess = data.get("essentiality", "unknown")
    node_colors.append(color_map.get(ess, "#7f7f7f"))

    # Slightly bigger nodes for reactions
    if data.get("node_type") == "reaction":
        node_sizes.append(120)
    else:
        node_sizes.append(80)

plt.figure(figsize=(8, 6))
nx.draw_networkx_edges(H, pos, alpha=0.3, width=0.5)
nx.draw_networkx_nodes(
    H,
    pos,
    node_color=node_colors,
    node_size=node_sizes,
    linewidths=0.3,
    edgecolors="black",
)

# Optional: draw labels for high-degree nodes only
labels = {}
for n, d in H.degree():
    if d >= 5:  # show labels only for hubs
        labels[n] = n

nx.draw_networkx_labels(H, pos, labels=labels, font_size=6)

plt.axis("off")
plt.title("Day 7: Gene–reaction bipartite network\n(node color = essentiality)")
plt.tight_layout()
net_fig_path = os.path.join(
    FIG_DIR, "day7_gene_reaction_bipartite_network.png")
plt.savefig(net_fig_path, dpi=300)
plt.close()
print(f"Saved bipartite network figure to: {net_fig_path}")

print("\nDay 7 – gene–reaction network analysis completed.")
