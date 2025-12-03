import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cobra.io import read_sbml_model


# Resolve paths relative to this script


BASE = os.path.dirname(os.path.abspath(__file__))

PSEUDO_BULK_FILE = os.path.join(BASE, "pseudo_bulk_counts_T_cells.csv")
MODEL_FILE = os.path.join(BASE, "../database/Human-GEM.xml")

OUT_DIR = BASE


# Create output directory (BASE already exists)


# No need to create since BASE exists, but call for safety
os.makedirs(OUT_DIR, exist_ok=True)

# Load Human-GEM model


print(f"Loading Human-GEM model: {MODEL_FILE}")
model = read_sbml_model(MODEL_FILE)
print(f"Model loaded: {len(model.genes)} genes, {len(model.reactions)} reactions")

geneid_to_symbol = {}

# Build mapping ENSG → gene symbol when known
for g in model.genes:
    symbol = None
    if g.name and g.name != g.id:
        symbol = g.name

    for key in ("hgnc_symbol", "geneSymbol", "symbol", "SYMBOL"):
        if key in g.annotation:
            symbol = g.annotation[key]
            break

    if symbol is None:
        symbol = g.id

    geneid_to_symbol[g.id] = str(symbol)

print(f"Gene mapping contains {len(geneid_to_symbol)} entries")

# Load pseudo-bulk dataset

print(f"Loading pseudo-bulk dataset: {PSEUDO_BULK_FILE}")
expr = pd.read_csv(PSEUDO_BULK_FILE, index_col=0)

expr.index = expr.index.astype(str).str.replace('"', "").str.strip()
expr.columns = expr.columns.astype(str).str.replace('"', "").str.strip()

expr = expr.apply(pd.to_numeric, errors="coerce").fillna(0.0)
expr = np.log2(1 + expr)

# Convert gene symbols → Ensembl IDs (Human-GEM uses ENSG IDs)

print("Mapping gene symbols → Ensembl IDs ...")

from mygene import MyGeneInfo
mg = MyGeneInfo()

# Query mygene.info
out = mg.querymany(list(expr.index),
                   scopes="symbol",
                   fields="ensembl.gene",
                   species="human")

symbol_to_ensg = {}

for item in out:
    sym = item.get("query")
    if "notfound" in item and item["notfound"]:
        continue
    ens = item.get("ensembl")
    if isinstance(ens, list):
        ens = ens[0]   # take first if multiple
    if isinstance(ens, dict):
        ens = ens.get("gene")
    if ens:
        symbol_to_ensg[sym] = ens

print(f"Mapped {len(symbol_to_ensg)} / {len(expr.index)} gene symbols to Ensembl IDs.")

# Replace index with ENSG, drop unmapped genes
new_expr = {}
for sym, row in expr.iterrows():
    if sym in symbol_to_ensg:
        new_expr[symbol_to_ensg[sym]] = row

expr = pd.DataFrame(new_expr).T
print(f"Expression matrix after ID conversion: {expr.shape[0]} genes")

samples = list(expr.columns)
print(f"Expression matrix: {expr.shape[0]} genes × {len(samples)} samples")


# GPR evaluation (AND=min, OR=max)

DEFAULT_EXPR = 0.1

def clean_token(x):
    return x.strip().strip("()")

def evaluate_gpr(gpr_string, gene_vector):
    if not gpr_string or gpr_string.strip() == "":
        return DEFAULT_EXPR

    rule = gpr_string.replace("AND", "and").replace("OR", "or")
    or_terms = re.split(r"\s+or\s+", rule)

    or_values = []
    for term in or_terms:
        and_genes = re.split(r"\s+and\s+", term)
        and_vals = []

        for t in and_genes:
            raw = clean_token(t)
            if not raw:
                continue

            symbol = geneid_to_symbol.get(raw, raw)

            if symbol not in gene_vector.index and "." in symbol:
                alt = symbol.split(".")[0]
                if alt in gene_vector.index:
                    symbol = alt

            val = gene_vector.get(symbol, DEFAULT_EXPR)
            and_vals.append(float(val))

        if not and_vals:
            or_values.append(DEFAULT_EXPR)
        else:
            or_values.append(min(and_vals))

    return max(or_values)


# Run reaction-level evaluation for each sample

rxn_ids = [rxn.id for rxn in model.reactions]
rxn_expr_matrix = pd.DataFrame(index=rxn_ids)

for sample in samples:
    print(f"Processing sample: {sample}")
    gene_vec = expr[sample]

    values = []
    for rxn in model.reactions:
        val = evaluate_gpr(rxn.gene_reaction_rule, gene_vec)
        values.append(val)

    rxn_expr_matrix[sample] = values

    # Plot histogram
    plt.figure()
    plt.hist(values, bins=60)
    plt.title(f"Reaction expression — {sample}")
    plt.xlabel("Reaction expression (log2 scale)")
    plt.ylabel("Count")

    fig_path = os.path.join(OUT_DIR, f"day17_hist_{sample}.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=140)
    plt.close()
    print(f"Saved: {fig_path}")


# Save output

out_csv = os.path.join(OUT_DIR, "day17_reaction_expression_Tcells.csv")
rxn_expr_matrix.to_csv(out_csv)

print(f"\nSaved reaction-level matrix: {out_csv}")
print("Day17 completed.")
