import os
import cobra
from cobra.io import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re


# Part 0 – Output folders and basic settings
DATA_DIR = "data"
FIG_DIR = "figures"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Random seed for reproducibility of mock expression
np.random.seed(123)

# Default expression value for genes not found in the table
DEFAULT_EXPR = 0.1

# Helper – parse and evaluate GPR rule with AND/OR logic


def evaluate_gpr(rule, gene_expr_dict, default_expr=DEFAULT_EXPR):
    """
    Evaluate a GPR (gene reaction rule) string using:
      - AND  -> min(expression)
      - OR   -> max(expression)

    Parameters
    ----------
    rule : str
        The gene_reaction_rule string (e.g. "(b0001 and b0002) or b0003").
    gene_expr_dict : dict
        Mapping {gene_id: expression_value}.
    default_expr : float
        Expression value to use if a gene is not found.

    Returns
    -------
    float
        Reaction-level expression value.
    """
    if rule is None:
        return np.nan
    rule = rule.strip()
    if rule == "":
        return np.nan

    # Tokenize: parentheses, 'and', 'or', or gene IDs (anything else)
    tokens = re.findall(r'\(|\)|and|or|[^\s()]+', rule)

    # Shunting-yard algorithm: infix -> postfix (RPN)
    output_queue = []
    op_stack = []

    precedence = {"or": 1, "and": 2}

    for tok in tokens:
        if tok in ("and", "or"):
            while (
                op_stack
                and op_stack[-1] in precedence
                and precedence[op_stack[-1]] >= precedence[tok]
            ):
                output_queue.append(op_stack.pop())
            op_stack.append(tok)
        elif tok == "(":
            op_stack.append(tok)
        elif tok == ")":
            while op_stack and op_stack[-1] != "(":
                output_queue.append(op_stack.pop())
            if op_stack and op_stack[-1] == "(":
                op_stack.pop()
        else:
            # gene ID token
            output_queue.append(tok)

    while op_stack:
        output_queue.append(op_stack.pop())

    # Evaluate postfix expression using min/max for AND/OR
    stack = []

    for tok in output_queue:
        if tok == "and":
            if len(stack) < 2:
                return np.nan
            b = stack.pop()
            a = stack.pop()
            stack.append(min(a, b))
        elif tok == "or":
            if len(stack) < 2:
                return np.nan
            b = stack.pop()
            a = stack.pop()
            stack.append(max(a, b))
        else:
            # gene ID -> expression
            expr_val = gene_expr_dict.get(tok, default_expr)
            stack.append(expr_val)

    if len(stack) == 0:
        return np.nan
    return float(stack[0])


# Part 1 – Load model and list genes
print("Loading model ...")
model = load_model("textbook")
print("Loaded model:", model)

genes = list(model.genes)
print(f"Total genes in model: {len(genes)}")

# Part 2 – Create mock gene expression table
# Here we create a simple mock expression profile:
#   expression ~ Uniform(0, 10)
# You can replace this later with real bulk / pseudo-bulk / scRNA expression.
gene_ids = [g.id for g in genes]
expr_values = np.random.uniform(low=0.0, high=10.0, size=len(gene_ids))

df_gene_expr = pd.DataFrame(
    {
        "gene_id": gene_ids,
        "expression": expr_values,
    }
)

gene_expr_csv = os.path.join(DATA_DIR, "day13_mock_gene_expression.csv")
df_gene_expr.to_csv(gene_expr_csv, index=False)
print(f"Saved mock gene expression table to: {gene_expr_csv}")

# Build dictionary for quick lookup
gene_expr_dict = dict(zip(df_gene_expr["gene_id"], df_gene_expr["expression"]))

# Part 3 – Compute reaction-level expression via GPR
print("\nComputing reaction-level expression from GPR rules ...")

rows = []

for rxn in model.reactions:
    rid = rxn.id
    name = rxn.name
    subsystem = rxn.subsystem if rxn.subsystem not in (None, "") else "NA"
    gpr_rule = rxn.gene_reaction_rule  # string with "and"/"or"
    num_genes = len(rxn.genes)

    if num_genes == 0 or gpr_rule.strip() == "":
        # No GPR: set to NaN or a neutral value
        rxn_expr = np.nan
    else:
        rxn_expr = evaluate_gpr(gpr_rule, gene_expr_dict,
                                default_expr=DEFAULT_EXPR)

    rows.append(
        {
            "rxn_id": rid,
            "reaction_name": name,
            "subsystem": subsystem,
            "gpr_rule": gpr_rule,
            "num_genes": num_genes,
            "reaction_expression": rxn_expr,
        }
    )

df_rxn_expr = pd.DataFrame(rows)

rxn_expr_csv = os.path.join(DATA_DIR, "day13_reaction_expression_from_GPR.csv")
df_rxn_expr.to_csv(rxn_expr_csv, index=False)
print(f"Saved reaction expression table to: {rxn_expr_csv}")


# Part 4 – Simple visualization of reaction expression
print("\nPlotting distribution of reaction expression ...")

# Drop NaN before plotting
valid_expr = df_rxn_expr["reaction_expression"].dropna()

plt.figure(figsize=(6, 5))
plt.hist(valid_expr, bins=30)
plt.xlabel("Reaction expression (GPR-based)")
plt.ylabel("Count")
plt.title("Day 13 – Distribution of reaction-level expression")
plt.tight_layout()

hist_fig = os.path.join(FIG_DIR, "day13_reaction_expression_hist.png")
plt.savefig(hist_fig, dpi=300)
plt.close()
print(f"Saved reaction expression histogram to: {hist_fig}")

print("\nDay 13 – GPR-based gene → reaction mapping finished.")
