import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Day 26 – Core vs Conditional Essentiality Across Subsystems
#
# Purpose:
#   Instead of focusing only on “TME-only essential” reactions
#   (which may be zero in some samples), today we analyze:
#
#     • essential_in_both
#         → reactions essential in both rich and TME conditions
#           (metabolic core backbone)
#
#     • essential_only_in_rich
#         → essential only under nutrient-replete conditions
#           (conditionally essential)
#
#     • essential_only_in_TME
#         → essential only under TME stress (rare but included)
#
#     • non_essential
#
#   For each sample (Blood, TumorEdge, TumorCore) we generate:
#
#   1) Global counts of each essentiality class
#   2) Subsystem-level summary for each class
#   3) Barplots for visualization
#
# Input (from Day 25):
#   day25_essentiality_compare_{sample}.csv
#
# Required columns:
#   reaction_id, reaction_name, subsystem, essentiality_class
#
# Outputs:
#   • day26_essentiality_class_counts_{sample}.csv/png
#   • day26_core_backbone_subsystems_{sample}.csv/png
#   • day26_rich_only_subsystems_{sample}.csv/png
#   • day26_TME_only_subsystems_{sample}.csv/png (if non-zero)
#
# ============================================================

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = BASE
os.makedirs(OUT_DIR, exist_ok=True)

SAMPLES = ["Blood", "TumorEdge", "TumorCore"]

COMPARE_TEMPLATE = os.path.join(BASE, "day25_essentiality_compare_{sample}.csv")


# ------------------------------------------------------------
# Helper: load the Day 25 comparison table
# ------------------------------------------------------------
def load_day25_table(sample: str) -> pd.DataFrame | None:
    """
    Load the Day 25 essentiality comparison table for a given sample.

    Returns
    -------
    DataFrame or None
        The full table if found; otherwise None.
    """
    path = COMPARE_TEMPLATE.format(sample=sample)
    if not os.path.exists(path):
        print(f"[Warning] Day 25 file not found for {sample}: {path}")
        return None

    print(f"Loading Day 25 comparison for {sample}: {path}")
    df = pd.read_csv(path)

    # Check required columns
    required_cols = ["reaction_id", "reaction_name",
                     "subsystem", "essentiality_class"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"{path} is missing required column '{col}'")

    return df


# ------------------------------------------------------------
# Helper: summarize global class counts + barplot
# ------------------------------------------------------------
def summarize_class_counts(df: pd.DataFrame, sample: str) -> None:
    """
    Count how many reactions fall into each essentiality class,
    and generate a barplot + CSV summary.
    """
    class_order = [
        "essential_in_both",
        "essential_only_in_rich",
        "essential_only_in_TME",
        "non_essential",
    ]

    counts = df["essentiality_class"].value_counts()
    counts_ordered = [int(counts.get(c, 0)) for c in class_order]

    # Save CSV
    class_df = pd.DataFrame({
        "essentiality_class": class_order,
        "count": counts_ordered,
    })
    out_csv = os.path.join(
        OUT_DIR, f"day26_essentiality_class_counts_{sample}.csv"
    )
    class_df.to_csv(out_csv, index=False)
    print(f"Saved class count table: {out_csv}")

    # Barplot
    plt.figure(figsize=(6, 4))
    x = np.arange(len(class_order))
    plt.bar(x, counts_ordered)
    plt.xticks(x, class_order, rotation=20, ha="right")
    plt.ylabel("Number of reactions")
    plt.title(f"Day 26 – Essentiality classes ({sample})")
    plt.tight_layout()

    out_png = os.path.join(
        OUT_DIR, f"day26_essentiality_class_counts_{sample}.png"
    )
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Saved class count barplot: {out_png}")


# ------------------------------------------------------------
# Helper: summarize subsystems for a specific essentiality class
# ------------------------------------------------------------
def summarize_subsystems_for_class(
    df: pd.DataFrame,
    sample: str,
    target_class: str,
    prefix: str,
    top_n: int = 15,
) -> None:
    """
    Filter reactions by essentiality_class, summarize number of reactions
    per subsystem, and generate a CSV + barplot.

    Parameters
    ----------
    df : DataFrame
        Day 25 comparison table.
    sample : str
        Sample name.
    target_class : str
        One of: essential_in_both, essential_only_in_rich,
                essential_only_in_TME, non_essential.
    prefix : str
        Output filename prefix, e.g. "core_backbone" or "rich_only".
    top_n : int
        Number of top subsystems to display in the barplot.
    """
    subset = df[df["essentiality_class"] == target_class].copy()
    print(f"{sample}: {target_class} reactions = {len(subset)}")

    # If no reactions in this class, write an empty table and skip plotting
    if subset.empty:
        out_csv = os.path.join(
            OUT_DIR, f"day26_{prefix}_subsystems_{sample}.csv"
        )
        pd.DataFrame(columns=["subsystem", "n_reactions"]).to_csv(out_csv, index=False)
        print(f"{sample}: No {target_class} reactions. Saved empty file: {out_csv}")
        return

    # Count reactions per subsystem
    sub_counts = (
        subset.groupby("subsystem")["reaction_id"]
        .nunique()
        .sort_values(ascending=False)
    )
    summary = sub_counts.reset_index().rename(
        columns={"reaction_id": "n_reactions"}
    )

    # Save CSV
    out_csv = os.path.join(
        OUT_DIR, f"day26_{prefix}_subsystems_{sample}.csv"
    )
    summary.to_csv(out_csv, index=False)
    print(f"Saved {target_class} subsystem summary: {out_csv}")

    # Barplot (Top N)
    top = summary.head(top_n)
    y_pos = np.arange(len(top))

    plt.figure(figsize=(8, 5))
    plt.barh(y_pos, top["n_reactions"])
    plt.yticks(y_pos, top["subsystem"])
    plt.xlabel("Number of reactions")
    plt.title(f"Day 26 – {target_class} subsystems ({sample})")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    out_png = os.path.join(
        OUT_DIR, f"day26_{prefix}_subsystems_{sample}.png"
    )
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Saved {target_class} subsystem barplot: {out_png}")


# ============================================================
# Main execution loop
# ============================================================

for sample in SAMPLES:
    print("\n" + "=" * 70)
    print(f"Day 26 – Core vs Conditional Essentiality for Sample: {sample}")
    print("=" * 70)

    df = load_day25_table(sample)
    if df is None:
        continue

    # 1) Global essentiality class counts
    summarize_class_counts(df, sample)

    # 2) Core metabolic backbone (essential in both conditions)
    summarize_subsystems_for_class(
        df,
        sample=sample,
        target_class="essential_in_both",
        prefix="core_backbone",
        top_n=15,
    )

    # 3) Conditionally essential in rich media
    summarize_subsystems_for_class(
        df,
        sample=sample,
        target_class="essential_only_in_rich",
        prefix="rich_only",
        top_n=15,
    )

    # 4) TME-specific essential reactions (if any)
    summarize_subsystems_for_class(
        df,
        sample=sample,
        target_class="essential_only_in_TME",
        prefix="TME_only",
        top_n=15,
    )

print("\nDay 26 core vs conditional essentiality analysis completed.")

