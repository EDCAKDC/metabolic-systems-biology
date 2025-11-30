import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Part 0 – Folders and input file
DATA_DIR = "data"
FIG_DIR = "figures"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

FLUX_COMP_FILE = os.path.join(DATA_DIR, "day14_WT_vs_EFlux_fluxes.csv")

# Part 1 – Load WT vs E-Flux flux comparison
print("Loading WT vs E-Flux flux comparison ...")
df = pd.read_csv(FLUX_COMP_FILE)

required_cols = [
    "rxn_id",
    "reaction_name",
    "subsystem",
    "flux_WT",
    "flux_EFlux",
    "delta",
    "abs_delta",
    "log2_FC_abs",
]

for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Column '{c}' is missing from {FLUX_COMP_FILE}")

print(f"Total reactions in table: {len(df)}")

# Part 2 – Clean subsystem names
print("\n[Part 2] Cleaning subsystem names ...")

df["subsystem"] = df["subsystem"].fillna("NA")
df.loc[df["subsystem"] == "", "subsystem"] = "NA"

print("Unique subsystems:", len(df["subsystem"].unique()))


# Part 3 – Subsystem-level summary statistics
print("\n[Part 3] Computing subsystem-level summary ...")

grouped = (
    df.groupby("subsystem")
    .agg(
        n_reactions=("rxn_id", "count"),
        mean_abs_log2FC=("log2_FC_abs", "mean"),
        median_abs_log2FC=("log2_FC_abs", "median"),
        mean_abs_delta=("abs_delta", "mean"),
        median_abs_delta=("abs_delta", "median"),
    )
    .reset_index()
)

# Sort by mean |log2FC|, descending
grouped_sorted = grouped.sort_values(
    "mean_abs_log2FC", ascending=False
).reset_index(drop=True)

summary_csv = os.path.join(
    DATA_DIR, "day15_subsystem_flux_rewiring_summary.csv")
grouped_sorted.to_csv(summary_csv, index=False)
print(f"Saved subsystem summary to: {summary_csv}")


# Part 4 – Barplot: Top 15 subsystems by mean |log2FC|
print("\n[Part 4] Plotting top 15 subsystems by mean |log2FC| ...")

top_n = 15
top_subsystems = grouped_sorted.head(top_n)

plt.figure(figsize=(8, 6))
plt.barh(top_subsystems["subsystem"], top_subsystems["mean_abs_log2FC"])
plt.gca().invert_yaxis()
plt.xlabel("Mean |log2 fold-change| (|E-Flux| / |WT|)")
plt.ylabel("Subsystem")
plt.title("Day 15 – Top subsystems by flux rewiring (WT vs E-Flux)")
plt.tight_layout()

bar_fig = os.path.join(FIG_DIR, "day15_top_subsystems_mean_abs_log2FC.png")
plt.savefig(bar_fig, dpi=300)
plt.close()
print(f"Saved barplot to: {bar_fig}")


# Part 5 – Boxplot of log2_FC_abs per subsystem (optional)
print("\n[Part 5] Plotting boxplot of log2_FC_abs per subsystem (filtered) ...")

subsystems_to_plot = top_subsystems["subsystem"].tolist()
df_box = df[df["subsystem"].isin(subsystems_to_plot)].copy()

df_box["subsystem"] = pd.Categorical(
    df_box["subsystem"],
    categories=subsystems_to_plot,
    ordered=True,
)

plt.figure(figsize=(10, 6))
df_box.boxplot(column="log2_FC_abs", by="subsystem", vert=True)
plt.suptitle("")
plt.xlabel("Subsystem")
plt.ylabel("log2 fold-change (|E-Flux| / |WT|)")
plt.title("Day 15 – Flux rewiring distribution per subsystem")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

box_fig = os.path.join(FIG_DIR, "day15_subsystems_log2FC_boxplot.png")
plt.savefig(box_fig, dpi=300)
plt.close()
print(f"Saved boxplot to: {box_fig}")

print("\nDay 15 – Subsystem-level flux rewiring analysis finished.")
