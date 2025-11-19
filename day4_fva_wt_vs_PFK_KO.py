import cobra
import pandas as pd
import matplotlib.pyplot as plt
from cobra.io import load_model
from cobra.flux_analysis import flux_variability_analysis


model = load_model("textbook")
print("Loaded model:", model)

# FVA for WT
print("\nRunning FVA for WT ...")
fva_wt = flux_variability_analysis(
    model,
    fraction_of_optimum=1.0  # keep biomass at 100% of optimum
)

# move index into a column and rename for clarity
fva_wt = (
    fva_wt.reset_index()
    .rename(columns={
        "index": "reaction",
        "minimum": "min_wt",
        "maximum": "max_wt"
    })
)

# flux range in WT
fva_wt["range_wt"] = fva_wt["max_wt"] - fva_wt["min_wt"]

# FVA for PFK knockout

with model:  # context manager: changes are temporary
    # knock out PFK
    pfk = model.reactions.get_by_id("PFK")
    pfk.knock_out()
    print("Knocked out reaction:", pfk.id)

    # check biomass after KO
    sol_ko = model.optimize()
    print("Biomass after PFK KO:", sol_ko.objective_value)

    # FVA under KO
    fva_ko = flux_variability_analysis(
        model,
        fraction_of_optimum=1.0
    )

# move index into a column and rename
fva_ko = (
    fva_ko.reset_index()
    .rename(columns={
        "index": "reaction",
        "minimum": "min_ko",
        "maximum": "max_ko"
    })
)

# flux range under KO
fva_ko["range_ko"] = fva_ko["max_ko"] - fva_ko["min_ko"]

# Merge WT and KO FVA tables
fva_merge = pd.merge(fva_wt, fva_ko, on="reaction", how="inner")

# change in range: KO - WT
fva_merge["delta_range"] = fva_merge["range_ko"] - fva_merge["range_wt"]
fva_merge["delta_abs"] = fva_merge["delta_range"].abs()

# also track changes in min/max if needed
fva_merge["delta_min"] = fva_merge["min_ko"] - fva_merge["min_wt"]
fva_merge["delta_max"] = fva_merge["max_ko"] - fva_merge["max_wt"]

# save full table
fva_merge.to_csv("day4_FVA_WT_vs_PFK_KO_full.csv", index=False)
print("\nSaved: day4_FVA_WT_vs_PFK_KO_full.csv")
print(fva_merge.head())

# Top 20 reactions with largest |Δrange|
top20 = (
    fva_merge.sort_values("delta_abs", ascending=False)
    .head(20)
    .copy()
)

top20.to_csv("day4_FVA_PFK_top20_delta_range.csv", index=False)
print("\nSaved: day4_FVA_PFK_top20_delta_range.csv")
print(top20[["reaction", "range_wt", "range_ko", "delta_range"]])

# Plot Δrange for top 20 reactions

plt.figure(figsize=(10, 5))
plt.bar(top20["reaction"], top20["delta_range"])
plt.axhline(0, linestyle="--")
plt.ylabel("Δ Flux range (KO - WT)")
plt.title("Day 4: Change in FVA range after PFK knockout (Top 20 reactions)")
plt.xticks(rotation=75)
plt.tight_layout()
plt.savefig("day4_FVA_PFK_delta_range_top20.png", dpi=300)
plt.close()
