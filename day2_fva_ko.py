import cobra
import pandas as pd
import matplotlib.pyplot as plt
from cobra.io import load_model
from cobra.flux_analysis import flux_variability_analysis

# Load model

model = load_model("textbook")

solution = model.optimize()
print("Optimal objective value (biomass):", solution.objective_value)

# Run FVA (keep biomass at 100% of optimum)

# fraction_of_optimum = 1.0 → biomass cannot drop below optimum
fva = flux_variability_analysis(
    model,
    fraction_of_optimum=1.0
)

# Reset and rename columns for clarity
fva = fva.reset_index().rename(columns={
    "index": "reaction",
    "minimum": "min_flux",
    "maximum": "max_flux"
})

# Add flux range column
fva["range"] = fva["max_flux"] - fva["min_flux"]

# Save results
fva.to_csv("day2_fva_result.csv", index=False)
print("\nSaved: day2_fva_result.csv")
print(fva.head())

# Plot Top 20 reactions by flux range
top20 = fva.sort_values("range", ascending=False).head(20)

plt.figure(figsize=(10, 5))
plt.bar(top20["reaction"], top20["range"])
plt.ylabel("Flux Range (max − min)")
plt.title("Top 20 Reactions by Flux Variability (FVA)")
plt.xticks(rotation=75)
plt.tight_layout()
plt.savefig("day2_fva_top20_range.png", dpi=300)
plt.show()

print("\nSaved: day2_fva_top20_range.png")
print(top20[["reaction", "min_flux", "max_flux", "range"]])


# KO one reaction
with model:
    # KO reaction
    rxn = model.reactions.get_by_id("PFK")
    rxn.knock_out()

    # FBA under KO
    solution_ko = model.optimize()
    print("Biomass after KO:", solution_ko.objective_value)

    # Flux under KO
    flux_ko = pd.Series(solution_ko.fluxes)
    flux_ko.index = [r.id for r in model.reactions]

    # Top 20 flux after KO
    top20_ko = flux_ko.abs().sort_values(ascending=False).head(20)
    print("\nTop 20 flux after KO:")
    print(top20_ko)

    # FVA under KO
    fva_ko = flux_variability_analysis(model, fraction_of_optimum=1.0)
    fva_ko = fva_ko.reset_index().rename(columns={
        "index": "reaction",
        "minimum": "min_flux",
        "maximum": "max_flux"
    })

    fva_ko["range"] = fva_ko["max_flux"] - fva_ko["min_flux"]
    print("\nFVA after KO:")
    print(fva_ko.head())
