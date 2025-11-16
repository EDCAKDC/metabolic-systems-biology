import os
import pandas as pd
import matplotlib.pyplot as plt
import cobra
from cobra.io import load_model
model = load_model("textbook")
solution = model.optimize()
print(solution.objective_value)
for rxn in model.reactions:
    print(rxn.id, solution.fluxes[rxn.id])


# Convert flux dictionary → pandas Series
flux = pd.Series(solution.fluxes)
flux.index = [rxn.id for rxn in model.reactions]

# Top 20 flux reactions
top20 = flux.abs().sort_values(ascending=False).head(20)
print(top20)

# Plot
plt.figure(figsize=(10, 5))
top20.plot(kind="bar")
plt.ylabel("Flux (mmol/gDW/hr)")
plt.title("Top 20 Reactions by Absolute Flux")
plt.xticks(rotation=75)
plt.tight_layout()
plt.show()
plt.savefig("flux_top20.png", dpi=300, bbox_inches="tight")
