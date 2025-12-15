Learning Flux Balance Analysis (FBA) and Metabolic Modeling with COBRApy

This project documents a stepwise, end-to-end learning and research pipeline for constraint-based metabolic modeling using COBRApy, progressing from foundational FBA concepts to realistic tumor microenvironment (TME)–constrained T-cell metabolic analysis.

Phase I – Core FBA and Network Analysis (Day 1–10)

Goal: Build a rigorous foundation in constraint-based modeling, robustness analysis, and flux space interpretation.

Day 1–3: Basic FBA implementation, flux visualization, FVA, reaction/gene essentiality scans, and essentiality classification.

Day 4–6: Metabolic robustness and vulnerability analysis, including single- and double-knockout simulations, synthetic lethality mapping, and GPR-based gene knockouts.

Day 7: Construction of gene–reaction bipartite metabolic networks with essentiality annotations and topological vulnerability analysis.

Day 8–10: Nutrient limitation modeling, robustness curves, trade-off analysis, and flux space exploration via sampling and PCA.

Phase II – Advanced Flux Trade-offs and Expression Integration (Day 11–15)

Goal: Understand metabolic trade-offs and incorporate transcriptomic information.

Day 11–12: Multi-objective FBA and Pareto frontier analysis to characterize biomass–product trade-offs.

Day 13–15: Reaction-level expression mapping and E-Flux–based constraint integration; subsystem-level flux rewiring analysis (validated using mock expression data).

Phase III – T Cell–Specific Expression-Constrained Modeling (Day 16–20)

Goal: Apply expression-constrained modeling to real immune-cell data.

Day 16–18: Construction of real T-cell pseudo-bulk expression profiles (Blood, TumorEdge, TumorCore) and generation of expression-constrained flux distributions using E-Flux.

Day 19–20: Quantification of metabolic flux rewiring across microenvironments, including reaction-level log₂FC analysis, subsystem summaries, and PCA-based metabolic signature analysis, revealing a metabolic activation gradient from Blood → TumorEdge → TumorCore.

Phase IV – Tumor Microenvironment–Constrained Vulnerability Analysis (Day 21–28)

Goal: Identify how realistic TME nutrient constraints reshape T-cell metabolic robustness.

Integrated literature-inspired nutrient uptake bounds to simulate physiologically realistic Blood, TumorEdge, and TumorCore environments.

Performed single-reaction knockout (KO) screening under expression- and TME-constrained models.

Compared rich vs TME essentiality patterns to assess environment-induced vulnerability shifts.

Demonstrated that TME constraints collapse metabolic buffering, leading to predominantly binary (essential) survival dependencies rather than creating new lethal reactions.

Current Focus and Next Steps

Extend KO analyses to partial reaction inhibition to model pharmacologic perturbations.

Replace biomass with functional objectives (ATP and nucleotide production).

Identify environment-selective functional vulnerabilities across Blood, TumorEdge, and TumorCore.  

Phase IV – Tumor Microenvironment–Constrained Vulnerability Analysis (Day 21–29)

Goal: Identify how realistic TME nutrient constraints reshape T-cell metabolic robustness.

Integrated literature-inspired nutrient uptake bounds to simulate physiologically realistic Blood, TumorEdge, and TumorCore environments.

Performed single-reaction knockout (KO) screening under expression- and TME-constrained models.

Compared rich vs TME essentiality patterns to assess environment-induced vulnerability shifts.

Demonstrated that TME constraints collapse metabolic buffering, leading to predominantly binary (essential) survival dependencies rather than creating new lethal reactions.

Day 29 – Partial inhibition analysis (pharmacologic proxy):

Extended KO analysis to partial reaction inhibition (α = 0.5) to model graded, drug-like perturbations.

Observed an almost complete absence of graded metabolic sensitivities across Blood, TumorEdge, and TumorCore.

Partial inhibition failed to induce near-essential or strongly sensitive phenotypes, even under severe TME constraints.

These results indicate that T-cell metabolism under realistic TME conditions operates in a binary survival regime, with reactions being either fully buffered or becoming lethal only upon complete inhibition.

Day 30 – Functional objective analysis under realistic TME

Replaced biomass maximization with functional metabolic objectives, including ATP and nucleotide (AMP, GMP, IMP, UMP, CTP) production, to assess environment-dependent metabolic capacity beyond growth.

Computed maximal functional capacities under expression- and TME-constrained models.

Revealed consistent environment-dependent functional hierarchies, with TumorCore exhibiting the highest functional capacity, followed by TumorEdge and Blood.

These results demonstrate that TME-induced metabolic reprogramming enhances specific functional outputs even under severe nutrient limitation, highlighting a decoupling between growth and functional metabolic capacity.

Day 31 – Robustness of functional capacity to growth constraints

Evaluated functional capacities under fixed biomass constraints ranging from 60% to 90% of maximal growth to assess sensitivity to growth trade-offs.

Observed that ATP production capacity remained nearly constant across a wide range of biomass constraints within each environment.

This robustness indicates that functional metabolic outputs are largely insensitive to moderate growth constraints and are instead primarily determined by microenvironmental and expression-driven constraints.

Together, these analyses reveal a clear separation between growth optimization and functional metabolic potential under realistic TME conditions.
