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

Day 29 – Partial inhibition analysis (pharmacologic proxy)

We extended classical knockout (KO) essentiality analysis to partial reaction inhibition (α = 0.5), aiming to approximate graded, drug-like perturbations of individual metabolic reactions.

Across Blood, TumorEdge, and TumorCore conditions, we observed a near-complete absence of graded metabolic sensitivities. Partial inhibition rarely induced strong or near-essential phenotypes, even under severe tumor microenvironment (TME) constraints.

Instead, most reactions remained fully buffered under partial inhibition and only became lethal upon complete loss of activity. These results indicate that T-cell metabolism under realistic TME conditions operates in a threshold-dominated, binary survival regime, rather than exhibiting smoothly graded vulnerabilities at the single-reaction level.

Importantly, this collapse-like behavior reflects structural bottlenecks in the metabolic network rather than numerical artifacts, consistent with feasibility transitions expected in constraint-based models.  

Day 30 – Functional objective analysis under realistic TME

To assess metabolic capabilities beyond growth, we replaced biomass maximization with functional metabolic objectives, including ATP and nucleotide (AMP, GMP, IMP, UMP, CTP) production.

Using expression- and TME-constrained models, we computed maximal functional capacities under each microenvironmental condition.

Despite severe nutrient limitations, we uncovered a consistent environment-dependent hierarchy of functional capacity: TumorCore > TumorEdge > Blood. Notably, this hierarchy contrasts with growth-based outcomes, indicating that functional metabolic output can be enhanced even when proliferation is constrained.

These results demonstrate a decoupling between growth optimization and functional metabolic capacity, highlighting TME-driven metabolic reprogramming as a mechanism that prioritizes specific functional outputs over maximal biomass accumulation.

Day 31 – Robustness of functional capacity to growth constraints

To test whether functional metabolic capacity depends on growth optimization, we evaluated ATP production capacity under fixed biomass constraints ranging from 60% to 90% of maximal growth.

Across all environments, ATP production capacity varied only minimally across a wide range of biomass constraints, displaying near-linear but shallow dependence on growth limitation.

This robustness indicates that functional metabolic outputs are largely insensitive to moderate growth trade-offs and are instead primarily governed by microenvironmental constraints and expression-driven flux limitations.

Together with Day 30, these results establish that functional metabolic capacity is not a byproduct of growth optimization but represents an independently regulated metabolic dimension.

Project Summary

This project establishes a comprehensive, end-to-end computational framework for modeling T-cell metabolism under realistic tumor microenvironment constraints. By integrating genome-scale metabolic modeling, expression-based flux constraints, physiologically informed nutrient limitations, and systematic perturbation analyses, we characterize the organizing principles of TME-constrained T-cell metabolism.

Across multiple analytical layers—including global flux states, subsystem rewiring, essentiality screening, partial inhibition, and functional capacity analysis—the results consistently support a binary, threshold-driven metabolic robustness regime. Individual reactions exhibit limited graded sensitivity, remaining buffered under partial inhibition and collapsing only upon complete loss of activity.

In contrast, functional metabolic capacity (e.g., ATP and nucleotide production) is shown to be decoupled from growth and robust to moderate biomass constraints. Instead, functional output is primarily determined by microenvironmental context and expression-driven metabolic reprogramming.

Together, these findings reveal a clear separation between growth optimization and functional metabolic potential in T cells under tumor-like conditions, highlighting hard metabolic bottlenecks and environment-driven reprogramming as dominant principles shaping immune cell metabolism in the TME.
