# metabolic-systems-biology
Learning Flux Balance Analysis (FBA) and metabolic modeling with COBRApy.   

Day 1: first FBA script + flux visualization.  

Day 2: Flux Variability Analysis (FVA) + knockout simulation.  

Day 3: reaction/gene essentiality scan + essentiality classification + basic visualization.  

Day 4: FVA-based metabolic rewiring analysis after PFK knockout (Δ flux-range detection + top-perturbed reactions).  

Day 5: robustness analysis (single-KO curve) + double knockout synthetic lethality mapping.  

Day 6: gene knockout analysis (GPR-based single-gene KO) + gene essentiality classification + robustness curves + optional pathway-level gene KO.  

Day 7: Constructed a gene–reaction bipartite metabolic network, annotated nodes with essentiality from single-gene/reaction KO, computed topological centrality metrics, identified articulation-point vulnerabilities, and generated multiple network visualizations.  

Day 8: Modeled glucose-dependent growth by varying exchange-reaction bounds, showing classic linear limitation behavior in FBA.  

Day 9: Explored the feasible flux space via randomized sampling and visualized flux variability and PCA structure.  

Day 10: Nutrient-scan trade-off analysis  

Day 11: Implemented multi-objective FBA by constraining biomass to a fixed fraction of its maximum value and optimizing product secretion, generating a full biomass–product Pareto frontier for metabolic trade-off analysis.  

Day 12 – Reaction-level metabolic rewiring: WT vs Pareto model  

Day 13: Computed reaction-level metabolic activity by mapping gene expression onto GPR rules, converting pseudo-bulk (or mock) gene expression into reaction expression values to enable expression-constrained metabolic modeling.  

Day 14: Applied gene-expression–based flux bound scaling (E-Flux) and compared WT vs expression-constrained fluxes to reveal expression-driven metabolic rewiring.  

Day 15: Performed subsystem-level flux rewiring analysis (WT vs E-Flux); results appear flat because Day 13 used random mock expression, so the plots are not biologically meaningful.  

Day 16: Built real T-cell pseudo-bulk expression (Blood/Core/Edge) for downstream expression-constrained metabolic modeling.  

Day 17: Computed reaction-level metabolic activity from real T-cell pseudo-bulk expression (Blood/Core/Edge) by mapping gene symbols to Ensembl IDs and evaluating GPR rules.  

Day 18: Applied E-Flux using reaction-level expression (Blood/Core/Edge) to generate expression-constrained flux distributions and compare biomass capacities across samples.  

Day 19: Quantified metabolic flux rewiring across Blood/Core/Edge by comparing E-Flux solutions at reaction and subsystem levels, identifying pathways with the largest |log₂FC| shifts.  

Day 20: Performed PCA and subsystem-level metabolic signature analysis across Blood/Core/Edge to identify global flux patterns, revealing a metabolic activation gradient from Blood → TumorEdge → TumorCore.  

Day 21: Performed single-reaction knockout (KO) essentiality screening under E-Flux–constrained Blood/Core/Edge models to identify condition-specific metabolic vulnerabilities.  

Day 22: Applied TME-specific nutrient uptake constraints on top of expression-constrained E-Flux models to simulate Blood, TumorEdge, and TumorCore T-cell metabolism under physiologically realistic microenvironment conditions.  

Day 23: Fixed the Human-GEM exchange reaction IDs and successfully activated true TME nutrient constraints, enabling a correct rich vs TME metabolic comparison for downstream flux and pathway analysis.  

Day 24: Performed single-reaction knockout screening under true TME-constrained E-Flux conditions to identify TME-specific essential metabolic reactions in Blood, TumorEdge, and TumorCore T cells.  

Day 25: Compared rich vs TME essentiality and identified that Blood shows no TME-specific essential reactions, while TumorEdge and TumorCore reveal environment-induced metabolic vulnerabilities.  

Day 26: Computed essentiality classes and subsystem summaries across Blood, TumorEdge, and TumorCore; found no rich-only or TME-only essential reactions, showing that TME impacts flux rewiring rather than creating new lethal points.  

Day 27: Implemented literature-inspired nutrient bounds for rich, tumor edge, and tumor core environments, combined them with reaction-level expression using E-Flux, and simulated realistic TME-constrained T-cell metabolism across Blood, TumorEdge, and TumorCore samples.  














