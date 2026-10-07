# Metabolic Systems Biology

A computational framework for studying **T-cell metabolism under tumor-microenvironment constraints** using COBRApy and constraint-based metabolic modeling.

The project progresses from core flux-balance analysis to expression-constrained and microenvironment-constrained models, with systematic perturbation analysis of metabolic robustness and vulnerability.

## Research Questions

This project explores how nutrient availability, gene-expression constraints, and objective selection reshape the feasible metabolic state of T cells.

Key questions include:

- How do blood, tumor-edge, and tumor-core environments alter feasible flux distributions?
- Which reactions become essential under tumor-microenvironment constraints?
- How robust are metabolic functions to partial inhibition?
- How do growth and functional metabolic objectives trade off?
- Can transcriptomic information be incorporated into genome-scale metabolic models to study immune-cell state?

## Analysis Framework

### 1. Core constraint-based modeling

- flux balance analysis (FBA)
- flux variability analysis (FVA)
- reaction and gene essentiality
- single- and double-knockout simulations
- synthetic-lethality analysis
- flux-space sampling and PCA

### 2. Network and robustness analysis

- gene–reaction network construction
- nutrient-limitation simulations
- robustness curves
- pathway-level flux summaries
- multi-objective FBA and Pareto-front analysis

### 3. Expression-constrained modeling

- reaction-level expression mapping
- E-Flux-style constraint integration
- pseudo-bulk expression profiles
- subsystem-level flux rewiring analysis

### 4. Tumor-microenvironment modeling

Models are evaluated under literature-informed nutrient constraints representing:

- **Blood**
- **Tumor Edge**
- **Tumor Core**

These environments are combined with expression-derived constraints to investigate context-dependent metabolic vulnerability.

### 5. Perturbation analysis

The framework includes:

- complete reaction knockouts
- partial reaction inhibition
- environment-specific essentiality
- growth-versus-function trade-offs
- alternative objectives such as ATP and nucleotide production

## Main Tools

```text
Python
COBRApy
pandas
NumPy
scikit-learn
matplotlib
genome-scale metabolic models
FBA / FVA / KO analysis / E-Flux
```

## Interpretation

This repository is intended as an exploratory systems-biology framework. Model outputs depend strongly on the selected reconstruction, objective function, exchange constraints, and expression-to-flux mapping assumptions; biological conclusions should therefore be interpreted in the context of those modeling choices.

## Research Context

Constraint-based modeling · cancer metabolism · T-cell biology · tumor microenvironment · systems biology · metabolic vulnerability · multi-omics integration
