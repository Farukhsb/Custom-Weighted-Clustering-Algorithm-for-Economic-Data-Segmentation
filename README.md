# Custom Weighted Clustering Algorithm for Economic Data Segmentation

A sophisticated Python implementation of an adaptive weighted clustering algorithm designed specifically for economic and policy analysis. This algorithm extends traditional clustering methods with dynamic feature weighting and robust centroid updates to handle the complexities of real-world economic datasets.

## Key Features

- **Adaptive Feature Weighting**: Automatically adjusts feature importance during clustering based on intra-cluster dispersion
- **Robust Centroid Calculation**: Hybrid mean-median approach for outlier-resistant cluster updates  
- **Economic Data Optimization**: Tailored for regional, firm-level, and labor market datasets
- **Convergence-Based Efficiency**: Stops automatically when cluster assignments stabilize

## Applications

- Regional economic profiling and development initiatives
- Industrial strategy through firm segmentation
- Labor market analysis for skills and automation risk assessment
- Targeted policy intervention design

## Output:
Cluster distribution: [[1.0, 59.0], [2.0, 65.0], [3.0, 54.0]]
This output corresponds to the S matrix required in the original algorithm specification —
a distribution of entities (rows) into clusters.

## Algorithm Overview

- Data Standardisation: Each feature scaled to [0, 1].

- Initialisation: Random selection of centroids and feature weights.

- Assignment Step: Compute weighted Euclidean distances and assign entities to closest centroids.

Update Step:

- Recompute centroids (mean–median combination).

- Recalculate feature weights based on intra-cluster dispersion.

- Convergence Check: Stop if assignments do not change between iterations.

## Mathematical Core

- Weighted Euclidean distance
    d = Σ wᵢ^β (aᵢ − bᵢ)²
- Feature weight update
   wⱼ = 1 / Σ [(Δⱼ / Δₜ)^(1 / (β − 1))]
- where Δⱼ is the dispersion of feature j across all clusters,
and β controls how strongly weighting adapts to dispersion.

## Research and Policy Relevance

This algorithm supports evidence-based decision-making in:

- Regional Economic Segmentation – Identify structurally similar areas.

- Industrial Strategy – Cluster firms by productivity, innovation, or trade exposure.

- Labour Market Analysis – Group occupations by automation risk and skill profile.

- Public Resource Allocation – Target interventions based on data-driven segmentation.

By adapting feature weighting dynamically, the algorithm highlights which economic factors matter most in defining structural similarity.

## Technical Highlights

- Written entirely in Python / NumPy (no external dependencies).

- Fully object-oriented and modular.

- Automatically standardises and reports convergence.

- Configurable parameters for cluster count and β value.

## Result Snippet

  | Clusters (K) | β (Beta) | Cluster Distribution                                 |
| ------------ | -------- | ---------------------------------------------------- |
| 2            | 1.1      | [[1.0, 89.0], [2.0, 89.0]]                           |
| 3            | 1.5      | [[1.0, 59.0], [2.0, 65.0], [3.0, 54.0]]              |
| 4            | 2.0      | [[1.0, 47.0], [2.0, 42.0], [3.0, 46.0], [4.0, 43.0]] |
