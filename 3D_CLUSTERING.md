# 3D Feature Clustering

Cluster 3D Gaussians from feature splatting checkpoints by semantic text labels using CLIP similarity and DBSCAN spatial clustering.

- [3D Feature Clustering](#3d-feature-clustering)
  - [Overview](#overview)
  - [Quick Start](#quick-start)
  - [cluster\_features.py](#cluster_featurespy)
    - [Basic Usage](#basic-usage)
    - [Parameters](#parameters)
    - [Example](#example)
    - [Parameter Tuning](#parameter-tuning)
    - [Outputs](#outputs)
  - [plot\_feature\_clusters.py](#plot_feature_clusterspy)
    - [Basic Usage](#basic-usage-1)
    - [Parameters](#parameters-1)
    - [Examples](#examples)
  - [Understanding Results](#understanding-results)
    - [Similarity Statistics](#similarity-statistics)
    - [Clustering Output](#clustering-output)
  - [Interactive Visualization](#interactive-visualization)

## Overview

Two scripts are available:

- **`cluster_features.py`**: main clustering script that takes feature splatting checkpoints and outputs cluster data
- **`plot_feature_clusters.py`**: interactive 3D visualization of clustering results

> [!IMPORTANT]
> An NVIDIA GPU with CUDA support is required.

## Quick Start

Assuming the feature splatting model has already been trained, and you have a `.ckpt` checkpoint file:

```bash
# Cluster Gaussians
python cluster_features.py data/checkpoint.ckpt --labels <text labels separated with whitespace>

# Visualize results
python plot_feature_clusters.py
```

Check the terminal for detailed information about the clustering process. To visualize results, open the generated `feature_clusters_plot.html` file.

## cluster_features.py

Run `python cluster_feature.py --help` for a quick summary of all parameters.

### Basic Usage

```bash
python cluster_features.py <checkpoint> --labels <label1> <label2> ... [options]
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--labels` | Required | Text labels to cluster by |
| `--output-dir` | `clustering_results` | Output directory |
| `--similarity-threshold` | `0.5` | Minimum similarity score (0.0-1.0) |
| `--dbscan-eps` | `0.1` | DBSCAN neighborhood radius |
| `--dbscan-min-samples` | `100` | Minimum points per cluster |
| `--batch-size` | `None` | GPU batch size (use if memory limited) |
| `--softmax-temp` | `0.5` | Similarity temperature (higher=softer) |

### Example

Advanced usage example:

```bash
python cluster_features.py data/step-000006999.ckpt \
    --labels table vase floor \
    --output-dir garden_clustering \
    --similarity-threshold 0.2 \
    --dbscan-eps 0.2 \
    --dbscan-min-samples 50 \
    --batch-size 250000 \
    --softmax-temp 2.0
```

### Parameter Tuning

**Too many small clusters?**

- Increase `--dbscan-eps` (e.g., 0.1 → 0.2). This makes DBSCAN consider points further apart as neighbors, creating larger clusters, but also potentially including more noise.
- Increase `--dbscan-min-samples` (e.g., 50 → 100). With this,  more points are required to form a core point, potentially eliminating small clusters.

**Not enough candidates found?**

- Lower `--similarity-threshold` (e.g., 0.5 → 0.3). This will allow Gaussians with a lower similarity score to be considered as candidates. May increase false positives.
- Increase `--softmax-temp` (e.g., 1.0 → 2.0). This makes similarity scores more evenly distributed (softer). If your threshold is below the mean, this can increase candidates. If above the mean, it may decrease candidates.
- Overall, it is wise to adjust both of these parameters together to achieve the desired outcome:
  - Lower threshold with higher temperature for broader, less precise matching
  - Higher threshold with lower temperature for selective, precise matching
  - Start with threshold adjustments first, then fine-tune with temperature
- These adjustments also depend on the training data and the quality of the model, as they are based on distilled knowledge. I.e., if an object does not appear in many training images, it's similarity score to a relevant label might be lower than desired.

**Limited GPU memory?**

- Decrease `--batch-size`. Every 50k gaussians occupy approximatelly 1GB of GPU memory. As such, for a 6GB GPU, use `--batch-size 250000` or lower. Using smaller batch sizes does not affect the quality of the results, as it is used only in similarity computations.

### Outputs

- `clustering_results.json` - Complete results grouped by label, with cluster information (e.g., gaussian indices) and other statistics
- `summary.json` - Summary per label (candidates, clusters, noise points)
- `{label}_clusters.csv` - Per-label cluster data for visualization (label, cluster id,gaussian index, x, y, z, and similarity)

## plot_feature_clusters.py

### Basic Usage

```bash
python plot_feature_clusters.py [options]
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--results-dir` | `clustering_results` | Input directory with clustering results |
| `--labels` | `None` | Specific labels to plot (all if None) |
| `--output` | `feature_clusters_plot.html` | Output HTML file |

### Examples

```bash
# Plot all clusters
python plot_feature_clusters.py

# Plot specific labels only
python plot_feature_clusters.py --labels table vase

# Custom results and output location
python plot_feature_clusters.py --results-dir garden_clustering --output my_plot.html
```

## Understanding Results

### Similarity Statistics

Console output shows statistics per label:

```txt
Similarity statistics for 'table':
  Highest: 0.3162    # Best matching Gaussian
  Median:  0.2045    # 50th percentile similarity
  Average: 0.2089    # Mean across all Gaussians
  Std Dev: 0.0456    # Variance in scores
```

**Interpretation:**

- High std dev (>0.05): label might be ambiguous
- Low max score (<0.3): label not well represented in scene, or temperature too high
- Large mean-median gap: skewed similarity distribution

### Clustering Output

```txt
table: 3 clusters from 1250 candidates
vase: 1 clusters from 890 candidates  
floor: No clustering performed - 45 candidates (minimum required: 100)
```

No clustering is performed if there are less candidates than `--dbscan-min-samples`.

## Interactive Visualization

The HTML output contains the interactive visualization. Click the labels in the legend to toggle individual clusters or entire label groups. Hover over points for detailed tooltip information.
