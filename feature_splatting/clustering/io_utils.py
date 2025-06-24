"""
I/O utilities for feature splatting clustering.

This module contains all file input/output operations including:
- Reading label files
- Saving clustering results (JSON, CSV) and summary (JSON)
- JSON encoding for numpy types

Use `save_all_results` to save all clustering outputs.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def read_labels_file(labels_file_path: str) -> list[str]:
    """
    Read labels from a text file.

    Args:
        labels_file_path: Path to the labels file

    Returns:
        List of labels read from the file

    Raises:
        FileNotFoundError: If the file doesn't exist
        IOError: If there's an error reading the file
    """
    labels_path = Path(labels_file_path)

    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_file_path}")

    try:
        with open(labels_path, "r", encoding="utf-8") as f:
            # Filter out empty lines and comments (starting with '#')
            labels = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
        return labels
    except Exception as e:
        raise IOError(f"Error reading labels file: {e}") from e


def save_clustering_results(
    results: dict,
    output_dir: str = "clustering_results",
    clustering_metadata: Optional[dict] = None,
) -> None:
    """
    Save clustering results to a JSON file.

    Args:
        results: Clustering results dictionary
        output_dir: Directory to save results
        clustering_metadata: Optional metadata to include in results
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Add clustering parameters as metadata
    if clustering_metadata:
        results["_metadata"] = clustering_metadata

    with open(output_path / "clustering_results.json", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)


def save_clustering_summary(
    results: dict,
    labels: list[str],
    output_dir: str = "clustering_results",
) -> None:
    """
    Save clustering summary statistics to a JSON file.

    Args:
        results: Clustering results dictionary
        labels: List of text labels
        output_dir: Directory to save results
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    summary = {}
    for label in labels:
        if label in results:
            result = results[label]
            summary[label] = {
                "num_candidates": result["num_candidates"],
                "num_clusters": result.get("num_clusters", 0),
                "num_noise_points": result.get("num_noise_points", 0),
                "largest_cluster_size": result["clusters"][0]["size"] if result.get("clusters") else 0,
            }

    with open(output_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)


def save_cluster_csvs(
    results: dict,
    labels: list[str],
    output_dir: str = "clustering_results",
) -> None:
    """
    Save individual cluster data as CSV files for each label.

    Args:
        results: Clustering results dictionary
        labels: List of text labels
        output_dir: Directory to save results
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    for label in labels:
        if label in results and results[label]["clusters"]:
            cluster_data = []
            for cluster in results[label]["clusters"]:
                for i, (pos, idx, sim) in enumerate(
                    zip(cluster["positions"], cluster["gaussian_indices"], cluster["similarities"])
                ):
                    cluster_data.append(
                        {
                            "label": label,
                            "cluster_id": cluster["cluster_id"],
                            "gaussian_index": idx,
                            "x": pos[0],
                            "y": pos[1],
                            "z": pos[2],
                            "similarity": sim,
                        }
                    )

            df = pd.DataFrame(cluster_data)
            output_file = output_path / f"{label} - clusters.csv"
            df.to_csv(output_file, index=False)


def save_all_results(
    results: dict,
    labels: list[str],
    output_dir: str = "clustering_results",
    clustering_metadata: Optional[dict] = None,
) -> None:
    """
    Save all clustering results (JSON, summary, and CSV files).

    Args:
        results: Clustering results dictionary
        labels: List of text labels
        output_dir: Directory to save results
        clustering_metadata: Optional metadata to include in results
    """
    output_path = Path(output_dir)
    print(f"\nSaving results to {output_path}/")

    save_clustering_results(results, output_dir, clustering_metadata)
    print("  Saved clustering_results.json")

    save_clustering_summary(results, labels, output_dir)
    print("  Saved summary.json")

    save_cluster_csvs(results, labels, output_dir)
    print("  Saved cluster CSVs")
