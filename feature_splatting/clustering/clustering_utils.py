"""
Clustering utilities for spatial clustering algorithms.

This module handles DBSCAN clustering, similarity analysis, and cluster organization
for feature splatting Gaussians.
"""

import numpy as np
import torch
from sklearn.cluster import DBSCAN


class SpatialClusterer:
    """Handles spatial clustering operations and similarity analysis."""

    def analyze_similarity_statistics(self, similarities: np.ndarray) -> dict:
        """
        Analyze similarity statistics for a label.
        
        Args:
            similarities: Array of similarity scores
            
        Returns:
            Dictionary with similarity statistics
        """
        return {
            "max": float(np.max(similarities)),
            "median": float(np.median(similarities)),
            "mean": float(np.mean(similarities)),
            "min": float(np.min(similarities)),
            "std_dev": float(np.std(similarities))
        }

    def filter_by_similarity(
        self,
        similarities: np.ndarray,
        threshold: float
    ) -> tuple[np.ndarray, int]:
        """
        Filter similarities by threshold.
        
        Args:
            similarities: Array of similarity scores
            threshold: Minimum similarity threshold
            
        Returns:
            Tuple of (mask, num_candidates)
        """
        mask = similarities > threshold
        num_candidates = mask.sum()
        return mask, num_candidates

    def apply_dbscan_clustering(
        self,
        positions: np.ndarray,
        eps: float = 0.1,
        min_samples: int = 100
    ) -> np.ndarray:
        """
        Apply DBSCAN clustering to positions.
        
        Args:
            positions: Array of 3D positions [num_points, 3]
            eps: DBSCAN epsilon parameter (neighborhood radius)
            min_samples: DBSCAN minimum samples parameter
            
        Returns:
            Cluster labels array (noise points labeled as -1)
        """
        if len(positions) < min_samples:
            # Not enough points for clustering
            return np.full(len(positions), -1)
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(positions)
        return cluster_labels

    def organize_cluster_data(
        self,
        cluster_labels: np.ndarray,
        positions: np.ndarray,
        indices: np.ndarray,
        similarities: np.ndarray
    ) -> list[dict]:
        """
        Organize clustering data into structured format.
        
        Args:
            cluster_labels: Array of cluster labels from DBSCAN
            positions: Array of 3D positions
            indices: Array of Gaussian indices
            similarities: Array of similarity scores
            
        Returns:
            List of cluster dictionaries
        """
        clusters = []
        unique_clusters = np.unique(cluster_labels[cluster_labels != -1])
        
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_positions = positions[cluster_mask]
            cluster_indices = indices[cluster_mask]
            cluster_similarities = similarities[cluster_mask]

            cluster_data = {
                "cluster_id": int(cluster_id),
                "size": len(cluster_positions),
                "positions": cluster_positions.tolist(),
                "gaussian_indices": cluster_indices.tolist(),
                "similarities": cluster_similarities.tolist(),
                "centroid": cluster_positions.mean(axis=0).tolist(),
                "bounds": {
                    "min": cluster_positions.min(axis=0).tolist(),
                    "max": cluster_positions.max(axis=0).tolist(),
                },
            }

            clusters.append(cluster_data)
        
        # Sort clusters by size (largest first)
        clusters.sort(key=lambda x: x["size"], reverse=True)
        return clusters

    def cluster_by_label(
        self,
        similarities: torch.Tensor,
        labels: list[str],
        gaussian_positions: np.ndarray,
        similarity_threshold: float = 0.5,
        dbscan_eps: float = 0.1,
        dbscan_min_samples: int = 20,
    ) -> dict[str, dict]:
        """
        Main clustering method. Cluster Gaussians with high enough similarity for each text label using DBSCAN.

        Args:
            similarities: Similarity scores of shape [num_gaussians, num_labels]
            labels: List of text labels
            gaussian_positions: Array of Gaussian 3D positions
            similarity_threshold: Minimum similarity score to consider a Gaussian for a label
            dbscan_eps: DBSCAN epsilon parameter (neighborhood radius)
            dbscan_min_samples: DBSCAN minimum samples parameter

        Returns:
            Dictionary containing clustering results for each label
        """
        print(f"Clustering Gaussians for {len(labels)} labels...")

        results = {}
        all_indices = torch.arange(gaussian_positions.shape[0])

        for i, label in enumerate(labels):
            print(f"\nProcessing label '{label}'...")

            # Get similarity scores for this label
            label_similarities = similarities[:, i].cpu().numpy()

            # Analyze similarity statistics
            sim_stats = self.analyze_similarity_statistics(label_similarities)
            
            print(f"  Similarity statistics for '{label}':")
            print(f"    Highest: {sim_stats['max']:.4f}")
            print(f"    Median:  {sim_stats['median']:.4f}")
            print(f"    Average: {sim_stats['mean']:.4f}")
            print(f"    Lowest:  {sim_stats['min']:.4f}")
            print(f"    Std Dev: {sim_stats['std_dev']:.4f}")

            # Filter by similarity threshold
            high_similarity_mask, num_candidates = self.filter_by_similarity(
                label_similarities, similarity_threshold
            )

            print(f"  Found {num_candidates} Gaussians above threshold {similarity_threshold}")

            if num_candidates < dbscan_min_samples:
                print(f"  !  Not enough candidates for clustering (minimum: {dbscan_min_samples})")
                results[label] = {
                    'num_candidates': num_candidates,
                    'clusters': [],
                    'similarity_statistics': sim_stats
                }
                continue

            # Get positions of high-similarity Gaussians
            candidate_positions = gaussian_positions[high_similarity_mask]
            candidate_indices = all_indices[high_similarity_mask].cpu().numpy()
            candidate_similarities = label_similarities[high_similarity_mask]

            # Apply DBSCAN clustering
            cluster_labels = self.apply_dbscan_clustering(
                candidate_positions, dbscan_eps, dbscan_min_samples
            )

            unique_clusters, cluster_counts = np.unique(
                cluster_labels[cluster_labels != -1], return_counts=True
            )
            num_clusters = len(unique_clusters)
            num_noise = (cluster_labels == -1).sum()

            print(f"  Found {num_clusters} clusters, {num_noise} noise points")

            # Organize results by cluster
            clusters = self.organize_cluster_data(
                cluster_labels, candidate_positions, candidate_indices, candidate_similarities
            )

            results[label] = {
                "num_candidates": num_candidates,
                "num_clusters": num_clusters,
                "num_noise_points": num_noise,
                "clusters": clusters,
                "similarity_statistics": sim_stats,
            }
        
        return results
