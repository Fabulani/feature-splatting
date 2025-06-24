"""
Clustering Pipeline for Feature Splatting

This module contains the main clustering pipeline that orchestrates the complete
workflow from model loading to result saving for 3D Gaussian clustering.

Input: Feature Splatting model checkpoint (.ckpt) and text labels for clustering.

Output: Clustering results saved to specified directory in JSON and CSV formats. The pipeline also returns a dictionary with clustering results for each label.
"""

from .pytorch_utils import ModelLoader, TextEncoder, compute_gaussian_similarities, get_gaussian_positions
from .clustering_utils import SpatialClusterer
from .io_utils import save_all_results


class ClusteringPipeline:
    """
    Orchestrates the complete clustering pipeline for feature splatting Gaussians.
    
    This class handles the entire workflow:
    0. Loading the necessary models
    1. Encoding text labels using CLIP
    2. Computing similarities between Gaussians and text embeddings
    3. Performing spatial clustering using DBSCAN
    4. Saving results to files
    """

    def __init__(self, checkpoint_path: str):
        """
        Initialize the ClusteringPipeline.

        Args:
            checkpoint_path: Path to the feature splatting checkpoint file
        """
        print(f"Initializing ClusteringPipeline with checkpoint: {checkpoint_path}")
        
        self.model_loader = ModelLoader(checkpoint_path)
        self.text_encoder = TextEncoder()
        self.clusterer = SpatialClusterer()

    def run(
        self,
        labels: list[str],
        output_dir: str = "clustering_results",
        similarity_threshold: float = 0.5,
        dbscan_eps: float = 0.1,
        dbscan_min_samples: int = 20,
        batch_size: int = None,
        softmax_temp: float = 0.5,
        checkpoint_path: str = None,
        labels_file: str = None,
    ) -> dict[str, dict]:
        """
        Run the complete clustering pipeline from start to finish.

        Args:
            labels: List of text labels for clustering
            output_dir: Directory to save results
            similarity_threshold: Minimum similarity score to consider a Gaussian for a label
            dbscan_eps: DBSCAN epsilon parameter (neighborhood radius)
            dbscan_min_samples: DBSCAN minimum samples parameter
            batch_size: Batch size for processing Gaussians (for memory efficiency)
            softmax_temp: Temperature for similarity computation
            checkpoint_path: Path to checkpoint (for metadata)
            labels_file: Path to labels file (for metadata)

        Returns:
            Dictionary containing clustering results for each label
        """
        print(f"\n=== Running Clustering Pipeline ===")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"{len(labels)} Labels: {labels}")
        print(f"Output directory: {output_dir}")
        print(f"Similarity threshold: {similarity_threshold}")
        print(f"Temperature: {softmax_temp}")
        print(f"DBSCAN parameters: eps={dbscan_eps}, min_samples={dbscan_min_samples}")
        
        # Step 1: Encode text labels
        print("\n1. Encoding text labels...")
        text_embeddings = self.text_encoder.encode_text_labels(labels)
        
        # Step 2: Compute similarities
        print("2. Computing Gaussian similarities...")
        similarities = compute_gaussian_similarities(
            model=self.model_loader.model,
            text_embeddings=text_embeddings,
            main_feature_name=self.model_loader.main_feature_name,
            batch_size=batch_size,
            softmax_temp=softmax_temp
        )
        
        # Step 3: Perform clustering
        print("3. Performing spatial clustering...")
        gaussian_positions = get_gaussian_positions(self.model_loader.model).numpy()
        results = self.clusterer.cluster_by_label(
            similarities=similarities,
            labels=labels,
            gaussian_positions=gaussian_positions,
            similarity_threshold=similarity_threshold,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples
        )
        
        # Step 4: Save results
        print("4. Saving results...")
        clustering_metadata = {
            "checkpoint": checkpoint_path,
            "labels": labels,
            "labels_file": labels_file,
            "output_dir": output_dir,
            "similarity_threshold": similarity_threshold,
            "dbscan_eps": dbscan_eps,
            "dbscan_min_samples": dbscan_min_samples,
            "batch_size": batch_size,
            "softmax_temp": softmax_temp,
        }
        
        save_all_results(results, labels, output_dir, clustering_metadata)
        
        # Step 5: Print summary
        print("\n=== Clustering Complete ===")
        for label in labels:
            if label in results:
                result = results[label]
                if "num_clusters" in result:
                    print(f"{label}: {result['num_clusters']} clusters from {result['num_candidates']} candidates")
                else:
                    print(
                        f"{label}: No clustering performed - {result['num_candidates']} candidates (minimum required: {dbscan_min_samples})"
                    )
        
        return results
