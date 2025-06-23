"""
Feature Splatting Clustering Script

This script clusters 3D Gaussians from a feature splatting model checkpoint based on their
semantic similarity to text labels. It uses DBSCAN for spatial clustering and CLIP for
text-to-feature similarity computation.

USAGE:
    Basic usage:
        python cluster_features.py data/checkpoint.ckpt --labels table flower vase

    Using labels from a file:
        python cluster_features.py data/checkpoint.ckpt --labels-file labels.txt

    Combining both command-line and file labels:
        python cluster_features.py data/checkpoint.ckpt --labels table chair --labels-file more_labels.txt

    Advanced usage with custom parameters:
        python cluster_features.py data/checkpoint.ckpt \
            --labels table flower vase floor grass \
            --labels-file labels.txt \
            --similarity-threshold 0.25 \
            --dbscan-eps 0.2 \
            --dbscan-min-samples 100 \
            --softmax-temp 2.0 \
            --batch-size 25000 \
            --output-dir my_results

    Example labels.txt file format:
        # This is a comment line (ignored)
        table
        chair
        vase
        # Another comment
        flower
        flowervase on top of the table
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from feature_splatting.model import FeatureSplattingModel, FeatureSplattingModelConfig
from feature_splatting.utils.clip_text_encoder import clip_text_encoder
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from nerfstudio.data.scene_box import SceneBox


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


class FeatureClusterer:
    """
    Handles clustering of Gaussians based on text labels and a feature splatting model.

    This class loads a trained feature splatting model from a checkpoint, then clusters
    Gaussians based on their spatial distance and semantic similarity to provided text labels.
    """

    # Config values hardcoded from the standard feature splatting model
    DEFAULT_FEAT_LATENT_DIM = 13
    DEFAULT_MLP_HIDDEN_DIM = 64
    DEFAULT_SH_DEGREE = 0
    DEFAULT_FEATURE_TYPE = "samclip"
    DEFAULT_BACKGROUND_COLOR = "random"

    FEATURE_DIMENSIONS = {
        "samclip": (768, 41, 64),
        "dinov2": (384, 37, 57),
    }

    def __init__(self, checkpoint_path: str):
        """
        Initialize the FeatureClusterer.

        Args:
            checkpoint_path: Path to the feature splatting checkpoint file
        """
        self.device = "cuda"  # CLIP text encoder requires GPU, so always use CUDA.
        self.checkpoint_path = Path(checkpoint_path)
        self.model = None
        self.text_encoder = None

        print(f"Initializing FeatureClusterer with checkpoint: {self.checkpoint_path}")

        # Text encoder must be initialized before the model
        self._initialize_text_encoder()
        self._load_model()

    def _load_model(self):
        """
        Load the feature splatting model from a checkpoint (.ckpt).

        This method handles loading the checkpoint and reconstructing the model
        with the correct parameters and weights.
        """
        print("Loading checkpoint...")

        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")

        if "pipeline" not in checkpoint:
            raise ValueError("Invalid checkpoint: missing 'pipeline' key")

        model_state = checkpoint["pipeline"]

        num_gaussians = model_state["_model.gauss_params.means"].shape[0]
        feat_latent_dim = model_state["_model.gauss_params.distill_features"].shape[1]

        print(f"Found {num_gaussians:,} Gaussians with {feat_latent_dim}D distilled features")

        config = FeatureSplattingModelConfig(
            feat_latent_dim=feat_latent_dim,
            mlp_hidden_dim=self.DEFAULT_MLP_HIDDEN_DIM,
            sh_degree=self.DEFAULT_SH_DEGREE,
            background_color=self.DEFAULT_BACKGROUND_COLOR,
        )
        main_feature_name = self.DEFAULT_FEATURE_TYPE

        # Create a minimal scene box
        # This doesn't affect clustering, but is necessary for model setup
        scene_box = SceneBox(aabb=torch.tensor([[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]], dtype=torch.float32))

        # Set up model metadata
        feature_dim_dict = self.FEATURE_DIMENSIONS
        metadata = {
            "main_feature_name": main_feature_name,
            "feature_dim_dict": feature_dim_dict,
            "feature_type": main_feature_name.upper(),
        }

        # Trigger the model to use its built-in text encoder (see `maybe_populate_text_encoder()` in model.py)
        # Must be the same as the model used during training!
        metadata["clip_model_name"] = "ViT-L/14@336px"

        self.model = FeatureSplattingModel(
            config=config,
            scene_box=scene_box,
            num_train_data=1000,  # Dummy value, not used for inference
            **{"metadata": metadata, "device": self.device},
        )

        self.model.text_encoding_func = self.text_encoder.get_text_token

        # The checkpoint saves model parameters with "_model." prefix, but load_state_dict expects them without it
        cleaned_state_dict = {}
        for key, value in model_state.items():
            if key.startswith("_model."):
                new_key = key[7:]
                cleaned_state_dict[new_key] = value
            else:
                cleaned_state_dict[key] = value

        self.model.load_state_dict(cleaned_state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.main_feature_name = main_feature_name

        print(f"Model loaded successfully (main_feature: {main_feature_name})")
        print(f"Feature dimensions: {feature_dim_dict}")

    def _initialize_text_encoder(self):
        """
        Initialize the text encoder for processing text labels.

        This must use same CLIP model that was used during training
        to ensure consistency in the embedding space.
        """
        clip_model_name = "ViT-L/14@336px"
        self.text_encoder = clip_text_encoder(clip_model_name, device=self.device)
        print(f"Text encoder initialized successfully: CLIP {clip_model_name}")

    def encode_text_labels(self, labels: list[str]) -> torch.Tensor:
        """
        Encode text labels into embedding vectors.

        Args:
            labels: List of text labels (e.g., ["table", "chair", "vase"])

        Returns:
            Tensor of shape [num_labels, embedding_dim] containing text embeddings
        """
        print(f"Encoding {len(labels)} text labels...")

        embeddings = []
        for label in labels:
            embedding = self.text_encoder.get_text_token([label])
            embeddings.append(embedding.squeeze(0))  # Remove batch dimension

        text_embeddings = torch.stack(embeddings, dim=0)  # [num_labels, embedding_dim]

        print(f"Text embeddings shape: {text_embeddings.shape}")
        return text_embeddings

    def compute_gaussian_similarities(
        self, text_embeddings: torch.Tensor, batch_size: int | None = None, softmax_temp: float = 1.0
    ) -> torch.Tensor:
        """
        Compute similarity scores between ALL Gaussians and text embeddings using batch processing.

        Args:
            text_embeddings: Text embeddings of shape [num_labels, embedding_dim]
            batch_size: Optional batch size for processing Gaussians (for memory efficiency)
            softmax_temp: Temperature for softmax computation

        Returns:
            similarities: Similarity scores of shape [num_gaussians, num_labels] for ALL Gaussians
        """
        print("Computing Gaussian-text similarities...")

        num_gaussians = self.model.means.shape[0]
        num_labels = text_embeddings.shape[0]

        if batch_size is None:
            batch_size = num_gaussians
            print(f"Processing all {num_gaussians:,} Gaussians at once")
        else:
            print(f"Processing all {num_gaussians:,} Gaussians in batches of {batch_size:,}")

        # Initialize results tensor to store all similarities
        all_similarities = torch.zeros(num_gaussians, num_labels, device=self.device)

        # Process Gaussians in batches
        for start_idx in tqdm(range(0, num_gaussians, batch_size), desc="Processing batches"):
            end_idx = min(start_idx + batch_size, num_gaussians)
            batch_indices = torch.arange(start_idx, end_idx, device=self.device)
            batch_distill_features = self.model.distill_features[batch_indices]

            # Process distilled features through the feature MLP to get CLIP features
            with torch.no_grad():
                # Use the per_gaussian_forward method for efficiency
                feature_dict = self.model.feature_mlp.per_gaussian_forward(batch_distill_features)
                clip_features = feature_dict[self.main_feature_name]  # [batch_size, feature_dim]

                # Normalize
                clip_features = clip_features / clip_features.norm(dim=1, keepdim=True)

                # Compute similarities: [batch_size, num_labels]
                raw_similarities = torch.mm(clip_features, text_embeddings.T)

                # Apply softmax temperature
                raw_similarities = raw_similarities / softmax_temp

                # Store results
                all_similarities[start_idx:end_idx] = raw_similarities

        print(f"Computed similarities shape: {all_similarities.shape}")
        return all_similarities

    def cluster_by_label(
        self,
        similarities: torch.Tensor,
        labels: list[str],
        similarity_threshold: float = 0.5,
        dbscan_eps: float = 0.1,
        dbscan_min_samples: int = 20,
    ) -> dict[str, dict]:
        """
        Cluster Gaussians for each text label using DBSCAN.

        Args:
            similarities: Similarity scores of shape [num_gaussians, num_labels]
            labels: List of text labels
            similarity_threshold: Minimum similarity score to consider a Gaussian for a label
            dbscan_eps: DBSCAN epsilon parameter (neighborhood radius)
            dbscan_min_samples: DBSCAN minimum samples parameter

        Returns:
            Dictionary containing clustering results for each label
        """
        print(f"Clustering Gaussians for {len(labels)} labels...")

        results = {}
        gaussian_positions = self.model.means.detach().cpu().numpy()
        all_indices = torch.arange(gaussian_positions.shape[0])

        for i, label in enumerate(labels):
            print(f"\nProcessing label '{label}'...")

            # Get similarity scores for this label
            label_similarities = similarities[:, i].cpu().numpy()

            max_sim = np.max(label_similarities)
            median_sim = np.median(label_similarities)
            mean_sim = np.mean(label_similarities)
            min_sim = np.min(label_similarities)
            std_sim = np.std(label_similarities)

            print(f"  Similarity statistics for '{label}':")
            print(f"    Highest: {max_sim:.4f}")
            print(f"    Median:  {median_sim:.4f}")
            print(f"    Average: {mean_sim:.4f}")
            print(f"    Lowest:  {min_sim:.4f}")
            print(f"    Std Dev: {std_sim:.4f}")

            # Filter by similarity threshold
            high_similarity_mask = label_similarities > similarity_threshold
            num_candidates = high_similarity_mask.sum()

            print(f"  Found {num_candidates} Gaussians above threshold {similarity_threshold}")

            if num_candidates < dbscan_min_samples:
                print(f"  !  Not enough candidates for clustering (minimum: {dbscan_min_samples})")
                results[label] = {
                    "num_candidates": num_candidates,
                    "clusters": [],
                    "cluster_info": None,
                    "similarity_statistics": {
                        "max": float(max_sim),
                        "median": float(median_sim),
                        "mean": float(mean_sim),
                        "min": float(min_sim),
                        "std_dev": float(std_sim),
                    },
                }
                continue

            # Get positions of high-similarity Gaussians
            candidate_positions = gaussian_positions[high_similarity_mask]
            candidate_indices = all_indices[high_similarity_mask].cpu().numpy()
            candidate_similarities = label_similarities[high_similarity_mask]

            # Apply DBSCAN clustering
            dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
            cluster_labels = dbscan.fit_predict(candidate_positions)

            unique_clusters, cluster_counts = np.unique(cluster_labels[cluster_labels != -1], return_counts=True)
            num_clusters = len(unique_clusters)
            num_noise = (cluster_labels == -1).sum()

            print(f"  Found {num_clusters} clusters, {num_noise} noise points")

            # Organize results by cluster
            clusters = []
            for cluster_id in unique_clusters:
                cluster_mask = cluster_labels == cluster_id
                cluster_data = {
                    "cluster_id": int(cluster_id),
                    "size": int(cluster_counts[cluster_id == unique_clusters][0]),
                    "gaussian_indices": candidate_indices[cluster_mask].tolist(),
                    "positions": candidate_positions[cluster_mask].tolist(),
                    "similarities": candidate_similarities[cluster_mask].tolist(),
                    "mean_similarity": float(candidate_similarities[cluster_mask].mean()),
                    "centroid": candidate_positions[cluster_mask].mean(axis=0).tolist(),
                }
                clusters.append(cluster_data)

            # Sort clusters by size (largest first)
            clusters.sort(key=lambda x: x["size"], reverse=True)

            results[label] = {
                "num_candidates": num_candidates,
                "num_clusters": num_clusters,
                "num_noise_points": num_noise,
                "clusters": clusters,
                "cluster_info": {
                    "similarity_threshold": similarity_threshold,
                    "dbscan_eps": dbscan_eps,
                    "dbscan_min_samples": dbscan_min_samples,
                },
                "similarity_statistics": {
                    "max": float(max_sim),
                    "median": float(median_sim),
                    "mean": float(mean_sim),
                    "min": float(min_sim),
                    "std_dev": float(std_sim),
                },
            }
        return results

    def save_results(
        self,
        results: dict,
        labels: list[str],
        output_dir: str = "clustering_results",
        clustering_metadata: dict | None = None,
    ):
        """
        Save clustering results to files.

        Args:
            results: Clustering results dictionary
            labels: List of text labels
            output_dir: Directory to save results
            clustering_metadata: Clustering parameters to save as metadata
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print(f"\nSaving results to {output_path}/")

        # Save complete results as JSON (with custom encoder for numpy types)
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        # Add clustering parameters as metadata
        if clustering_metadata:
            results["_metadata"] = clustering_metadata

        with open(output_path / "clustering_results.json", "w") as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        print("  Saved clustering_results.json")

        # Save summary statistics
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
        print("  Saved summary.json")

        # Save individual cluster files for each label
        for label in labels:
            if label in results and results[label]["clusters"]:
                label_results = results[label]
                rows = []

                for cluster in label_results["clusters"]:
                    for i, (pos, sim, idx) in enumerate(
                        zip(cluster["positions"], cluster["similarities"], cluster["gaussian_indices"])
                    ):
                        rows.append(
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

                if rows:
                    df = pd.DataFrame(rows)
                    df.to_csv(output_path / f"{label}_clusters.csv", index=False)
                    print(f"  Saved {label}_clusters.csv ({len(rows)} points)")


def main():
    """
    CLI interface.
    """
    parser = argparse.ArgumentParser(description="Cluster feature splatting Gaussians by text labels")
    parser.add_argument("checkpoint", help="Path to feature splatting checkpoint (.ckpt)")
    parser.add_argument(
        "--labels", nargs="*", default=[], help="Text labels for clustering (e.g., --labels table chair vase)"
    )
    parser.add_argument(
        "--labels-file",
        type=str,
        help="Path to text file containing labels (one per line). Lines starting with # are ignored as comments. Labels will be combined with --labels argument.",
    )
    parser.add_argument("--output-dir", default="clustering_results", help="Output directory for results")
    parser.add_argument(
        "--similarity-threshold", type=float, default=0.5, help="Minimum similarity threshold for considering Gaussians"
    )
    parser.add_argument("--dbscan-eps", type=float, default=0.1, help="DBSCAN epsilon parameter (neighborhood radius)")
    parser.add_argument("--dbscan-min-samples", type=int, default=100, help="DBSCAN minimum samples parameter")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for processing Gaussians (for memory efficiency). 50k Gaussians use approximately 1GB of GPU memory. By default, will process all at once.",
    )
    parser.add_argument(
        "--softmax-temp",
        type=float,
        default=0.5,
        help="Temperature for similarity computation (higher=softer, lower=sharper, default=0.5)",
    )

    args = parser.parse_args()

    # Combine labels from --labels argument and --labels-file
    all_labels = list(args.labels) if args.labels else []

    if args.labels_file:
        try:
            labels_file_path = Path(args.labels_file)
            with open(labels_file_path, "r", encoding="utf-8") as f:
                file_labels = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
            all_labels.extend(file_labels)
            print(f"Loaded {len(file_labels)} labels from {labels_file_path}")
        except FileNotFoundError:
            print(f"Error: Labels file not found: {args.labels_file}")
            return
        except Exception as e:
            print(f"Error reading labels file: {e}")
            return

    # Check if there's at least one label
    if not all_labels:
        print("Error: No labels provided. Use --labels or --labels-file to specify labels.")
        return

    # Remove duplicates
    unique_labels = list(set(all_labels))

    print(f"Total unique labels to process: {len(unique_labels)}")

    print("=== Feature Splatting Clustering ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Labels: {unique_labels}")
    print(f"Output directory: {args.output_dir}")
    print(f"Similarity threshold: {args.similarity_threshold}")
    print(f"Temperature: {args.softmax_temp}")
    print(f"DBSCAN parameters: eps={args.dbscan_eps}, min_samples={args.dbscan_min_samples}")

    try:
        clusterer = FeatureClusterer(args.checkpoint)
        text_embeddings = clusterer.encode_text_labels(unique_labels)

        similarities = clusterer.compute_gaussian_similarities(
            text_embeddings, batch_size=args.batch_size, softmax_temp=args.softmax_temp
        )

        results = clusterer.cluster_by_label(
            similarities,
            unique_labels,
            similarity_threshold=args.similarity_threshold,
            dbscan_eps=args.dbscan_eps,
            dbscan_min_samples=args.dbscan_min_samples,
        )

        clustering_metadata = {
            "checkpoint": args.checkpoint,
            "labels": unique_labels,
            "labels_file": args.labels_file,
            "output_dir": args.output_dir,
            "similarity_threshold": args.similarity_threshold,
            "dbscan_eps": args.dbscan_eps,
            "dbscan_min_samples": args.dbscan_min_samples,
            "batch_size": args.batch_size,
            "softmax_temp": args.softmax_temp,
        }

        clusterer.save_results(results, unique_labels, args.output_dir, clustering_metadata)

        print("\n=== Clustering Complete ===")
        for label in unique_labels:
            if label in results:
                result = results[label]
                if "num_clusters" in result:
                    print(f"{label}: {result['num_clusters']} clusters from {result['num_candidates']} candidates")
                else:
                    print(
                        f"{label}: No clustering performed - {result['num_candidates']} candidates (minimum required: {args.dbscan_min_samples})"
                    )

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
