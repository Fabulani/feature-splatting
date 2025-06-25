"""
Feature Splatting Clustering - CLI Interface

This script provides a command-line interface for clustering 3D Gaussians from a 
feature splatting model checkpoint based on their semantic similarity to text labels.

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
import traceback

from feature_splatting.clustering import ClusteringPipeline
from feature_splatting.clustering.io_utils import read_labels_file


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
            file_labels = read_labels_file(args.labels_file)
            all_labels.extend(file_labels)
            print(f"Loaded {len(file_labels)} labels from {args.labels_file}")
        except FileNotFoundError:
            print(f"Error: Labels file not found: {args.labels_file}")
            return
        except IOError as e:
            print(f"Error reading labels file: {e}")
            return

    # Check if there's at least one label
    if not all_labels:
        print("Error: No labels provided. Use --labels or --labels-file to specify labels.")
        return

    # Remove duplicates
    unique_labels = list(set(all_labels))

    try:
        pipeline = ClusteringPipeline(args.checkpoint)
        results = pipeline.run(
            labels=unique_labels,
            output_dir=args.output_dir,
            similarity_threshold=args.similarity_threshold,
            dbscan_eps=args.dbscan_eps,
            dbscan_min_samples=args.dbscan_min_samples,
            batch_size=args.batch_size,
            softmax_temp=args.softmax_temp,
            checkpoint_path=args.checkpoint,
            labels_file=args.labels_file,
        )

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
