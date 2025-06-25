"""
Feature Splatting Clustering Module

This module provides tools for clustering 3D Gaussians from feature splatting models
based on semantic similarity to text labels.

Main components:
- pipeline: Main clustering pipeline orchestration
- pytorch_utils: Model loading and PyTorch operations  
- clustering_utils: Spatial clustering algorithms (DBSCAN)
- io_utils: File I/O operations for results and labels
"""

from .pipeline import ClusteringPipeline

__all__ = ["ClusteringPipeline"]
