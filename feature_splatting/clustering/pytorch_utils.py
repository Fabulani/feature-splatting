"""
PyTorch utilities for model loading and feature processing.

This module handles PyTorch-specific operations including checkpoint loading,
model initialization, text encoding, and Gaussian feature processing.
"""

from pathlib import Path

import torch
from feature_splatting.model import FeatureSplattingModel, FeatureSplattingModelConfig
from feature_splatting.utils.clip_text_encoder import clip_text_encoder
from tqdm import tqdm

from nerfstudio.data.scene_box import SceneBox


class ModelLoader:
    """Handles loading and initializing the feature splatting model from checkpoints."""

    # Config values hardcoded from the standard feature splatting model
    DEFAULT_FEAT_LATENT_DIM = 13
    DEFAULT_MLP_HIDDEN_DIM = 64
    DEFAULT_SH_DEGREE = 0
    DEFAULT_FEATURE_TYPE = "samclip"
    DEFAULT_BACKGROUND_COLOR = "random"
    CLIP_MODEL_NAME = "ViT-L/14@336px"

    FEATURE_DIMENSIONS = {
        "samclip": (768, 41, 64),
        "dinov2": (384, 37, 57),
    }

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        """
        Initialize the ModelLoader and load the model.

        Args:
            checkpoint_path: Path to the feature splatting checkpoint file
            device: Device to load the model on (e.g., "cuda" or "cpu")
        """
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)
        self.model = None
        self.main_feature_name = None

        print(f"Initializing ModelLoader with checkpoint: {self.checkpoint_path}")
        self._load_model()

    def _load_model(self):
        """
        Load the feature splatting model from a checkpoint (.ckpt).

        This method handles loading the checkpoint and reconstructing the model
        with the correct parameters and weights.
        """
        print("Loading checkpoint...")

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
        metadata["clip_model_name"] = self.CLIP_MODEL_NAME

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


class TextEncoder:
    """Handles CLIP text encoding operations."""

    def __init__(self, clip_model_name: str = "ViT-L/14@336px"):
        """
        Initialize the TextEncoder.

        Args:
            clip_model_name: Name of the CLIP model to use for text encoding.
        """
        self.clip_model_name = clip_model_name
        self.device = "cuda"  # CLIP text encoder requires GPU
        self.text_encoder = clip_text_encoder(self.clip_model_name, device=self.device)

        print(f"Text encoder initialized successfully: CLIP {self.clip_model_name}")

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
    model: FeatureSplattingModel,
    text_embeddings: torch.Tensor,
    main_feature_name: str,
    batch_size: int | None = None,
    softmax_temp: float = 1.0,
) -> torch.Tensor:
    """
    Compute similarity scores between ALL Gaussians and text embeddings using batch processing.

    Args:
        model: The loaded feature splatting model
        text_embeddings: Text embeddings of shape [num_labels, embedding_dim]
        main_feature_name: Name of the main feature type (e.g., "samclip")
        batch_size: Optional batch size for processing Gaussians (for memory efficiency)
        softmax_temp: Temperature for softmax computation

    Returns:
        similarities: Similarity scores of shape [num_gaussians, num_labels] for ALL Gaussians
    """
    print("Computing Gaussian-text similarities...")

    device = next(model.parameters()).device
    num_gaussians = model.means.shape[0]
    num_labels = text_embeddings.shape[0]

    if batch_size is None:
        batch_size = num_gaussians
        print(f"Processing all {num_gaussians:,} Gaussians at once")
    else:
        print(f"Processing all {num_gaussians:,} Gaussians in batches of {batch_size:,}")

    # Initialize results tensor to store all similarities
    all_similarities = torch.zeros(num_gaussians, num_labels, device=device)

    # Process Gaussians in batches
    for start_idx in tqdm(range(0, num_gaussians, batch_size), desc="Processing batches"):
        end_idx = min(start_idx + batch_size, num_gaussians)
        batch_indices = torch.arange(start_idx, end_idx, device=device)
        batch_distill_features = model.distill_features[batch_indices]

        # Process distilled features through the feature MLP to get CLIP features
        with torch.no_grad():
            # Use the per_gaussian_forward method for efficiency
            feature_dict = model.feature_mlp.per_gaussian_forward(batch_distill_features)
            clip_features = feature_dict[main_feature_name]  # [batch_size, feature_dim]

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


def get_gaussian_positions(model: FeatureSplattingModel) -> torch.Tensor:
    """
    Get the 3D positions of all Gaussians.

    Args:
        model: The loaded feature splatting model

    Returns:
        Tensor of shape [num_gaussians, 3] containing Gaussian positions
    """
    return model.means.detach().cpu()
