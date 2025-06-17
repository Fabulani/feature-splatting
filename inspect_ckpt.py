"""Script to inspect a nerfstudio checkpoint and print details about the model state, Gaussian parameters, and training info."""

import argparse
from pathlib import Path

import torch


def inspect_checkpoint(checkpoint_path, device="cpu"):
    """Inspect a nerfstudio checkpoint and print detailed information.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the checkpoint on
    """
    checkpoint_path = Path(checkpoint_path)
    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    print("=== CHECKPOINT CONTENTS ===")
    print("Keys in checkpoint:")
    for key in checkpoint.keys():
        print(f"  {key}")

    print("\n=== MODEL STATE DICT ===")
    model_state = checkpoint["pipeline"]

    print("Keys in pipeline state:")
    for key in model_state.keys():
        if hasattr(model_state[key], "shape"):
            print(f"  {key}: {model_state[key].shape}")
        else:
            print(f"  {key}: {type(model_state[key])}")

    print("\n=== GAUSSIAN PARAMETERS ===")
    if "_model.gauss_params.means" in model_state:
        num_gaussians = model_state["_model.gauss_params.means"].shape[0]
        print(f"  count: {num_gaussians:,}")

        gaussian_params = ["means", "scales", "quats", "opacities", "features_dc", "features_rest", "distill_features"]
        for param in gaussian_params:
            param_key = f"_model.gauss_params.{param}"
            if param_key in model_state:
                tensor = model_state[param_key]
                print(f"  {param}: {tensor.shape} (dtype: {tensor.dtype})")
            else:
                print(f"  {param}: NOT FOUND")
    else:
        print("  No Gaussian parameters found in checkpoint")

    print("\n=== TRAINING INFO ===")
    if "step" in checkpoint:
        print(f"  Training step: {checkpoint['step']:,}")
    else:
        print("  No training step info found")

    print("\n=== SUMMARY ===")
    total_params = sum(p.numel() for p in model_state.values() if hasattr(p, "numel"))
    bytes_per_param = 4  # Assuming float32
    to_mb = 1 / (1024**2)
    total_memory_mb = total_params * bytes_per_param * to_mb

    # distill_features contribution
    distill_key = "_model.gauss_params.distill_features"
    if distill_key in model_state:
        distill_params = model_state[distill_key].numel()
        distill_percentage = (distill_params / total_params) * 100
        distill_mb = distill_params * bytes_per_param * to_mb

        print(f"Total parameters: {total_params:,}")
        print(f"Distill features: {distill_params:,} ({distill_percentage:.1f}% of total)")
        print(f"Distill features memory: {distill_mb:.2f} MB")
        print(f"Rough total memory estimation: {total_memory_mb:.2f} MB (float32)")
    else:
        print(f"Total parameters: {total_params:,}")
        print(f"Rough total memory estimation: {total_memory_mb:.2f} MB (float32)")
        print("No distill features found")


def main():
    parser = argparse.ArgumentParser(description="Inspect nerfstudio checkpoint details")
    parser.add_argument("checkpoint_path", help="Path to the checkpoint file (.ckpt)")

    args = parser.parse_args()

    # Auto-detect device if not specified
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        inspect_checkpoint(args.checkpoint_path, device)
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found: {args.checkpoint_path}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")


if __name__ == "__main__":
    main()
