import torch
import yaml
from pathlib import Path

checkpoint_path = Path("data/checkpoint.ckpt")
print(f"Loading checkpoint: {checkpoint_path}")

# Load the checkpoint
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print("=== CHECKPOINT CONTENTS ===")
print("Keys in checkpoint:")
for key in checkpoint.keys():
    print(f"  {key}")

print("\n=== MODEL STATE DICT ===")
model_state = checkpoint['pipeline']
print("Keys in pipeline state:")
for key in model_state.keys():
    if hasattr(model_state[key], 'shape'):
        print(f"  {key}: {model_state[key].shape}")
    else:
        print(f"  {key}: {type(model_state[key])}")

print("\n=== GAUSSIAN PARAMETERS ===")
num_gaussians = model_state["_model.gauss_params.means"].shape[0]
print(f"  count: {num_gaussians}")

gaussian_params = ['means', 'scales', 'quats', 'opacities', 'features_dc', 'features_rest', 'distill_features']
for param in gaussian_params:
    param_key = f"_model.gauss_params.{param}"
    if param_key in model_state:
        tensor = model_state[param_key]
        print(f"  {param}: {tensor.shape} (dtype: {tensor.dtype})")


print("\n=== TRAINING INFO ===")
print(f"  Training step: {checkpoint['step']}")



print(f"\n=== SUMMARY ===")
total_params = sum(p.numel() for p in model_state.values() if hasattr(p, 'numel'))
bytes_per_param = 4  # Assuming float32
to_mb = 1 / (1024**2)
total_memory_mb = total_params * bytes_per_param * to_mb

# distill_features contribution
distill_key = "_model.gauss_params.distill_features"
distill_params = model_state[distill_key].numel()
distill_percentage = (distill_params / total_params) * 100
distill_mb = distill_params * bytes_per_param * to_mb

print(f"Total parameters: {total_params:,}")
print(f"Distill features: {distill_params:,} ({distill_percentage:.1f}% of total)")
print(f"Distill features memory: {distill_mb:.2f} MB")
print(f"Rough total memory estimation: {total_memory_mb:.2f} MB (float32)")

