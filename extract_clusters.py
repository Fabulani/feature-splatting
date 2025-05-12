from feature_splatting.model import FeatureSplattingModel, FeatureSplattingModelConfig
import yaml

# Replace with the path to your config YAML file
config_path = "data/nerfstudio/garden_8/garden_8/feature-splatting/2025-05-12_071531/config.yml"



# Load config YAML
with open(config_path, "r") as f:
    config_dict = yaml.safe_load(f)

print("CONFIG DICT HERE:", config_dict)

# Instantiate your custom config
config = FeatureSplattingModelConfig(**config_dict["pipeline"]["_target_config"])

# Initialize the model
model = FeatureSplattingModel(config)
model.populate_modules()



# # Load the configuration
# config = tyro.cli(method_configs["feature-splatting"], args=["--load-config", config_path])



# # Create an instance of FeatureSplattingModel
# model = FeatureSplattingModel(config=config)

# # Initialize the model's modules (if required)
# model.populate_modules()

clusters = model.extract_clusters(field_name='positive', threshold=0.6)

# Save the clustered information to a text file
with open("clusters_info.txt", "w") as file:
    file.write("Clustered Points:\n")
    file.write(str(clusters['clustered_points']) + "\n")
    file.write("Indices:\n")
    file.write(str(clusters['indices']) + "\n")
    file.write("Bounding Box:\n")
    file.write(str(clusters['bounding_box']) + "\n")