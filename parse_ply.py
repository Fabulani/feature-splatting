import numpy as np
import open3d as o3d
from tqdm import tqdm

# Load ply with open3d
ply_path = "../data/nerfstudio/garden_8/sparse_pc.ply"
pcd = o3d.io.read_point_cloud(ply_path)

# Then dump the points into a numpy array
points = np.asarray(pcd.points)
print(points)

# This will be the class_id array. Every point must have at least one class_id. Assume `0.0` as "no class".
class_ids = np.zeroes(len(points))

# Dummy values to test. Test ply has >40k points
class_ids[:10000] = [0.0]
class_ids[10000:20000] = [1.0]
class_ids[20000:30000] = [1.0, 2.0]
class_ids[30000:] = [0.0]


def standard_properties(point):
    return f"{point[0]} {point[1]} {point[2]}"

def class_properties(class_id):
    property_string = f"{len(class_id)}"
    for cid in class_id:
        property_string += f" {cid}"
    return property_string

file_name = "with_classes.ply"
with open(file_name, "w") as f:
    f.write("ply\n")
    f.write("format ascii 1.0\n")
    f.write(f"element vertex {len(points)}\n")
    f.write("property float x\n")
    f.write("property float y\n")
    f.write("property float z\n")
    f.write("property list uchar float class_id\n")
    f.write("end_header\n")

    for pt, cid in tqdm(zip(points, class_id), total=len(points)):
        std_props = standard_properties(pt)
        class_props = class_properties(cid)
        f.write(f"{std_props} {class_props}\n")

print(f"Saved to {file_name}.")