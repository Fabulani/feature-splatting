import open3d as o3d
import numpy as np

import matplotlib.pyplot as plt


def render_as_png(pcd):
    points = np.asarray(pcd.points)

    # downsample
    if len(points) > 20000:
        pcd = pcd.farthest_point_down_sample(20000)
        points = np.asarray(pcd.points)

    # 2D projection 
    plt.figure(figsize=(8, 6))
    plt.scatter(points[:, 0], points[:, 1], s=0.1, c=points[:, 2], cmap='viridis')
    plt.axis("equal")
    plt.title("Render of the point cloud")
    plt.savefig("render.png", dpi=300)


ply_path = "../data/nerfstudio/garden_8/sparse_pc.ply"
pcd = o3d.io.read_point_cloud(ply_path)

print(pcd)
print("Number of points:", len(pcd.points))

o3d.io.write_point_cloud("custom_pc.ply", pcd, print_progress=True)


render_as_png(pcd)

