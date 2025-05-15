import open3d as o3d
import numpy as np

import matplotlib.pyplot as plt

def render_as_png(pcd):
    # width, height = 1024, 768

    # renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    # renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])  # white

    # material = o3d.visualization.rendering.MaterialRecord()
    # material.shader = "defaultUnlit"

    # renderer.scene.add_geometry("pcd", pcd, material)
    # bounds = pcd.get_axis_aligned_bounding_box()
    # center = bounds.get_center()
    # extent = bounds.get_extent().max()

    # renderer.setup_camera(60.0, bounds, center)
    # img = renderer.render_to_image()
    # o3d.io.write_image("render.png", img)

    points = np.asarray(pcd.points)

    # downsample
    if len(points) > 30000:
        pcd = pcd.farthest_point_down_sample(30000)
        points = np.asarray(pcd.points)

    # 2D projection - ortho xy view
    plt.figure(figsize=(8, 6))
    plt.scatter(points[:, 0], points[:, 1], s=0.2, c=points[:, 2], cmap='viridis')
    plt.axis("equal")
    plt.title("XY Projection of Point Cloud")
    plt.savefig("render.png", dpi=300)





ply_path = "../data/nerfstudio/garden_8/sparse_pc.ply"
pcd = o3d.io.read_point_cloud(ply_path)

print(pcd)
print("Number of points:", len(pcd.points))

render_as_png(pcd)

