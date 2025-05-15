import open3d as o3d
import numpy as np

def render_as_png(pcd):
    width, height = 1024, 768

    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])  # white

    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"

    renderer.scene.add_geometry("pcd", pcd, material)
    bounds = pcd.get_axis_aligned_bounding_box()
    center = bounds.get_center()
    extent = bounds.get_extent().max()

    renderer.setup_camera(60.0, bounds, center)
    img = renderer.render_to_image()
    o3d.io.write_image("render.png", img)




ply_path = "../data/nerfstudio/garden_8/sparse_pc.ply"
pcd = o3d.io.read_point_cloud(ply_path)

print(pcd)
print("Number of points:", len(pcd.points))

render_as_png(pcd)

