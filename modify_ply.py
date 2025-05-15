import open3d as o3d

def render_as_png(pcd):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image('f{i}.png', do_render=True)
    vis.destroy_window()


ply_path = "../data/.fabiano/nerfstudio/garden_8/sparse_pc.ply"
pcd = o3d.io.read_point_cloud(ply_path)

print(pcd)
print("Number of points:", len(pcd.points))

render_as_png(pcd)

