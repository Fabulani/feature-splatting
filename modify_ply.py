import open3d as o3d

ply_path = "../data/.fabiano/nerfstudio/garden_8/sparse_pc.ply"
pcd = o3d.io.read_point_cloud(ply_path)

print(pcd)
print("Number of points:", len(pcd.points))

o3d.visualization.draw_geometries([pcd])

