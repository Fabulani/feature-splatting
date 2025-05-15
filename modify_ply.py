import open3d as o3d
import numpy as np

import matplotlib.pyplot as plt

def rotate_point_cloud(points, angles_degrees):
    """Rotate point cloud by given angles (in degrees) around x, y, z."""
    angles = np.radians(angles_degrees)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]),  np.cos(angles[0])]])
    
    Ry = np.array([[ np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]),  np.cos(angles[2]), 0],
                   [0, 0, 1]])
    
    R = Rz @ Ry @ Rx  
    return points @ R.T

def render_as_png(pcd):
    points = np.asarray(pcd.points)

    # rotate
    points = rotate_point_cloud(points, angles_degrees=(45, 45, 45))


    # downsample
    if len(points) > 30000:
        pcd = pcd.farthest_point_down_sample(30000)
        points = np.asarray(pcd.points)

    # 2D projection 
    plt.figure(figsize=(8, 6))
    plt.scatter(points[:, 0], points[:, 1], s=0.2, c=points[:, 2], cmap='viridis')
    plt.axis("equal")
    plt.title("Render of the point cloud")
    plt.savefig("render.png", dpi=300)





ply_path = "../data/nerfstudio/garden_8/sparse_pc.ply"
pcd = o3d.io.read_point_cloud(ply_path)

print(pcd)
print("Number of points:", len(pcd.points))

render_as_png(pcd)

