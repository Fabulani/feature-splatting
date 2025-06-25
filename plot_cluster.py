""" Visualize clustered points and bounding box using plotly. Requires a clusters_info.csv file and clusters_bbox.json files generated from the extract_clusters() method in feature_splatting/model.py script (using the nerfstudio interface). """

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import webbrowser
import json

df = pd.read_csv("clusters_info.csv")

with open("clusters_bbox.json", "r") as f:
    bbox = json.load(f)

min_corner = np.array([bbox["bbox_min_x"], bbox["bbox_min_y"], bbox["bbox_min_z"]])
max_corner = np.array([bbox["bbox_max_x"], bbox["bbox_max_y"], bbox["bbox_max_z"]])

corners = np.array([
    [x, y, z]
    for x in [min_corner[0], max_corner[0]]
    for y in [min_corner[1], max_corner[1]]
    for z in [min_corner[2], max_corner[2]]
])

scatter = go.Scatter3d(
    x=df['x'],
    y=df['y'],
    z=df['z'],
    mode='markers',
    marker=dict(size=2, color='blue'),
    text=df['index'],  # index shown in tooltip
    hovertemplate="x: %{x}<br>y: %{y}<br>z: %{z}<br>index: %{text}",
    name='Clustered Points'
)

corner_scatter = go.Scatter3d(
    x=corners[:, 0],
    y=corners[:, 1],
    z=corners[:, 2],
    mode='markers',
    marker=dict(size=5, color='red', symbol='circle'),
    name='Bounding Box Corners'
)

fig = go.Figure(data=[scatter, corner_scatter])
fig.update_layout(scene=dict(
    xaxis_title='X', yaxis_title='Y', zaxis_title='Z'
))

fig.write_html("cluster_with_bbox.html")
webbrowser.open("cluster_with_bbox.html")
