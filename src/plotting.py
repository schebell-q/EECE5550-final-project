from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from src.geometry import Environment, Graph


def plot_environment(env: Environment, fig: go.Figure = None, ellipse_heuristic: float = None) -> go.Figure:
    if fig is None:
        fig = go.Figure()

    start = env.points[env.start, :]
    goal = env.points[env.goal, :]
    fig.add_trace(
        go.Scatter(
            x=[start[0], goal[0]], y=[start[1], goal[1]],
            name="Start and Goal", mode="markers",
            marker=dict(color="black", size=30)
        )
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    for i, region in enumerate(env.regions):
        # The polygon needs to be closed, so add the first point again at the end
        region_pts = env.points[[*region.vertices, region.vertices[0]]]
        fig.add_trace(
            go.Scatter(
                x=region_pts[:, 0], y=region_pts[:, 1],
                name=f"Region {i + 1}, {region.traversability * 100:.0f}%", fill="toself",
                line=dict(color="red"),
                fillcolor=f"rgba(255,0,0,{region.traversability})"
            )
        )

    if ellipse_heuristic is not None:
        # The ellipse has the start and goal as foci
        a = ellipse_heuristic / 2
        c = np.linalg.norm(goal - start) / 2
        b = np.sqrt(a ** 2 - c ** 2)

        delta = goal - start
        rotation_angle = np.arctan(delta[1] / delta[0])
        translation = (start + goal) / 2

        theta = np.linspace(0, 2*np.pi, 100)
        scaled_ellipse = np.vstack([a * np.cos(theta), b * np.sin(theta)]).T
        rotated_ellipse = scaled_ellipse @ np.array([
            [np.cos(rotation_angle), -np.sin(rotation_angle)],
            [np.sin(rotation_angle), np.cos(rotation_angle)]
        ])
        ellipse = rotated_ellipse + translation

        fig.add_trace(
            go.Scatter(
                x=ellipse[:, 0], y=ellipse[:, 1],
                name=f"Ellipse Heuristic",
                line=dict(color="black"),
            )
        )

    return fig


def plot_graph(env: Environment, graph: Graph, fig: go.Figure = None) -> go.Figure:
    if fig is None:
        fig = go.Figure()

    graph_vertices = env.points[list(graph.vertices), :]
    fig.add_trace(
        go.Scatter(
            x=graph_vertices[:, 0], y=graph_vertices[:, 1],
            name="Graph Vertices", mode="markers",
            marker=dict(color="blue", size=10)
        )
    )

    edges_x = []
    edges_y = []
    for edge in graph.edges:
        v1, v2 = edge
        edges_x.append(env.points[v1, 0])
        edges_x.append(env.points[v2, 0])
        edges_y.append(env.points[v1, 1])
        edges_y.append(env.points[v2, 1])
        edges_x.append(None)
        edges_y.append(None)
    fig.add_trace(
        go.Scatter(
            x=edges_x, y=edges_y,
            name=f"Graph Edges", mode="lines",
            line=dict(color="blue")
        )
    )

    return fig


def save_fig(figure: go.Figure, filename: Path):
    filename.parent.mkdir(exist_ok=True)
    filename.open('w').write(figure.to_html())
