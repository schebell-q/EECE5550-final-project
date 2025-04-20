from pathlib import Path

from src.algorithm import algorithm
from src.geometry import Environment
from src.plotting import plot_environment, save_fig, plot_graph
from tests.test_environment import make_basic_env, make_long_toy_env
from tests.test_graph import make_freespace_graph

output_dir = Path("test_plots")


def test_plot_environment():
    fig = plot_environment(make_basic_env())
    file = output_dir / "basic_environment.html"
    save_fig(fig, file)
    assert file.exists()
    fig = plot_environment(make_long_toy_env())
    file = output_dir / "long_toy_environment.html"
    save_fig(fig, file)
    assert file.exists()


def test_plot_graph():
    env = make_basic_env()
    fig = plot_environment(env)
    fig = plot_graph(env, make_freespace_graph(), fig=fig)
    file = output_dir / "freespace_graph.html"
    save_fig(fig, file)
    assert file.exists()


def plot_all_graphs(env_name: str, env: Environment, k: float, alpha: float):
    graphs_dict = algorithm(env, k, alpha, return_all_graphs=True)
    ellipse_heurustic = graphs_dict["ellipse_heuristic"]
    for i, (graph_name, plot_ellipse) in enumerate([
        ("g_free", False),
        ("g_pruned", True),
        ("g_shortcuts", False),
        ("g_final", False),
    ]):
        h = ellipse_heurustic if plot_ellipse else None
        fig = plot_environment(env, ellipse_heuristic=h)
        fig = plot_graph(env, graphs_dict[graph_name], fig=fig)
        file = output_dir / f"{env_name}_{i}_{graph_name}.html"
        save_fig(fig, file)
        assert file.exists()


def test_plot_all_graphs():
    plot_all_graphs("basic", make_basic_env(), 1.1, 2)
    plot_all_graphs("long_toy", make_long_toy_env(), 4.0, 1.5)
