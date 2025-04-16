from src.algorithm import algorithm
from tests.test_environment import make_long_toy_env


def test_algorithm():
    env = make_long_toy_env()
    k = 5
    alpha = 1.1

    graphs_dict = algorithm(env, k, alpha, return_all_graphs=True)
    for graph_name in [
        "g_free",
        "g_pruned",
        "g_shortcuts",
        "g_final",
    ]:
        g = graphs_dict[graph_name]
        assert len(g.vertices) > 10
        assert len(g.edges) > 10

    g_free = graphs_dict["g_free"]
    assert (2, 5) in g_free.edges

    g_pruned = graphs_dict["g_pruned"]
    assert (2, 3) in g_pruned.edges
    assert (3, 4) in g_pruned.edges
    assert (4, 5) in g_pruned.edges
    assert (2, 5) in g_pruned.edges
