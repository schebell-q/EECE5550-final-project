from src.algorithm import compute_reduced_visibility_graph
from tests.test_environment import make_basic_env
from tests.test_graph import make_freespace_graph


def test_basic_rvg():
    env = make_basic_env()
    g_free = compute_reduced_visibility_graph(env)
    g_expected = make_freespace_graph()
    assert g_free.vertices == g_expected.vertices
    assert g_free.edges == g_expected.edges
