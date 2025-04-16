import math
import numpy as np

from pytest import approx

from src.geometry import Graph
from tests.test_environment import make_basic_env


def make_empty_graph() -> Graph:
    vertices = set(range(6))
    edges = set()
    return Graph(make_basic_env(), vertices, edges)


def make_freespace_graph() -> Graph:
    vertices = set(range(6))
    edges = {
        (0, 2),
        (0, 5),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 2),
        (1, 3),
        (1, 4),
    }
    return Graph(make_basic_env(), vertices, edges)


def test_attributes():
    graphs = [
        make_empty_graph(),
        make_freespace_graph(),
    ]

    for g1 in graphs:
        v = g1._vertices.copy()
        e = g1._edges.copy()

        assert v == g1.vertices
        assert e == g1.edges

        g2 = g1.copy()

        assert v == g2.vertices
        assert e == g2.edges

        g1._vertices.add(999)
        g1._edges.add((999, 999))

        assert g1.vertices != g2.vertices
        assert g1.edges != g2.edges


def test_remove_vertex():
    graphs = [
        make_empty_graph(),
        make_freespace_graph(),
    ]

    for g in graphs:
        v = g.vertices
        g.remove_vertex(0)
        if 0 in v:
            assert 0 not in g.vertices
        assert 0 not in g.vertices
        for e in g.edges:
            assert 0 not in e


def test_add_remove_edge():
    graphs = [
        make_empty_graph(),
        make_freespace_graph(),
    ]

    for g in graphs:
        assert (0, 1) not in g.edges
        assert (1, 0) not in g.edges
        g.add_edge((0, 1))
        assert (0, 1) in g.edges
        assert (1, 0) not in g.edges
        g.remove_edge((0, 1))
        assert (0, 1) not in g.edges
        assert (1, 0) not in g.edges
        g.add_edge((1, 0))
        assert (0, 1) in g.edges
        assert (1, 0) not in g.edges
        g.remove_edge((0, 1))
        assert (0, 1) not in g.edges
        assert (1, 0) not in g.edges


def test_path_distance():
    g_empty = make_empty_graph()
    assert g_empty.path_distance(0, 1) == np.inf
    g_free = make_freespace_graph()
    assert approx(g_free.path_distance(0, 1)) == 1 + 2 * math.sqrt(2)


def test_split_edges():
    g_free = make_freespace_graph()
    assert len(g_free.edges) == 8
    old_edges = g_free.edges.copy()
    g_free.add_edge_and_split((0, 1))
    new_edges = g_free.edges.copy()
    assert len(old_edges.difference(new_edges)) == 2
    assert len(new_edges.difference(old_edges)) == 7


def test_edge_order():
    g = make_freespace_graph()
    for v1, v2 in g.edges:
        assert v1 < v2
