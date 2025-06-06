import itertools
import random
import math

from src.geometry import Environment, Graph


def algorithm(env: Environment, k: float, alpha: float, return_all_graphs=False) -> Graph | dict[str, Graph | float]:
    if k <= 1:
        raise ValueError("k must be greater than 1 but was %f" % k)
    if alpha <= 1:
        raise ValueError("alpha must be greater than 1 but was %f" % alpha)

    # Reduced visibility graph, or free-space graph
    g_free = compute_reduced_visibility_graph(env)

    ellipse_heuristic = g_free.path_distance(env.start, env.goal)

    # Remove edges that are guaranteed to not lie on the optimal path
    g_pruned = prune_suboptimal_edges(env, g_free, ellipse_heuristic)

    # Add paths through uncertain regions, split at region boundaries to make recovery paths
    g_shortcuts = add_shortcut_edges(env, g_pruned, k, g_free)

    # Remove redundant edges
    g_final = optimize_redundant_edges(env, alpha, g_shortcuts)

    if return_all_graphs:
        return {
            "k": k,
            "alpha": alpha,
            "g_free": g_free,
            "ellipse_heuristic": ellipse_heuristic,
            "g_pruned": g_pruned,
            "g_shortcuts": g_shortcuts,
            "g_final": g_final,
        }
    return g_final


def compute_reduced_visibility_graph(env: Environment) -> Graph:
    v_free = set(range(len(env.points)))

    # Prune edges that intersect any region
    e_free = set()
    for e in itertools.combinations(v_free, 2):
        v1, v2 = e
        if v2 < v1:
            raise ValueError(f"v1 should always be less than v2 but they were: {v1}, {v2}")
        if next(env.regions_intersected_by_edge(e), None) is None:
            # Add the edge since it doesn't intersect any regions
            e_free.add(e)

    g_free = Graph(env, v_free, e_free)
    return g_free


def prune_suboptimal_edges(env: Environment, g_free: Graph, ellipse_heuristic: float) -> Graph:
    g_pruned = g_free.copy()
    for v in g_free.vertices:
        if env.distance_between(env.start, v) + env.distance_between(v, env.goal) > ellipse_heuristic:
            g_pruned.remove_vertex(v)
    return g_pruned


def add_shortcut_edges(env: Environment, g_pruned: Graph, k: float, g_free: Graph) -> Graph:
    # The set of all regions represented in g_pruned
    # NOTE: some regions may have been pruned if they're far from the optimal path
    regions = set()
    for region in env.regions:
        # If any of region's vertices are in graph, add it
        for v in region.vertices:
            if v in g_pruned.vertices:
                regions.add(region)
                break

    g_shortcuts = g_pruned.copy()
    # Test all possible edges and add the ones where gamma > k
    for v1, v2 in itertools.combinations(g_pruned.vertices, 2):
        if (v1, v2) in g_pruned.edges:
            continue

        distance = env.distance_between(v1, v2)
        ellipse_heuristic = g_free.path_distance(v1, v2)

        # The probability that all regions relevant to the shortcut are untraversable
        regions = env.regions_within_ellipse((v1, v2), ellipse_heuristic)
        rho_b = math.prod((1 - r.traversability) for r in regions)

        # A heuristic ratio based on how 'useful' the edge is
        # If it's likely to be traversable or much shorter, add it
        gamma = ellipse_heuristic / (rho_b * ellipse_heuristic + (1 - rho_b) * distance)
        if gamma > k:
            g_shortcuts.add_edge_and_split((v1, v2))

    return g_shortcuts


def optimize_redundant_edges(env, alpha: float, g_recovery: Graph) -> Graph:
    g_final = g_recovery.copy()

    shuffled_edges = list(g_final.edges)
    random.shuffle(shuffled_edges)
    for e in shuffled_edges:
        if e not in g_final.edges:
            continue

        u, v = e
        duv = env.distance_between(u, v)

        # Remove all edges (w, v) where detouring through u
        # doesn't increase the path distance significantly
        for w in g_final.vertices:
            if (w, u) in g_final.edges and (w, v) in g_final.edges:
                dwu = env.distance_between(w, u)
                dwv = env.distance_between(w, v)
                if (dwu + duv) / dwv < alpha:
                    g_final.remove_edge((w, v))
                elif (dwv + duv) / dwu < alpha:
                    g_final.remove_edge((w, u))

    return g_final
