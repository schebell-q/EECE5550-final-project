from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Optional

import numpy as np

Vertex = int
Point = tuple[float, float]
Edge = tuple[Vertex, Vertex]

# This is a minimum buffer distance from the endpoints to be considered an intersection
# If too close to an end, it's not a useful intersection so we treat it as a miss
INTERSECTION_EPS = 0.01


@dataclass
class Region:
    vertices: tuple[Vertex, ...]
    traversability: float  # [0, 1)

    def __hash__(self):
        return self.vertices.__hash__()


class Environment:
    start: Vertex
    goal: Vertex
    points: np.array  # Nx2
    regions: set[Region]
    _n_points: int

    def __init__(self, start: Vertex, goal: Vertex, points: np.ndarray,
                 regions: set[Region]):
        self.start = start
        self.goal = goal
        self.regions = regions
        n, d = points.shape
        if d != 2:
            raise ValueError("points must have shape Nx2 but was Nx%i" % d)
        self.points = points
        self._n_points = n

    def distance_between(self, v1: Vertex, v2: Vertex):
        p1 = self.points[v1]
        p2 = self.points[v2]
        return np.linalg.norm(p1 - p2).item()

    def add_point(self, p: Point) -> Vertex:
        max_n, _ = self.points.shape
        v = self._n_points + 1
        if v >= max_n:
            points = np.zeros((2 * max_n, 2))
            points[:max_n] = self.points
            self.points = points
        self.points[v, 0] = p[0]
        self.points[v, 1] = p[1]
        self._n_points = v
        return v

    def intersect_edges(self, e1: Edge, e2: Edge) -> Optional[Point]:
        v1, v2 = e1
        v3, v4 = e2
        p = self.points[v1]
        r = self.points[v2] - p
        q = self.points[v3]
        s = self.points[v4] - q

        def cross(v, w):
            return v[0] * w[1] - v[1] * w[0]

        # Algorithm from https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect#565282
        q_minus_p = q - p
        r_cross_s = cross(r, s)
        q_minus_p_cross_r = cross(q_minus_p, r)
        q_minus_p_cross_s = cross(q_minus_p, s)

        if abs(r_cross_s) == 0 and abs(q_minus_p_cross_r) == 0:
            # Lines are collinear
            # For this implementation, assume there's some nonzero angle between intersecting edges
            # and so any edges that happen to be collinear are not intersecting
            return None
        elif abs(r_cross_s) == 0:
            # Parallel and non-intersecting
            return None

        t = q_minus_p_cross_s / r_cross_s
        u = q_minus_p_cross_r / r_cross_s

        # If the intersection is very close to or at an endpoint, consider that a miss
        if t < INTERSECTION_EPS or 1 - INTERSECTION_EPS < t or \
                u < INTERSECTION_EPS or 1 - INTERSECTION_EPS < u:
            return None

        intersection = p + t * r
        return intersection[0].item(), intersection[1].item()

    def regions_intersected_by_edge(self, e: Edge) -> list[Region]:
        intersected_regions = []
        for region in self.regions:
            # Regions must be convex, so both endpoints on the edge and non-adjacent means it passes inside
            if e[0] in region.vertices and e[1] in region.vertices:
                v1, v2 = e
                if v1 > v2:
                    v1, v2 = v2, v1
                v1_idx = region.vertices.index(v1)
                v2_idx = region.vertices.index(v2)
                if not (abs(v1_idx - v2_idx) == 1 or v1_idx == 0 and v2_idx == len(region.vertices) - 1):
                    intersected_regions.append(region)
                    continue

            # If at least one of the endpoints is not on the edge,
            # at least one edge would be intersected if it passes inside
            for i in range(len(region.vertices)):
                v1 = region.vertices[i]
                v2 = region.vertices[(i + 1) % len(region.vertices)]
                if self.intersect_edges(e, (v1, v2)):
                    intersected_regions.append(region)
                    continue
        return intersected_regions


class Graph:
    _environment: Environment
    _vertices: set[Vertex]
    _edges: set[Edge]

    def __init__(self, env: Environment, vertices: set[Vertex], edges: set[Edge]):
        self._environment = env
        self._vertices = vertices
        self._edges = {
            (v1, v2) if v1 < v2 else (v2, v1)
            for (v1, v2) in edges
        }

    def copy(self):
        return Graph(self._environment, self.vertices, self.edges)

    @property
    def vertices(self):
        return self._vertices.copy()

    @property
    def edges(self):
        return self._edges.copy()

    def remove_vertex(self, v: Vertex):
        edges_to_remove = set()
        for e in self._edges:
            if v in e:
                edges_to_remove.add(e)
        self._edges.difference_update(edges_to_remove)
        self._vertices.remove(v)
        self._distances = None

    def add_edge(self, e: Edge):
        u, v = e
        if u == v:
            return
        if v < u:
            v, u = u, v
        self._edges.add((u, v))
        self._vertices.add(u)
        self._vertices.add(v)

    def remove_edge(self, e: Edge):
        u, v = e
        if u == v:
            return
        if v < u:
            v, u = u, v
        self._edges.remove((u, v))
        self._distances = None

    def path_distance(self, v1: Vertex, v2: Vertex):
        # This uses a simplified A* search with distance metric d
        def d(a, b):
            return self._environment.distance_between(a, b)

        searched_verts = set()

        # Items are: (score, distance_to, vertex) where score is distance_to + Euclidean heuristic
        search_queue = [(0.0, 0.0, v1)]

        while search_queue:
            _, distance_to, vertex = heappop(search_queue)
            if vertex == v2:
                return distance_to
            searched_verts.add(vertex)

            # Find all neighbors of vertex that haven't been searched
            for e in self.edges:
                if vertex not in e:
                    continue
                neighbor = e[0] if e[0] != vertex else e[1]
                if neighbor in searched_verts:
                    continue

                distance_after = distance_to + d(vertex, neighbor)
                heappush(search_queue, (distance_after + d(neighbor, v2), distance_after, neighbor))

        return np.inf

    def split_intersecting_edges(self, e1: Edge):
        v1, v2 = e1
        for e2 in self.edges:
            # If one of the endpoints is shared, there's nowhere to split
            if v1 in e2 or v2 in e2:
                continue

            p_int = self._environment.intersect_edges(e1, e2)
            if p_int is None:
                continue

            v_int = self._environment.add_point(p_int)
            v3, v4 = e2

            self.remove_edge(e2)
            self.add_edge((v_int, v3))
            self.add_edge((v_int, v4))
