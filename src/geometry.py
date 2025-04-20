from heapq import heappop, heappush
from typing import Optional, Generator

import numpy as np

Vertex = int
Point = tuple[float, float]
Edge = tuple[Vertex, Vertex]

# This is a minimum buffer distance from the endpoints to be considered an intersection
# If too close to an end, it's not a useful intersection so we treat it as a miss
INTERSECTION_EPS = 0.001

# If we're adding a new point to the environment, and it's really close to an existing point,
# just use the existing point to reduce redundancy.
NEW_POINT_EPS = 0.001


class Region:
    vertices: tuple[Vertex, ...]
    traversability: float  # [0, 1)
    _edges: Optional[set[Edge]]

    def __init__(self, vertices, traversability):
        self.vertices = vertices
        self.traversability = traversability
        self._edges = None

    def __hash__(self):
        return self.vertices.__hash__()

    @property
    def edges(self) -> set[Edge]:
        if self._edges:
            return self._edges

        edges = set()
        for i in range(len(self.vertices)):
            v1 = self.vertices[i]
            v2 = self.vertices[(i+1) % len(self.vertices)]
            e = (v1, v2) if v1 < v2 else (v2, v1)
            edges.add(e)

        self._edges = edges
        return self._edges


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
        self.points = points.astype("float64")
        self._n_points = n

    def distance_between(self, v1: Vertex, v2: Vertex):
        p1 = self.points[v1]
        p2 = self.points[v2]
        return np.linalg.norm(p1 - p2).item()

    def add_point(self, p: Point) -> Vertex:
        point_dists = np.linalg.norm(self.points[:self._n_points] - p, axis=1)
        if (point_dists < NEW_POINT_EPS).any():
            return int(np.argmin(np.abs(point_dists)))

        max_n, _ = self.points.shape
        v = self._n_points
        self._n_points += 1
        if self._n_points > max_n:
            points = np.zeros((2 * max_n, 2))
            points[:max_n] = self.points
            self.points = points
        self.points[v, 0] = p[0]
        self.points[v, 1] = p[1]
        return v

    def intersect_edges(self, e1: Edge, e2: Edge) -> (Optional[Point], bool):
        """Returns (intersection: Point, intersected_at_endpoint: bool)"""
        if e1 == e2:
            return None, False

        v1, v2 = e1
        v3, v4 = e2
        p = self.points[v1]
        r = self.points[v2] - p
        q = self.points[v3]
        s = self.points[v4] - q

        if v1 in e2:
            intersection = p
            return (intersection[0].item(), intersection[1].item()), True
        if v2 in e2:
            intersection = p + r
            return (intersection[0].item(), intersection[1].item()), True

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
            return None, False
        elif abs(r_cross_s) == 0:
            # Parallel and non-intersecting
            return None, False

        t = q_minus_p_cross_s / r_cross_s
        u = q_minus_p_cross_r / r_cross_s

        if t < 0 or t > 1 or u < 0 or u > 1:
            return None, False

        intersected_at_endpoint = t < INTERSECTION_EPS or t > 1 - INTERSECTION_EPS

        intersection = p + t * r
        return (intersection[0].item(), intersection[1].item()), intersected_at_endpoint

    def regions_intersected_by_edge(self, e: Edge) -> Generator[Region]:
        for region in self.regions:
            # If the edge is in the polygon, that's doesn't pass inside the region
            if e in region.edges:
                continue

            # But if both vertices are otherwise in the polygon, that would pass inside
            v1, v2 = e
            if v1 in region.vertices and v2 in region.vertices:
                yield region
                continue

            # If at least one of the vertices is not on the polygon,
            # and it intersects an edge on the polygon,
            # then it passes inside the region
            for region_edge in region.edges:
                intersection, intersected_at_endpoint = self.intersect_edges(e, region_edge)
                if intersection and not intersected_at_endpoint:
                    yield region
                    continue

    def regions_within_ellipse(self, e: Edge, ellipse_dist: float) -> Generator[Region]:
        for region in self.regions:
            # If either vertex is in the region, the region is definitely within the ellipse
            v1, v2 = e
            if v1 in region.vertices and v2 in region.vertices:
                yield region
                continue

            # If the shortest path from v1 to v2 through vr is in the ellipse, then the region is in the ellipse
            for vr in region.vertices:
                if self.distance_between(v1, vr) + self.distance_between(v2, vr) <= ellipse_dist:
                    yield region
                    break


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
        return Graph(self._environment, self.vertices.copy(), self.edges.copy())

    @property
    def vertices(self):
        return self._vertices

    @property
    def edges(self):
        return self._edges

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

    def add_edge_and_split(self, edge: Edge):
        split_edges = {edge}

        for e2 in self.edges.copy():
            p_int, intersected_at_endpoint = self._environment.intersect_edges(edge, e2)
            if not p_int or intersected_at_endpoint:
                continue

            for e1 in split_edges.copy():
                p_int, intersected_at_endpoint = self._environment.intersect_edges(e1, e2)

                # If one of the endpoints is shared, there's nowhere to split
                if intersected_at_endpoint:
                    break

                if not p_int:
                    continue

                v_int = self._environment.add_point(p_int)
                v1, v2 = e1
                v3, v4 = e2

                split_edges.remove(e1)
                split_edges.add((v1, v_int))
                split_edges.add((v2, v_int))

                self.remove_edge(e2)
                self.add_edge((v3, v_int))
                self.add_edge((v4, v_int))

                # Only one edge segment could be intersected by e2, so move on
                break

        for e in split_edges:
            self.add_edge(e)
