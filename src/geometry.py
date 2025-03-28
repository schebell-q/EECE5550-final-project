from dataclasses import dataclass

import numpy as np


@dataclass
class Point(object):
	x: int
	y: int

	def distance_to(self, other):
		return np.linalg.norm([self.x - other.x, self.y - other.y])


@dataclass
class Region:
	vertices: list[int]
	traversability: float  # [0, 1)


@dataclass
class Environment:
	start: Point
	goal: Point
	vertices: set[Point]
	regions: set[Region]


class Graph:
	_vertices: set[int]
	_edges: set[(int, int)]
	_distances: np.array

	def __init__(self, vertices, edges):
		self._vertices = vertices
		self._edges = edges

	def copy(self):
		return Graph(self.vertices, self.edges)

	@property
	def vertices(self):
		return self._vertices.copy()

	@property
	def edges(self):
		return self._edges.copy()

	def remove_vertex(self, v):
		edges_to_remove = set()
		for e in self._edges:
			if v in e:
				edges_to_remove.add(e)
		self._edges.remove(edges_to_remove)
		self._vertices.remove(v)
		self._distances = None

	def add_edge(self, e):
		u, v = e
		if u == v:
			return
		if v < u:
			v, u = u, v
		self._edges.add((u, v))
		self._distances = None

	def remove_edge(self, e):
		u, v = e
		if u == v:
			return
		if v < u:
			v, u = u, v
		self._edges.remove((u, v))
		self._distances = None

	def path_distance(self, v1, v2):
		if self._distances is not None:
			return self._distances[v1.id, v2.id]
		n = len(self.vertices)
		d = np.full((n, n), np.inf)
		# TODO calculate distances
		raise NotImplementedError

	def split_intersecting_edges(self, e1):
		for e2 in self.edges:
			# TODO: calculate intersection
			# if e1 intersects e2 at vi:
			# 	v1, v2 = e2
			# 	self.remove_edge(e2)
			# 	self.add_edge((v1, vi))
			# 	self.add_edge((v2, vi))
			raise NotImplementedError
