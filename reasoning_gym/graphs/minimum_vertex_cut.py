"""Find a minimum vertex cut separating two terminals in an undirected graph.

A minimum vertex cut is a smallest set of non-terminal vertices whose removal
eliminates every path between the designated source and target. The exact
oracle below enumerates subsets; this is intentionally suited to small graph
instances used for reasoning benchmarks and avoids an additional dependency.
"""

import json
import re
from dataclasses import dataclass
from itertools import combinations
from random import Random
from typing import Any, Optional

from ..coaching import BaseCurriculum, RangeAttributeDefinition, ScalarAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

DATASET_NAME = "minimum_vertex_cut"
SOURCE = "s"
TARGET = "t"

QUESTION_TEMPLATE = """You are given a simple, connected, undirected graph G=(V,E).
The vertices `{source}` and `{target}` are the source and target terminals.
You may remove only intermediate vertices (all vertices except `{source}` and `{target}`). Removing a vertex also removes all edges incident to it.

Find a set C of the smallest possible cardinality such that no path remains from `{source}` to `{target}` after removing every vertex in C. The direct edge ({source}, {target}) is not present.

Vertices: {vertices}
Edges: {edges}

Return only a JSON object with this schema:
{{"cut_vertices": ["v0"], "cut_size": 1}}
The list must contain distinct intermediate vertices and cut_size must equal the list length. If several minimum cuts exist, any one of them is accepted.
"""


def _has_path(
    vertices: list[str],
    edges: list[tuple[str, str]] | list[list[str]],
    source: str,
    target: str,
    removed: set[str],
) -> bool:
    """Return whether source can still reach target after removing vertices."""
    if source in removed or target in removed:
        return False

    adjacency = {vertex: set() for vertex in vertices if vertex not in removed}
    for u, v in edges:
        if u not in removed and v not in removed:
            adjacency[u].add(v)
            adjacency[v].add(u)

    stack = [source]
    visited = {source}
    while stack:
        current = stack.pop()
        if current == target:
            return True
        for neighbour in adjacency[current]:
            if neighbour not in visited:
                visited.add(neighbour)
                stack.append(neighbour)
    return False


def _minimum_vertex_cut(
    vertices: list[str],
    edges: list[tuple[str, str]] | list[list[str]],
    source: str,
    target: str,
) -> list[str]:
    """Return one minimum-cardinality non-terminal cut using exact enumeration."""
    candidates = sorted(vertex for vertex in vertices if vertex not in {source, target})
    for cut_size in range(1, len(candidates) + 1):
        for subset in combinations(candidates, cut_size):
            if not _has_path(vertices, edges, source, target, set(subset)):
                return list(subset)
    # The graph generator guarantees a source-target path with no direct edge,
    # so removing all intermediate vertices must always disconnect the terminals.
    raise ValueError("No vertex cut found; graph or terminal configuration is invalid")


def _edge_list_for_question(edges: list[tuple[str, str]]) -> str:
    return "[" + ", ".join(f"({u}, {v})" for u, v in edges) + "]"


@dataclass
class MinimumVertexCutConfig:
    """Configuration for the Minimum Vertex Cut task generator."""

    min_num_vertices: int = 6
    max_num_vertices: int = 10
    min_cut_size: int = 1
    max_cut_size: int = 3
    edge_probability: float = 0.3
    max_generation_attempts: int = 512
    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        assert self.min_num_vertices >= 4, "min_num_vertices must be at least 4"
        assert self.max_num_vertices >= self.min_num_vertices, (
            "max_num_vertices must be greater than or equal to min_num_vertices"
        )
        assert self.min_cut_size >= 1, "min_cut_size must be at least 1"
        assert self.max_cut_size >= self.min_cut_size, (
            "max_cut_size must be greater than or equal to min_cut_size"
        )
        assert self.max_cut_size <= self.max_num_vertices - 2, (
            "max_cut_size cannot exceed max_num_vertices - 2"
        )
        assert 0.0 <= self.edge_probability < 1.0, (
            "edge_probability must be in the interval [0, 1)"
        )
        assert not (self.edge_probability == 0.0 and self.min_cut_size > 1), (
            "edge_probability must be positive when min_cut_size is greater than 1"
        )
        assert self.max_generation_attempts > 0, "max_generation_attempts must be positive"
        assert self.size > 0, "size must be positive"


class MinimumVertexCutDataset(ProceduralDataset):
    """Generate connected graph problems with an exactly verifiable minimum cut."""

    def __init__(self, config: MinimumVertexCutConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _generate_connected_graph(
        self, rng: Random, num_vertices: int
    ) -> tuple[list[str], list[tuple[str, str]]]:
        """Build a connected graph from a random terminal-to-terminal path and chords."""
        vertices = [SOURCE, TARGET] + [f"v{i}" for i in range(num_vertices - 2)]
        path_vertices = vertices[2:].copy()
        rng.shuffle(path_vertices)
        path = [SOURCE] + path_vertices + [TARGET]
        edge_set: set[tuple[str, str]] = set()

        def add_edge(u: str, v: str) -> None:
            edge_set.add(tuple(sorted((u, v))))

        for u, v in zip(path, path[1:]):
            add_edge(u, v)

        # Add random chords, but never add the direct source-target edge.
        for i, u in enumerate(vertices):
            for v in vertices[i + 1 :]:
                if {u, v} == {SOURCE, TARGET}:
                    continue
                edge = tuple(sorted((u, v)))
                if edge not in edge_set and rng.random() < self.config.edge_probability:
                    edge_set.add(edge)

        edges = sorted(edge_set)
        return vertices, edges

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rng = Random(self.seed + idx)
        found = None

        for _ in range(self.config.max_generation_attempts):
            num_vertices = rng.randint(
                self.config.min_num_vertices, self.config.max_num_vertices
            )
            # At most n-2 intermediate vertices can be removed.
            if num_vertices - 2 < self.config.min_cut_size:
                continue
            vertices, edges = self._generate_connected_graph(rng, num_vertices)
            cut = _minimum_vertex_cut(vertices, edges, SOURCE, TARGET)
            if self.config.min_cut_size <= len(cut) <= self.config.max_cut_size:
                found = (num_vertices, vertices, edges, cut)
                break

        if found is None:
            raise RuntimeError(
                "Could not generate a graph whose minimum vertex cut falls within "
                f"[{self.config.min_cut_size}, {self.config.max_cut_size}] after "
                f"{self.config.max_generation_attempts} attempts. Adjust edge_probability, "
                "the cut-size range, or max_generation_attempts."
            )

        num_vertices, vertices, edges, cut = found
        solution = {"cut_vertices": cut, "cut_size": len(cut)}
        answer = json.dumps(solution, separators=(",", ":"))
        graph = {
            "vertices": vertices,
            "edges": [list(edge) for edge in edges],
            "source": SOURCE,
            "target": TARGET,
        }
        question = QUESTION_TEMPLATE.format(
            source=SOURCE,
            target=TARGET,
            vertices=vertices,
            edges=_edge_list_for_question(edges),
        )

        return {
            "question": question,
            "answer": answer,
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "graph": graph,
                "solution": solution,
                "difficulty": {
                    "num_vertices": (
                        self.config.min_num_vertices,
                        self.config.max_num_vertices,
                    ),
                    "cut_size": (self.config.min_cut_size, self.config.max_cut_size),
                    "edge_probability": self.config.edge_probability,
                },
            },
        }

    @staticmethod
    def _parse_candidate(answer: str) -> Optional[dict[str, Any]]:
        """Parse JSON responses, allowing a single surrounding Markdown fence."""
        candidate_text = answer.strip()
        if candidate_text.startswith("```"):
            candidate_text = re.sub(r"^```(?:json)?\s*", "", candidate_text, count=1)
            candidate_text = re.sub(r"\s*```$", "", candidate_text, count=1)
        try:
            parsed = json.loads(candidate_text)
        except (TypeError, json.JSONDecodeError):
            return None
        return parsed if isinstance(parsed, dict) else None

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Award full credit for any valid minimum cut, including tied optima."""
        if not isinstance(answer, str) or not answer.strip():
            return 0.0
        candidate = self._parse_candidate(answer)
        if candidate is None:
            return 0.0

        cut_vertices = candidate.get("cut_vertices")
        cut_size = candidate.get("cut_size")
        if not isinstance(cut_vertices, list) or not cut_vertices:
            return 0.0
        if any(not isinstance(vertex, str) for vertex in cut_vertices):
            return 0.0
        if len(cut_vertices) != len(set(cut_vertices)):
            return 0.0
        if isinstance(cut_size, bool) or not isinstance(cut_size, int):
            return 0.0
        if cut_size != len(cut_vertices):
            return 0.0

        graph = entry.get("metadata", {}).get("graph", {})
        vertices = graph.get("vertices")
        edges = graph.get("edges")
        source = graph.get("source")
        target = graph.get("target")
        if not isinstance(vertices, list) or not isinstance(edges, list):
            return 0.0
        if source not in vertices or target not in vertices or source == target:
            return 0.0

        allowed = set(vertices) - {source, target}
        cut_set = set(cut_vertices)
        if not cut_set.issubset(allowed):
            return 0.0

        try:
            optimum = _minimum_vertex_cut(vertices, edges, source, target)
            separates = not _has_path(vertices, edges, source, target, cut_set)
        except (KeyError, TypeError, ValueError):
            return 0.0

        if separates and len(cut_set) == len(optimum):
            return 1.0
        return 0.0


class MinimumVertexCutCurriculum(BaseCurriculum):
    """Curriculum that increases graph size and requested cut complexity."""

    def __init__(self):
        super().__init__(MinimumVertexCutCurriculum.__name__, MinimumVertexCutConfig)
        self._define_attributes(
            RangeAttributeDefinition(
                name="num_vertices",
                levels=[6, 8, 10, 12],
                description="Number of vertices in the graph",
                lower_field_name="min_num_vertices",
                upper_field_name="max_num_vertices",
                ensure_interval=True,
            ),
            RangeAttributeDefinition(
                name="cut_size",
                levels=[1, 2, 3],
                description="Allowed minimum vertex cut size",
                lower_field_name="min_cut_size",
                upper_field_name="max_cut_size",
                ensure_interval=True,
            ),
            ScalarAttributeDefinition(
                name="edge_probability",
                field_name="edge_probability",
                levels=[0.15, 0.3, 0.45, 0.6],
                description="Probability of adding a non-path edge",
            ),
        )


register_dataset(
    DATASET_NAME,
    MinimumVertexCutDataset,
    MinimumVertexCutConfig,
    MinimumVertexCutCurriculum,
)
