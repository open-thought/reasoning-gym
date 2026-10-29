"""Find the minimum spanning tree of a weighted, undirected graph."""

import random
import re
import string
from dataclasses import dataclass
from typing import Any, Optional

from ..coaching import BaseCurriculum, RangeAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

DATASET_NAME = "minimum_spanning_tree"
MAX_NODES = len(string.ascii_uppercase)  # nodes are labelled A, B, C, ...

Edge = tuple[str, str, int]  # (node, node, weight), always with node1 < node2

QUESTION_TEMPLATE = """Your task is to find a minimum spanning tree of an undirected, weighted graph.

The graph has {num_nodes} nodes, labelled {first} to {last}, and the following edges:
{edges}

A minimum spanning tree is a set of edges that connects all nodes, contains no cycle, and has the smallest possible total weight. If several minimum spanning trees exist, any one of them is accepted.

List the edges of your tree as pairs of node labels joined by a hyphen and separated by commas, e.g. A-B, B-C, C-D. The order of the edges, and the order of the two nodes within an edge, does not matter.
"""

EDGE_PATTERN = re.compile(r"([A-Z])\s*[-\u2013\u2014]\s*([A-Z])")


@dataclass
class MinimumSpanningTreeConfig:
    """Configuration for minimum spanning tree task generation"""

    min_nodes: int = 5
    max_nodes: int = 8
    min_extra_edges: int = 2  # edges added on top of the n-1 needed to connect the graph
    max_extra_edges: int = 6
    min_weight: int = 1
    max_weight: int = 20
    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.size > 0, "size must be positive"
        assert 2 <= self.min_nodes <= self.max_nodes <= MAX_NODES, f"need 2 <= min_nodes <= max_nodes <= {MAX_NODES}"
        assert 0 <= self.min_extra_edges <= self.max_extra_edges, "need 0 <= min_extra_edges <= max_extra_edges"
        assert 1 <= self.min_weight <= self.max_weight, "need 1 <= min_weight <= max_weight"


def generate_graph(
    rng: random.Random, num_nodes: int, num_extra_edges: int, min_weight: int, max_weight: int
) -> list[Edge]:
    """Generate a connected, undirected, weighted graph on nodes A, B, C, ...

    Connectivity is guaranteed by first building a random tree over all nodes,
    then adding extra edges (capped by the number of node pairs still unused).
    """
    nodes = list(string.ascii_uppercase[:num_nodes])

    # 1. random tree: attach each node to a random node placed earlier in a shuffled order
    order = nodes[:]
    rng.shuffle(order)
    pairs = set()
    for i in range(1, num_nodes):
        parent = order[rng.randrange(i)]
        pairs.add(tuple(sorted((order[i], parent))))

    # 2. extra edges chosen from the node pairs not used yet
    all_pairs = [(a, b) for idx, a in enumerate(nodes) for b in nodes[idx + 1 :]]
    unused = [p for p in all_pairs if p not in pairs]
    pairs.update(rng.sample(unused, min(num_extra_edges, len(unused))))

    # 3. weights (sorted first so the result does not depend on set ordering)
    return [(a, b, rng.randint(min_weight, max_weight)) for a, b in sorted(pairs)]


class _UnionFind:
    """Tracks which nodes are already connected."""

    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path halving
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> bool:
        """Join the groups of a and b. Returns False if they were already joined (a cycle)."""
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            return False
        self.parent[root_a] = root_b
        return True


def kruskal(nodes: list[str], edges: list[Edge]) -> list[Edge]:
    """Return the edges of one minimum spanning tree (ties broken by node labels)."""
    groups = _UnionFind(nodes)
    tree = []
    for a, b, weight in sorted(edges, key=lambda e: (e[2], e[0], e[1])):
        if groups.union(a, b):
            tree.append((a, b, weight))
    return sorted(tree)


def parse_edges(answer: str) -> list[tuple[str, str]]:
    """Extract (node, node) pairs such as 'A-B' from an answer, each pair ordered node1 <= node2."""
    return [tuple(sorted(pair)) for pair in EDGE_PATTERN.findall(answer)]


class MinimumSpanningTreeDataset(ProceduralDataset):
    """Generates minimum spanning tree tasks on small weighted graphs"""

    def __init__(self, config: MinimumSpanningTreeConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def __getitem__(self, idx: int) -> dict:
        """Generate a single minimum spanning tree task

        Returns:
            dict with keys:
                - question: str, the graph and the instructions
                - answer: str, one optimal tree such as "A-B, B-C"
                - metadata: dict with the full graph and the optimal total weight
        """
        rng = random.Random(self.seed + idx)

        num_nodes = rng.randint(self.config.min_nodes, self.config.max_nodes)
        num_extra_edges = rng.randint(self.config.min_extra_edges, self.config.max_extra_edges)
        edges = generate_graph(rng, num_nodes, num_extra_edges, self.config.min_weight, self.config.max_weight)

        nodes = list(string.ascii_uppercase[:num_nodes])
        tree = kruskal(nodes, edges)

        question = QUESTION_TEMPLATE.format(
            num_nodes=num_nodes,
            first=nodes[0],
            last=nodes[-1],
            edges="\n".join(f"{a} - {b} (weight {w})" for a, b, w in edges),
        )

        return {
            "question": question,
            "answer": ", ".join(f"{a}-{b}" for a, b, _ in tree),
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "num_nodes": num_nodes,
                "edges": [list(e) for e in edges],
                "mst_weight": sum(w for _, _, w in tree),
                "difficulty": {
                    "num_nodes": (self.config.min_nodes, self.config.max_nodes),
                    "num_extra_edges": (self.config.min_extra_edges, self.config.max_extra_edges),
                },
            },
        }

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """1.0 if the answer is a spanning tree of minimum total weight, else 0.0.

        Several optimal trees can exist when weights tie, so we check the answer's
        properties instead of comparing it with the stored tree.
        """
        if not isinstance(answer, str) or not answer.strip():
            return 0.0

        metadata = entry["metadata"]
        num_nodes = metadata["num_nodes"]
        weights = {(a, b): w for a, b, w in metadata["edges"]}

        chosen = parse_edges(answer)
        if len(chosen) != num_nodes - 1 or len(set(chosen)) != len(chosen):
            return 0.0  # a spanning tree on n nodes has exactly n-1 distinct edges
        if any(pair not in weights for pair in chosen):
            return 0.0  # uses an edge that is not in the graph

        groups = _UnionFind(string.ascii_uppercase[:num_nodes])
        for a, b in chosen:
            if not groups.union(a, b):
                return 0.0  # cycle; with n-1 edges this also means some node is left out

        total = sum(weights[pair] for pair in chosen)
        return 1.0 if total == metadata["mst_weight"] else 0.0


class MinimumSpanningTreeCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(MinimumSpanningTreeCurriculum.__name__, MinimumSpanningTreeConfig)

        # Define attributes
        self._define_attributes(
            RangeAttributeDefinition(
                name="num_nodes",
                levels=[4, 6, 8, 10, 13, 16, 20, 26],
                default_level=0,  # Start with 4 nodes
                description="Maximum number of nodes in the graph",
                lower_field_name="min_nodes",
                upper_field_name="max_nodes",
            ),
            RangeAttributeDefinition(
                name="num_extra_edges",
                levels=[1, 2, 4, 7, 11, 16, 22, 30],
                default_level=0,  # Start with a single cycle
                description="Maximum number of edges added beyond those needed to connect the graph",
                lower_field_name="min_extra_edges",
                upper_field_name="max_extra_edges",
            ),
        )


# Register the dataset
register_dataset(DATASET_NAME, MinimumSpanningTreeDataset, MinimumSpanningTreeConfig, MinimumSpanningTreeCurriculum)
