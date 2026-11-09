"""Find the total weight of a minimum spanning tree of a weighted undirected graph"""

from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from ..coaching import BaseCurriculum, RangeAttributeDefinition, ScalarAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

DATASET_NAME = "minimum_spanning_tree"

QUESTION_TEMPLATE = """You are given a connected, undirected, weighted graph with {num_nodes} nodes labelled 0 to {max_node}.
Each line below describes one edge in the format "u - v: w", meaning nodes u and v are connected by an edge of weight w.

{edges}

A spanning tree is a subset of edges that connects all nodes without forming any cycle.
Find a minimum spanning tree of this graph, i.e. a spanning tree whose edge weights have the smallest possible sum.

Respond with only the total weight of the minimum spanning tree as a single integer.
"""


@dataclass
class MinimumSpanningTreeConfig:
    """Configuration for Minimum Spanning Tree dataset generation"""

    min_nodes: int = 4
    max_nodes: int = 8
    min_weight: int = 1
    max_weight: int = 20
    p_extra_edge: float = 0.4  # Probability of adding each non-tree edge on top of a random spanning tree

    size: int = 500  # Virtual dataset size
    seed: Optional[int] = None

    def validate(self):
        """Validate configuration parameters"""
        assert 2 <= self.min_nodes, "min_nodes must be at least 2"
        assert self.min_nodes <= self.max_nodes, "min_nodes must be less than or equal to max_nodes"
        assert 1 <= self.min_weight, "min_weight must be at least 1"
        assert self.min_weight <= self.max_weight, "min_weight must be less than or equal to max_weight"
        assert 0 <= self.p_extra_edge <= 1, "p_extra_edge must be between 0 and 1"


class MinimumSpanningTreeDataset(ProceduralDataset):
    """Generates Minimum Spanning Tree exercises with configurable difficulty"""

    def __init__(self, config: MinimumSpanningTreeConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _get_edges(self, rng: Random, num_nodes: int) -> list[tuple[int, int, int]]:
        """Generate the edges of a random connected weighted graph"""
        # Random spanning tree first so the graph is guaranteed to be connected
        order = list(range(num_nodes))
        rng.shuffle(order)
        pairs = set()
        for i in range(1, num_nodes):
            u, v = order[i], order[rng.randrange(i)]
            pairs.add((min(u, v), max(u, v)))

        for u in range(num_nodes):
            for v in range(u + 1, num_nodes):
                if (u, v) not in pairs and rng.random() < self.config.p_extra_edge:
                    pairs.add((u, v))

        edges = [(u, v, rng.randint(self.config.min_weight, self.config.max_weight)) for u, v in sorted(pairs)]
        rng.shuffle(edges)
        return edges

    def _kruskal(self, num_nodes: int, edges: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
        """Compute a minimum spanning tree using Kruskal's algorithm"""
        parent = list(range(num_nodes))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        tree = []
        for u, v, w in sorted(edges, key=lambda e: e[2]):
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[ru] = rv
                tree.append((u, v, w))
        return tree

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Score the answer by comparing the reported total weight with the oracle"""
        if not isinstance(answer, str):
            return 0.0
        try:
            return 1.0 if int(answer.strip()) == int(entry["answer"]) else 0.0
        except ValueError:
            return 0.0

    def __getitem__(self, idx: int) -> dict:
        """Generate a single Minimum Spanning Tree question"""
        rng = Random(self.seed + idx)

        num_nodes = rng.randint(self.config.min_nodes, self.config.max_nodes)
        edges = self._get_edges(rng, num_nodes)
        mst = self._kruskal(num_nodes, edges)
        total_weight = sum(w for _, _, w in mst)

        edges_str = "\n".join(f"{u} - {v}: {w}" for u, v, w in edges)
        question = QUESTION_TEMPLATE.format(num_nodes=num_nodes, max_node=num_nodes - 1, edges=edges_str)

        return {
            "question": question,
            "answer": str(total_weight),
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "num_nodes": num_nodes,
                "edges": [list(e) for e in edges],
                "mst_edges": [list(e) for e in mst],
                "solution": total_weight,
                "difficulty": {
                    "nodes": (self.config.min_nodes, self.config.max_nodes),
                    "max_weight": self.config.max_weight,
                    "p_extra_edge": self.config.p_extra_edge,
                },
            },
        }


class MinimumSpanningTreeCurriculum(BaseCurriculum):
    """Curriculum for Minimum Spanning Tree dataset"""

    def __init__(self):
        super().__init__(MinimumSpanningTreeCurriculum.__name__, MinimumSpanningTreeConfig)

        self._define_attributes(
            RangeAttributeDefinition(
                name="nodes",
                levels=[4, 8, 12, 16],
                description="Number of nodes in the graph",
                lower_field_name="min_nodes",
                upper_field_name="max_nodes",
            ),
            ScalarAttributeDefinition(
                name="max_weight",
                levels=[10, 20, 50, 100],
                description="Maximum edge weight",
                field_name="max_weight",
            ),
            ScalarAttributeDefinition(
                name="p_extra_edge",
                levels=[0.2, 0.4, 0.6, 0.8],
                description="Probability of each additional edge beyond the spanning tree",
                field_name="p_extra_edge",
            ),
        )


register_dataset(DATASET_NAME, MinimumSpanningTreeDataset, MinimumSpanningTreeConfig, MinimumSpanningTreeCurriculum)
