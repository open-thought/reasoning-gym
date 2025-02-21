import json
from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from ..factory import ProceduralDataset, register_dataset


def generate_random_graph(rng, num_vertices, edge_probability=0.3):
    """
    Generate an undirected random graph.

    Args:
        num_vertices (int): The number of vertices.
        edge_probability (float): Probability for an edge to exist between any two vertices.

    Returns:
        tuple: (vertices, edges)
            - vertices: A list of vertex identifiers (0 to num_vertices-1).
            - edges: A list of tuples (u, v) representing undirected edges.
    """
    vertices = list(range(num_vertices))
    edges = []
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if rng.random() < edge_probability:
                edges.append((i, j))
    return vertices, edges


def generate_graph_coloring_puzzle(rng, num_vertices=10, edge_probability=0.3, num_colors=3):
    """
    Generates a graph coloring puzzle.

    Args:
        num_vertices (int): Number of vertices in the graph.
        edge_probability (float): Probability that an edge exists between any two vertices.
        num_colors (int): Number of allowed colors.

    Returns:
        dict: A dictionary with the following keys:
            - "vertices": List of vertices.
            - "edges": List of edges (tuples).
            - "num_colors": The number of allowed colors.
            - "color_options": A list of allowed colors (e.g., [1, 2, ..., num_colors]).
    """
    vertices, edges = generate_random_graph(rng, num_vertices, edge_probability)
    puzzle = {
        "vertices": vertices,
        "edges": edges,
        "num_colors": num_colors,
        "color_options": list(range(1, num_colors + 1)),
    }
    return puzzle


def verify_graph_coloring_solution(puzzle, coloring):
    """
    Verifies that a candidate coloring is a valid solution to the graph coloring puzzle.

    Args:
        puzzle (dict): The puzzle specification containing 'vertices', 'edges', and 'color_options'.
        coloring (dict): A dictionary mapping each vertex to a color. The keys can be integers or strings.

    Returns:
        tuple: (is_valid, message) where is_valid is a boolean and message is a string explanation.
    """
    vertices = puzzle["vertices"]
    edges = puzzle["edges"]
    allowed_colors = set(puzzle["color_options"])

    # Helper function to get a vertex's color regardless of key type.
    def get_color(vertex):
        # If the key matches as-is, return it.
        if vertex in coloring:
            return coloring[vertex]
        # If the vertex is an integer and its string form is a key, return that.
        elif isinstance(vertex, int) and str(vertex) in coloring:
            return coloring[str(vertex)]
        # If the vertex is a string, try to convert it to int and look it up.
        elif isinstance(vertex, str):
            try:
                vertex_int = int(vertex)
                if vertex_int in coloring:
                    return coloring[vertex_int]
            except ValueError:
                pass
        # If no matching key is found, signal an error.
        raise KeyError(f"Vertex {vertex} has not been assigned a color.")

    # Check that every vertex has been assigned a color.
    for vertex in vertices:
        try:
            get_color(vertex)
        except KeyError:
            return False, f"Not all vertices have been assigned a color (missing vertex {vertex})."

    # Check that only allowed colors are used.
    for vertex in vertices:
        try:
            color = get_color(vertex)
        except KeyError as e:
            return False, str(e)
        if color not in allowed_colors:
            return False, f"Vertex {vertex} uses an invalid color: {color}."

    # Ensure that adjacent vertices do not share the same color.
    for u, v in edges:
        try:
            color_u = get_color(u)
            color_v = get_color(v)
        except KeyError as e:
            return False, str(e)
        if color_u == color_v:
            return False, f"Adjacent vertices {u} and {v} both have color {color_u}."

    return True, "The coloring is valid."


def greedy_graph_coloring(puzzle):
    """
    Attempts to color the graph using a simple greedy algorithm.
    (Note: This may fail if the graph requires more than the given number of colors.)

    Args:
        puzzle (dict): The puzzle specification.

    Returns:
        dict or None: A dictionary mapping vertices to colors if successful; otherwise, None.
    """
    vertices = puzzle["vertices"]
    edges = puzzle["edges"]
    color_options = puzzle["color_options"]

    # Build an adjacency list for each vertex.
    adjacency = {v: set() for v in vertices}
    for u, v in edges:
        adjacency[u].add(v)
        adjacency[v].add(u)

    coloring = {}
    for v in vertices:
        # Find colors already used by neighbors.
        neighbor_colors = {coloring.get(neighbor) for neighbor in adjacency[v] if neighbor in coloring}
        # Pick the first available color not used by any neighbor.
        available = [color for color in color_options if color not in neighbor_colors]
        if not available:
            return None  # Failed to color with the given number of colors.
        coloring[v] = available[0]
    return coloring


@dataclass
class GraphColorConfig:
    """Configuration for GraphColor puzzle generation"""

    num_colors: int = 4
    num_vertices: int = 10
    edge_probability: float = 0.4
    seed: Optional[int] = None
    size: int = 500

    def validate(self):
        """Validate configuration parameters"""
        assert self.edge_probability < 1, "edge_probability must be less than 1"


class GraphColorDataset(ProceduralDataset):
    """Generates graph coloring problems with configurable parameters"""

    def __init__(self, config: GraphColorConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def __getitem__(self, idx: int) -> dict:
        """Generate a single GraphColor task

        Returns:
            dict with keys:
                - question: str, the task description
                - answer: str, a solution string
                - metadata: dict with generation parameters
        """
        rng = Random(self.seed + idx)

        puzzle = None
        solution = None
        while solution is None:
            puzzle = generate_graph_coloring_puzzle(
                rng=rng,
                num_vertices=self.config.num_vertices,
                edge_probability=self.config.edge_probability,
                num_colors=self.config.num_colors,
            )
            solution = greedy_graph_coloring(puzzle)

        edges = str(puzzle["edges"])
        question = f"""Please provide a coloring for this graph such that every vertex is not connected to a vertex of the same color. The graph has these properties:

Vertices: {puzzle["vertices"]}
Edges: {edges}
Possible colors: {puzzle["color_options"]}

Return your solution as a JSON map of vertices to colors. (For example: {{0: 1, 1: 2, 2: 3}})
"""

        return {
            "question": question,
            "answer": None,
            "metadata": {"possible_answer": solution, "puzzle": puzzle},
        }

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Determine if the solution provided solves the GraphColor task.

        The function awards 1.0 for a correct answer.

        Args:
            answer (Optional[str]): The user's answer.
            entry (dict[str, Any]): The original dataset entry containing the correct answer.

        Returns:
            float: The computed score between 0.0 and 1.0.
        """

        if answer == None:
            return 0.0

        danswer = json.loads(answer)
        solved, failure = verify_graph_coloring_solution(entry["metadata"]["puzzle"], danswer)
        if not solved:
            return 0.01
        else:
            return 1.0  # Yay


register_dataset("graph_color", GraphColorDataset, GraphColorConfig)
