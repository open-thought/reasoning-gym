import random
from typing import Dict, Any
from collections import deque

class TransitRoutingConfig:
    def __init__(self, min_stations: int = 5, max_stations: int = 15):
        self.min_stations = min_stations
        self.max_stations = max_stations

class TransitRoutingTask:
    """
    Procedurally generates a transit network to calculate the optimal route.
    Scope strictly excludes user registration, login, and journey history.
    """
    def __init__(self, config: TransitRoutingConfig):
        self.config = config

    def _generate_connected_graph(self, num_stations: int) -> Dict[str, list]:
        # Create a spanning tree to guarantee the network is fully connected
        stations = [f"Station_{i}" for i in range(num_stations)]
        random.shuffle(stations)
        
        graph = {station: [] for station in stations}
        for i in range(1, num_stations):
            u = stations[i]
            v = stations[random.randint(0, i - 1)]
            graph[u].append(v)
            graph[v].append(u)
            
        # Add a few random extra transit connections (edges)
        extra_edges = random.randint(1, num_stations // 2)
        for _ in range(extra_edges):
            u, v = random.sample(stations, 2)
            if v not in graph[u]:
                graph[u].append(v)
                graph[v].append(u)
                
        # Sort adjacency lists for deterministic output presentation
        for station in graph:
            graph[station].sort()
            
        return graph

    def _bfs_shortest_path(self, graph: Dict[str, list], start: str, target: str) -> list:
        queue = deque([[start]])
        visited = set([start])
        
        while queue:
            path = queue.popleft()
            node = path[-1]
            
            if node == target:
                return path
                
            for adjacent in graph.get(node, []):
                if adjacent not in visited:
                    visited.add(adjacent)
                    new_path = list(path)
                    new_path.append(adjacent)
                    queue.append(new_path)
        return []

    def generate(self) -> Dict[str, Any]:
        num_stations = random.randint(self.config.min_stations, self.config.max_stations)
        graph = self._generate_connected_graph(num_stations)
        
        stations = list(graph.keys())
        start, destination = random.sample(stations, 2)
        
        shortest_path = self._bfs_shortest_path(graph, start, destination)
        
        # Format the network description for the prompt
        network_description = ""
        for station, connections in graph.items():
            network_description += f"- {station} connects to: {', '.join(connections)}\n"
            
        prompt = (
            f"You are navigating a transit network. Find the shortest path (fewest stops) "
            f"from {start} to {destination}.\n\n"
            f"Transit Network Map:\n{network_description}\n"
            f"Output only the sequence of stations separated by ' -> '."
        )
        
        answer = " -> ".join(shortest_path)
        
        return {
            "question": prompt,
            "answer": answer,
            "metadata": {
                "source_dataset": "transit_routing",
                "difficulty": {
                    "min_stations": self.config.min_stations,
                    "max_stations": self.config.max_stations
                },
                "solution": shortest_path
            }
        }

# --- Local Testing Block ---
if __name__ == "__main__":
    # Initialize the config and task
    config = TransitRoutingConfig(min_stations=5, max_stations=8)
    task = TransitRoutingTask(config)
    
    # Generate a single problem instance
    sample = task.generate()
    
    # Print the output to verify
    print("=== PROMPT ===")
    print(sample["question"])
    print("\n=== ANSWER ===")
    print(sample["answer"])
    print("\n=== METADATA ===")
    print(sample["metadata"])