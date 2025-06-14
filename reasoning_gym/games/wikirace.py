import re
from collections import defaultdict, deque
from dataclasses import dataclass
from functools import lru_cache
from random import Random
from typing import Any, Optional

from ..coaching import BaseCurriculum, RangeAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

try:
    from datasets import load_dataset
except:
    raise Exception("wikirace requires datasets library. Run `pip install datasets`")

QUESTION_FORMAT_TEMPLATE = """
You are playing WikiRace, trying to navigate from one Wikipedia article to another using only links.

Answer with just the link number.

Current article: {current}
Target article: {target}
Available links (numbered):
{formatted_links}

Your path so far: {formatted_path}

Think about which link is most likely to lead you toward the target article.
First, analyze each link briefly and how it connects to your goal, then select the most promising one.
"""


DATASET_NAME = "wikirace"


@dataclass
class WikiraceConfig:
    """Configuration for WikiRace task generation"""

    min_distance: int = 3
    max_distance: int = 6
    max_tries: int = 100
    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.min_distance > 1, "min_distance must be greater than 1"
        assert self.max_distance >= self.min_distance, "max_distance must be >= min_distance"

        assert self.max_tries >= 1, "max_tries must be greater than 1"


def load_wiki_graph():
    dataset = load_dataset("HuggingFaceTB/simplewiki-pruned-350k")

    graph = defaultdict(set)
    titles = set()

    # Build the graph
    for example in dataset["train"]:
        title = example["article"]
        links = example["links"]

        titles.add(title)

        for link in links:
            graph[title].add(link)

    # Note: Since titles was a set, and hash are naturally unstable
    # We want to sort it, so that prng.choice() is stable
    return graph, sorted(list(titles))


class WikiraceDataset(ProceduralDataset):
    """Generates Wikirace Game tasks"""

    def __init__(self, config: WikiraceConfig):
        self.wikigraph, self.wikititles = load_wiki_graph()
        super().__init__(config=config, seed=config.seed, size=config.size)

    # We'll be computing a lot of shortest_path of very similar paths
    # So cache it
    @lru_cache(maxsize=128 * 1024)
    def shortest_path(self, source, target):
        if source not in self.wikigraph or target not in self.wikigraph:
            return None

        if source == target:
            return 1, [source]

        visited = {source}
        queue = deque([(source, [source], 0)])

        while queue:
            current_node, path, l = queue.popleft()
            for neighbor in self.wikigraph[current_node]:
                if neighbor == target:
                    return 1 + l, (path + [neighbor])
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor], 1 + l))

        return None  # No path found

    def __getitem__(self, idx: int) -> dict:
        """Generate a single Wikirace Game task

        Returns:
            dict with keys:
                - question: str, the task description with a source article, target article, and current chosen path
                - answer: str, one possible article on the shortest path
                - metadata: dict with generation parameters
        """
        rng = Random(self.seed + idx)

        # Find a task that suits our min_distance/max_distance
        # Since some pages might be dead-ends, we might need to try multiple times
        for _ in range(self.config.max_tries):
            source = rng.choice(self.wikititles)
            target = source
            chosen_distance = rng.randint(self.config.min_distance, self.config.max_distance)
            path = [source]
            length = 0
            while self.shortest_path(source, target)[0] != chosen_distance:
                possibilities = self.wikigraph[target] - set(path)
                if not possibilities:
                    break
                # Since hash() is random, we need to sort the set into a list
                # for prng stability
                possibilities = sorted(list(possibilities))
                target = rng.choice(possibilities)
                length += 1
                # Are we lost? Are we looping? Aborting
                if length > 12:
                    break
                path.append(target)
            if self.shortest_path(source, target)[0] == chosen_distance:
                break
            # We got lost in a loop  or a dead end, try again

        if self.shortest_path(source, target)[0] != chosen_distance:
            raise Exception(f"After {self.config.max_tries}, we failed to find a suitable wikipedia articles pair")

        _, path = self.shortest_path(source, target)
        # This is the length of the current path (let's call it state)
        # 0 mean that we are still at the source of the path we're searching for
        path_len = rng.randint(0, min(self.config.min_distance, len(path)) - 2)
        given_path = path[:path_len]
        given_path = " => ".join(given_path)
        current = path[path_len]
        # Stable links
        links = sorted(list(self.wikigraph[current]))
        links = list(enumerate(links))
        question = QUESTION_FORMAT_TEMPLATE.format(
            current=current,
            target=target,
            formatted_links=[f"{x[0]} - {x[1]}\n" for x in links],
            formatted_path=given_path,
        )
        answer = [x[0] for x in links if x[1] == path[path_len + 1]][0]

        return {
            "question": question,
            "answer": str(answer),
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "source": source,
                "current": current,
                "target": target,
                "distance": chosen_distance,
                "path": given_path,
                "remaining_path": path[path_len:],
                "links": links,
                "difficulty": {
                    "distance": (self.config.min_distance, self.config.max_distance),
                },
            },
        }

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Determine if the solution provided solves the problem"""
        reward = 0.01  # Default reward
        source = entry["metadata"]["source"]
        target = entry["metadata"]["target"]
        current = entry["metadata"]["current"]
        links = entry["metadata"]["links"]

        if answer is None or not answer.strip():
            return reward

        try:
            answer = answer.strip()
            answer = int(answer)
            if answer < 0:
                return 0.01
            link = links[answer][1]
            new_distance = self.shortest_path(link, target)[0]
            old_distance = self.shortest_path(current, target)[0]
            if new_distance < old_distance:
                # Path is shortet than before, it is following (a) shortest path!
                return 1.0
            elif new_distance == old_distance:
                # Path isn't shorter, but not longer either, that's still something
                return 0.5
            else:
                # At least answer is valid...
                return 0.1

        except Exception:
            return 0.01


class WikiraceCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(WikiraceCurriculum.__name__, WikiraceConfig)

        # Define attributes
        self._define_attributes(
            RangeAttributeDefinition(
                name="distance",
                levels=[3, 6, 9, 12, 15],
                description="Number of source numbers",
                lower_field_name="min_distance",
                upper_field_name="max_distance",
                ensure_interval=True,
            ),
            ScalarAttributeDefinition(
                name="max_tries",
                description="Max number of tries to find test cases",
                field_name="max_tries",
            ),
        )


# Register the dataset
register_dataset(DATASET_NAME, WikiraceDataset, WikiraceConfig, WikiraceCurriculum)
