from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import random
import re

from clrs._src import samplers as clrs_samplers
from clrs._src.clrs_text import clrs_utils

from ..factory import ProceduralDataset, register_dataset


@dataclass
class ClrsConfig:
    """
    Configuration for a CLRS subtask dataset.

    - subtask: Name of the CLRS algorithm/subtask, e.g. "heapsort", "bfs".
    - min_length: Minimum number of length.
    - max_length: Maximum number of length.
    - seed: Optional random seed for reproducibility.
    - size: Number of examples in the dataset.
    - num_decimals_in_float: Truncation for float decimals in the sampler
      (see clrs samplers docs). None = no truncation.

    Supported subtasks:
        'activity_selector'
        'articulation_points'
        'bellman_ford'
        'bfs'
        'binary_search'
        'bridges'
        'bubble_sort'
        'dag_shortest_paths'
        'dfs'
        'dijkstra'
        'find_maximum_subarray_kadane'
        'floyd_warshall'
        'graham_scan'
        'heapsort'
        'insertion_sort'
        'jarvis_march'
        'kmp_matcher'
        'lcs_length'
        'matrix_chain_order'
        'minimum'
        'mst_kruskal'
        'mst_prim'
        'naive_string_matcher'
        'optimal_bst'
        'quickselect'
        'quicksort'
        'segments_intersect'
        'strongly_connected_components'
        'task_scheduling'
        'topological_sort'
    """

    subtask: str
    min_length: int
    max_length: int
    seed: Optional[int] = None
    size: int = 50
    num_decimals_in_float: Optional[int] = None

    def validate(self):
        assert self.min_length > 0, "min_length must be positive"
        assert self.max_length >= self.min_length, "max_length must be >= min_length"
        assert len(self.subtask) > 0, "subtask must be non-empty"


class ClrsDataset(ProceduralDataset):
    """
    Dataset that generates CLRS examples for a single subtask, each with a
    randomly chosen length in [min_length, max_length].
    """

    def __init__(self, config: ClrsConfig):
        config.validate()
        super().__init__(seed=config.seed, size=config.size, config=config)
        self.config = config

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns:
          A dict with keys:
            - "question": The text prompt from CLRS (the 'prompt' field).
            - "answer": The last reference from CLRS (string).
            - "metadata": Additional info, including:
                * "length" (the length used in this sample)
                * "arrays" (any arrays we parsed from the prompt, e.g. 'key')
        """
        rng = random.Random(self.seed + idx)

        # Pick a random length in [min_length, max_length]
        length = rng.randint(self.config.min_length, self.config.max_length)

        # Build a sampler for this subtask, with the chosen length
        sampler, _ = clrs_samplers.build_sampler(
            name=self.config.subtask,
            seed=self.seed + idx,
            num_samples=-1,  # data generated on the fly
            length=length,
            track_max_steps=False,
            truncate_decimals=self.config.num_decimals_in_float,
        )

        # Get a single sample from the sampler
        #   (batch_size=1 => we get exactly one example)
        sample = sampler.next(batch_size=1)
        # Format the result using clrs_text utilities
        question_text, answer_text = clrs_utils.format_clrs_example(
            algo=self.config.subtask,
            sample=sample,
            use_hints=False,
        )

        return {
            "question": question_text,
            "answer": answer_text.strip(),
            "metadata": {
                "length": length,
                "arrays": self._parse_question(self.config.subtask, question_text),
            },
        }

    def _parse_question(self, subtask: str, question: str) -> Dict[str, Any]:
        """
        Parse the question text for known lines, arrays, or scalars
        that differ by subtask. We'll demonstrate a few tasks
        as examples. You can add more as needed.
        """
        if subtask == "activity_selector":
            return self._parse_activity_selector(question)
        elif subtask == "bfs":
            return self._parse_bfs(question)
        elif subtask == "bellman_ford":
            return self._parse_bellman_ford(question)
        elif subtask == "binary_search":
            return self._parse_binary_search(question)
        elif subtask == "bubble_sort":
            return self._parse_bubble_sort(question)
        else:
            return self._parse_generic_arrays(question)

    def _parse_activity_selector(self, question: str) -> Dict[str, Any]:
        """
        Example question:
          "activity_selector:\n
           s: [0.6964691855978616 0.28613933495037946 ...], f: [0.7194689697 ...]\n
           selected:\n"
        We want to parse the line with "s: [...]" and "f: [...]" as 1D arrays.
        """
        arrays_dict: Dict[str, Any] = {"s": [], "f": []}

        lines = question.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("s: ["):
                [s, f] = line.split("f:")
                arrays_dict["s"] = self._parse_1d_float_array(s)
                arrays_dict["f"] = self._parse_1d_float_array(f)
        return arrays_dict

    def _parse_bfs(self, question: str) -> Dict[str, Any]:
        """
        Example question:
          "bfs:\n
           s: 3, A: [[1 0 0 0], [0 0 0 0], ...]\n
           pi:\n"
        We'll parse 's' as an integer, 'A' as a 2D adjacency matrix.
        """
        arrays_dict: Dict[str, Any] = {"s": None, "A": []}

        # s: 3, A: [[1 0 0 0], [0 0 0 0], [0 0 0 0], [0 0 0 1]]
        # We'll find 's:' as an integer, 'A:' as 2D
        lines = question.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("s:"):
                # could be "s: 3, A: [[1 0 0 ...]]"
                # We'll separate by comma
                parts = re.split(r", (?=[a-zA-Z])", line)
                # The first part should be "s: 3"
                s_part = parts[0].replace("s:", "").strip()
                arrays_dict["s"] = int(s_part)

                # The second part might have "A: [[...]]"
                if len(parts) > 1 and "A:" in parts[1]:
                    a_str = parts[1].split("A:")[1].strip()
                    arrays_dict["A"] = self._parse_2d_array(a_str)

            elif line.startswith("A: [["):
                # e.g. A: [[1 0 0 0], [0 0 0 0], ...]
                arrays_dict["A"] = self._parse_2d_array(line.replace("A:", "").strip())

        return arrays_dict

    def _parse_bellman_ford(self, question: str) -> Dict[str, Any]:
        """
        e.g. "bellman_ford:\n
              s: 2, A: [[0.1852 0.0 ...], [0.0 0.0 ...], ...]\n
              pi:\n"
        """
        arrays_dict = {"s": None, "A": []}

        lines = question.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("s:"):
                # "s: 2, A: [[0.1852 0.0 ...]]"
                parts = re.split(r", (?=[a-zA-Z])", line)
                s_part = parts[0].replace("s:", "").strip()
                arrays_dict["s"] = int(s_part)
                # second part might be " A: [[...]]"
                if len(parts) > 1 and "A:" in parts[1]:
                    a_str = parts[1].split("A:")[1].strip()
                    arrays_dict["A"] = self._parse_2d_array(a_str)
            elif line.startswith("A: [["):
                arrays_dict["A"] = self._parse_2d_array(line.replace("A:", "").strip())
        return arrays_dict

    def _parse_binary_search(self, question: str) -> Dict[str, Any]:
        """
        e.g. "binary_search:\n
              key: [0.2268 0.2861 0.5513 0.6964], target: 0.7194\n
              return:\n"
        We'll parse "key" as a 1D array, "target" as float
        """
        arrays_dict = {"key": [], "target": None}
        lines = question.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("key: ["):
                # might be "key: [0.2268 0.2861 ...], target: 0.7194"
                # separate by comma
                parts = line.split(",")
                # first part => "key: [0.2268 0.2861 ...]"
                arrays_dict["key"] = self._parse_1d_float_array(parts[0])

                # second part => " target: 0.7194"
                if len(parts) > 1 and "target:" in parts[1]:
                    t_str = parts[1].replace("target:", "").strip()
                    arrays_dict["target"] = int(t_str) if "." not in t_str else float(t_str)
            elif line.startswith("target:"):
                # If it's on its own line
                t_str = line.replace("target:", "").strip()
                arrays_dict["target"] = int(t_str) if "." not in t_str else float(t_str)
        return arrays_dict

    def _parse_bubble_sort(self, question: str) -> Dict[str, Any]:
        """
        e.g. "bubble_sort:\n
              key: [0.6964 0.2861 0.2268 0.5513]\n
              pred:\n"
        We'll parse 'key' as a 1D float array.
        """
        arrays_dict = {"key": []}
        lines = question.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("key: ["):
                arrays_dict["key"] = self._parse_1d_float_array(line)
        return arrays_dict

    def _parse_generic_arrays(self, question: str) -> Dict[str, Any]:
        """
        Fallback parser that tries to parse lines matching:
          X: [1 2 3]
          X: [[1 0 0], [0 1 0], ...]
          X: 42
        and stores them in arrays[X].
        This won't perfectly handle all tasks, but it's a start.
        """
        arrays_dict: Dict[str, Any] = {}

        lines = question.split("\n")
        # skip the first line "<subtask>:" e.g. "bfs:"
        # because it's just the subtask name
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            # Example forms:
            # "s: 3, A: [[1 0 0 0], [0 0 0 0], ...]"
            # "A: [[1 0], [0 1]]"
            # "key: [0.1 0.2 0.3]"
            # "x: 2"
            if ":" not in line:
                continue

            # We handle multiple fields on one line if separated by commas and a letter
            parts = [p.strip() for p in re.split(r", (?=[a-zA-Z])", line)]
            for part in parts:
                if ":" not in part:
                    continue
                field, raw_val = part.split(":", 1)
                field = field.strip()
                raw_val = raw_val.strip()
                # Decide if it's 1D array, 2D array, or single value
                if raw_val.startswith("[["):
                    # 2D array
                    arr = self._parse_2d_array(raw_val)
                    arrays_dict[field] = arr
                elif raw_val.startswith("["):
                    # 1D array
                    arr = self._parse_1d_array(raw_val)
                    arrays_dict[field] = arr
                else:
                    # possibly a single integer or float
                    # Try parse int first, fallback float
                    try:
                        val = int(raw_val)
                        arrays_dict[field] = val
                    except ValueError:
                        try:
                            arrays_dict[field] = int(raw_val) if "." not in raw_val else float(raw_val)
                        except ValueError:
                            # can't parse => store raw string
                            arrays_dict[field] = raw_val
        return arrays_dict

    # ----------------------------------------------------------------
    # Basic string -> array parsing utilities
    # ----------------------------------------------------------------
    def _parse_1d_float_array(self, line: str) -> List[float]:
        """
        Given a string like
          "key: [0.6964 0.2861 0.2268 0.5513]"
        or
          "[0.6964 0.2861 0.2268 0.5513]"
        returns [0.6964, 0.2861, ...]
        """
        # remove "key:" if present
        if ":" in line:
            line = line.split(":", 1)[1].strip()
        # remove brackets [...]
        line = line.strip()
        # e.g. "[0.6964 0.2861 0.2268 0.5513]"
        if line.startswith("["):
            line = line[1:]
        if line.endswith("]"):
            line = line[:-1]
        line = line.strip()
        if not line:
            return []
        # split by whitespace
        parts = line.split()
        floats = []
        for p in parts:
            try:
                floats.append(int(p) if "." not in p else float(p))
            except ValueError:
                pass
        return floats

    def _parse_1d_array(self, line: str) -> List[float]:
        """
        Same as _parse_1d_float_array but can also parse int.
        For simplicity, we treat them as floats. Adjust as needed.
        """
        return self._parse_1d_float_array(line)

    def _parse_2d_array(self, line: str) -> List[List[float]]:
        """
        e.g. "[[1 0 0 0], [0 0 0 0], [0 0 0 0], [0 0 0 1]]"
        -> [[1,0,0,0],[0,0,0,0],...]
        """
        # remove leading/trailing bracket
        line = line.strip()
        if line.startswith("[["):
            line = line[1:]  # remove one '[' so left is "[1 0 0 0], ..."
        if line.endswith("]]"):
            line = line[:-1]  # remove one ']' so left is "...], [0 0 0 0]"
        line = line.strip()
        # Now we can split by "],"
        row_strings = line.split("],")
        matrix = []
        for row_str in row_strings:
            r = row_str.strip()
            # remove leading '[' if any
            if r.startswith("["):
                r = r[1:]
            # remove trailing ']' if any
            if r.endswith("]"):
                r = r[:-1]
            # now split by whitespace
            parts = r.split()
            row = []
            for p in parts:
                try:
                    val = int(p) if "." not in p else float(p)
                except ValueError:
                    val = 0.0
                row.append(val)
            matrix.append(row)
        return matrix


register_dataset("clrs", ClrsDataset, ClrsConfig)
