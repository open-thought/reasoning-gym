"""
Determine if you can complete all courses given their prerequisite relationships.

A popular topological sort Leetcode problem:
https://leetcode.com/problems/course-schedule/description/
"""

from collections import defaultdict
from dataclasses import dataclass
from random import Random
from typing import List, Optional

from ..factory import ProceduralDataset, register_dataset

MAX_NUM_COURSES = 1_000

QUESTION_TEMPLATE = """There are a total of {num_courses} courses you have to take, labeled from 0 to {last_index}.

You are given the following list of prerequisites, where prerequisites[i] = (a_i, b_i) indicates that you must first take course b_i if you want to take course a_i:
{prerequisites}

Return True if you can finish all courses considering the prerequisites, or False otherwise.
"""


@dataclass
class CourseScheduleConfig:
    """Configuration for Course Schedule dataset generation"""

    num_courses: int = 5  # Total number of courses (ranging from 0 to num_courses - 1)
    max_num_prerequisites: int = 2  # Maximum number of prerequisites (per course)
    p_solvable: float = 0.5  # Probability that the course schedule is solvable
    min_cycle_length: int = 3  # Minimum length of a cycle in the prerequisites (if unsolvable)
    max_cycle_length: int = 5  # Maximum length of a cycle in the prerequisites (if unsolvable)

    size: int = 500  # Virtual dataset size
    seed: Optional[int] = None

    def validate(self):
        """Validate configuration parameters"""
        assert 1 <= self.num_courses <= MAX_NUM_COURSES, f"num_courses must be between 1 and {MAX_NUM_COURSES}"
        assert (
            1 <= self.max_num_prerequisites <= self.num_courses
        ), "max_num_prerequisites must be between 1 and num_courses"
        assert 0 <= self.p_solvable <= 1, "p_solvable must be between 0 and 1"
        assert (
            3 <= self.min_cycle_length <= self.max_cycle_length
        ), "min_cycle_length must be between 3 and max_cycle_length"


class CourseScheduleDataset(ProceduralDataset):
    """Generates Course Schedule exercises with configurable difficulty"""

    def __init__(self, config: CourseScheduleConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _can_finish(self, num_courses: int, prerequisites: List[List[int]]) -> bool:
        adj = defaultdict(list)
        for course, prereq in prerequisites:
            adj[course].append(prereq)

        visited, cycle = set(), set()

        def topological_sort(idx):
            if idx in cycle:
                return False
            if idx in visited:
                return True

            cycle.add(idx)
            for nei in adj[idx]:
                if not topological_sort(nei):
                    return False
            cycle.remove(idx)
            visited.add(idx)
            return True

        for i in range(num_courses):
            if not topological_sort(i):
                return False

        return True

    def _create_prerequisites(self, rng: Random, courses: List[int], solvable: bool) -> List[List[int]]:
        """Create a list of prerequisites for each course"""
        prerequisites = []
        # Generate a valid course schedule
        for idx in range(len(courses) - 1, 0, -1):
            current_course = courses[idx]
            available_prereqs = courses[:idx]  # Only earlier courses can be prerequisites
            num_prerequisites = rng.randint(0, min(len(available_prereqs), self.config.max_num_prerequisites))
            if num_prerequisites > 0:
                chosen_prereqs = rng.sample(available_prereqs, num_prerequisites)
                prerequisites.extend([[current_course, p] for p in chosen_prereqs])

        if not solvable:
            # If solution should be unsolvable, create a cycle
            cycle_length = rng.randint(self.config.min_cycle_length, min(self.config.max_cycle_length, len(courses)))
            cycle_courses = rng.sample(courses, cycle_length)
            for i in range(cycle_length):
                prerequisites.append([cycle_courses[i], cycle_courses[(i + 1) % cycle_length]])

        # remove potential duplicates
        prerequisites = list(set(tuple(prereq) for prereq in prerequisites))
        rng.shuffle(prerequisites)
        return prerequisites

    def __getitem__(self, idx: int) -> dict:
        """Generate a single Course Schedule question"""
        rng = Random(self.seed + idx)

        courses = list(range(self.config.num_courses))
        rng.shuffle(courses)

        solvable = rng.random() < self.config.p_solvable

        prerequisites = self._create_prerequisites(rng, courses, solvable)
        answer = self._can_finish(self.config.num_courses, prerequisites)

        return {
            "question": QUESTION_TEMPLATE.format(
                num_courses=self.config.num_courses,
                last_index=self.config.num_courses - 1,
                prerequisites=str(prerequisites),
            ),
            "answer": str(answer),
            "metadata": {"courses": courses, "prerequisites": prerequisites, "solution": answer, "solvable": solvable},
        }


register_dataset("course_schedule", CourseScheduleDataset, CourseScheduleConfig)
