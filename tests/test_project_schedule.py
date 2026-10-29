"""Tests for project scheduling with durations and parallel execution."""

import json
from itertools import product

import pytest

from reasoning_gym import create_dataset
from reasoning_gym.factory import create_curriculum
from reasoning_gym.graphs import ProjectScheduleConfig, ProjectScheduleDataset
from reasoning_gym.graphs.project_schedule import ProjectScheduleCurriculum


def _simulate_schedule(durations, prerequisites):
    """Independently simulate all available workers one time unit at a time."""
    finishes = [None] * len(durations)
    running = {}
    started = set()
    for time in range(sum(durations) + 1):
        for task, finish_time in list(running.items()):
            if finish_time == time:
                finishes[task] = time
                del running[task]
        if all(finish is not None for finish in finishes):
            return finishes
        for task, parents in enumerate(prerequisites):
            if task not in started and all(finishes[parent] is not None for parent in parents):
                running[task] = time + durations[task]
                started.add(task)
        assert running, "Generated prerequisites contain a cycle"
    raise AssertionError("Schedule did not finish within its serial execution time")


def _dependency_depths(prerequisites):
    """Remove ready tasks in rounds to measure the graph's depth."""
    remaining = set(range(len(prerequisites)))
    depths = [0] * len(prerequisites)
    depth = 0
    while remaining:
        depth += 1
        ready = {task for task in remaining if not remaining.intersection(prerequisites[task])}
        assert ready, "Generated prerequisites contain a cycle"
        for task in ready:
            depths[task] = depth
        remaining -= ready
    return depths


@pytest.mark.parametrize(
    "overrides",
    [
        {"min_num_tasks": 0},
        {"min_num_tasks": 13, "max_num_tasks": 12},
        {"min_duration": 0},
        {"min_duration": 10, "max_duration": 9},
        {"dependency_depth": 0},
        {"dependency_depth": 7},
        {"max_prerequisites": 0},
        {"size": 0},
    ],
)
def test_project_schedule_config_rejects_invalid_bounds(overrides):
    with pytest.raises(AssertionError):
        ProjectScheduleConfig(**overrides).validate()


@pytest.mark.parametrize(
    "field",
    [
        "min_num_tasks",
        "max_num_tasks",
        "min_duration",
        "max_duration",
        "dependency_depth",
        "max_prerequisites",
        "size",
    ],
)
@pytest.mark.parametrize("value", [True, 3.5, "6"])
def test_project_schedule_config_requires_integers(field, value):
    with pytest.raises(AssertionError):
        ProjectScheduleConfig(**{field: value}).validate()


@pytest.mark.parametrize(
    "durations, prerequisites, expected",
    [
        ([7], [[]], [7]),
        ([2, 3, 5], [[], [0], [1]], [2, 5, 10]),
        ([3, 2, 4, 2], [[], [], [0, 1], [1]], [3, 2, 7, 4]),
        ([2, 3, 4, 5], [[], [0], [0], [1, 2]], [2, 5, 6, 11]),
        ([8, 2, 3, 4], [[], [], [1], []], [8, 2, 5, 4]),
        ([4, 3, 2], [[2], [2], []], [6, 5, 2]),
        ([3, 3, 1], [[], [], [0, 1]], [3, 3, 4]),
    ],
    ids=["singleton", "chain", "fork_join", "diamond", "disconnected", "unordered_ids", "ties"],
)
def test_project_schedule_earliest_finish_times(durations, prerequisites, expected):
    assert ProjectScheduleDataset._earliest_finish_times(durations, prerequisites) == expected


@pytest.mark.parametrize("prerequisites", [[[0]], [[1], [0]], [[], [2], [1]]])
def test_project_schedule_oracle_rejects_cycles(prerequisites):
    with pytest.raises(ValueError):
        ProjectScheduleDataset._earliest_finish_times([1] * len(prerequisites), prerequisites)


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"min_num_tasks": 1, "max_num_tasks": 1, "dependency_depth": 1},
        {"min_num_tasks": 5, "max_num_tasks": 5, "dependency_depth": 1},
        {"min_num_tasks": 4, "max_num_tasks": 4, "dependency_depth": 4, "max_prerequisites": 1},
        {"min_num_tasks": 7, "max_num_tasks": 9, "dependency_depth": 3, "max_prerequisites": 1},
        {"min_num_tasks": 4, "max_num_tasks": 4, "dependency_depth": 3, "max_prerequisites": 20},
        {"min_duration": 5, "max_duration": 5},
    ],
)
def test_project_schedule_generated_graphs_and_answers(overrides):
    config = ProjectScheduleConfig(seed=42, size=12, **overrides)
    config.validate()
    dataset = ProjectScheduleDataset(config)
    for index, entry in enumerate(dataset):
        metadata = entry["metadata"]
        assert metadata["source_dataset"] == "project_schedule"
        assert metadata["source_index"] == index
        assert "difficulty" in metadata
        durations = metadata["durations"]
        prerequisites = metadata["prerequisites"]
        num_tasks = len(durations)
        assert config.min_num_tasks <= num_tasks <= config.max_num_tasks
        assert len(prerequisites) == num_tasks
        assert all(
            type(duration) is int and config.min_duration <= duration <= config.max_duration for duration in durations
        )
        for task, parents in enumerate(prerequisites):
            assert len(parents) == len(set(parents))
            assert len(parents) <= config.max_prerequisites
            assert all(type(parent) is int and 0 <= parent < num_tasks and parent != task for parent in parents)

        depths = _dependency_depths(prerequisites)
        assert max(depths) == config.dependency_depth
        for task, parents in enumerate(prerequisites):
            assert all(depths[parent] < depths[task] for parent in parents)
            if parents:
                assert any(depths[parent] == depths[task] - 1 for parent in parents)
        if config.dependency_depth > 1 and num_tasks > config.dependency_depth:
            assert sum(not parents for parents in prerequisites) >= 2
            if config.max_prerequisites >= 2:
                assert any(len(parents) >= 2 for parents in prerequisites)

        expected = _simulate_schedule(durations, prerequisites)
        assert metadata["solution"] == expected
        assert json.loads(entry["answer"]) == expected
        assert dataset.score_answer(entry["answer"], entry) == 1.0


def test_project_schedule_question_specifies_execution_and_answer_rules():
    entry = ProjectScheduleDataset(ProjectScheduleConfig(seed=1))[0]
    question = entry["question"].lower()
    assert "time 0" in question
    assert "unlimited" in question
    assert "parallel" in question
    assert "json" in question and "array" in question
    assert "exact instant" in question and "finish" in question


def test_project_schedule_deterministic_random_access_and_iteration():
    config = ProjectScheduleConfig(seed=17, size=8)
    dataset = ProjectScheduleDataset(config)
    reference = ProjectScheduleDataset(config)
    for index in [7, 2, 0, 5, 2, 7]:
        assert dataset[index] == reference[index]
    assert len(dataset) == config.size
    assert list(dataset) == list(reference) == list(dataset)


def test_project_schedule_task_labels_are_not_topological_order():
    dataset = ProjectScheduleDataset(ProjectScheduleConfig(seed=23, size=25))
    assert any(
        parent > task
        for entry in dataset
        for task, parents in enumerate(entry["metadata"]["prerequisites"])
        for parent in parents
    )


def test_project_schedule_factory_registration():
    dataset = create_dataset("project_schedule", seed=42, size=3)
    assert isinstance(dataset, ProjectScheduleDataset)
    assert len(dataset) == 3
    assert isinstance(create_curriculum("project_schedule"), ProjectScheduleCurriculum)


@pytest.mark.parametrize(
    "answer, expected_score",
    [
        ("[1]", 1.0),
        ("\n [ 1 ] \t", 1.0),
        (None, 0.0),
        ("", 0.0),
        ("[0]", 0.0),
        ("[]", 0.0),
        ("[1, 1]", 0.0),
        ("[1.0]", 0.0),
        ("[1e0]", 0.0),
        ("[true]", 0.0),
        ('["1"]', 0.0),
        ("[[1]]", 0.0),
        ("1", 0.0),
        ('{"0": 1}', 0.0),
        ("[1,]", 0.0),
        ("[1] trailing text", 0.0),
        ("```json\n[1]\n```", 0.0),
    ],
)
def test_project_schedule_strict_json_scoring(answer, expected_score):
    dataset = ProjectScheduleDataset(
        ProjectScheduleConfig(
            min_num_tasks=1,
            max_num_tasks=1,
            min_duration=1,
            max_duration=1,
            dependency_depth=1,
            seed=42,
        )
    )
    assert dataset.score_answer(answer, dataset[0]) == expected_score


def test_project_schedule_scoring_requires_entire_solution():
    dataset = ProjectScheduleDataset(ProjectScheduleConfig(seed=42))
    entry = dataset[0]
    wrong = json.loads(entry["answer"])
    wrong[-1] += 1
    assert dataset.score_answer(json.dumps(wrong), entry) == 0.0


def test_project_schedule_curriculum_defaults_and_progression():
    curriculum = ProjectScheduleCurriculum()
    defaults = {"seed": 123, "size": 7}
    config = curriculum.generate_configuration(defaults)
    assert config == ProjectScheduleConfig(**defaults)
    assert curriculum.get_attr_level("dependency_depth") == 1
    assert curriculum.get_attr_level("max_prerequisites") == 1

    curriculum.increment_attr_level("num_tasks")
    curriculum.increment_attr_level("duration")
    curriculum.increment_attr_level("dependency_depth")
    curriculum.increment_attr_level("max_prerequisites")
    increased = curriculum.generate_configuration(defaults)
    assert (increased.min_num_tasks, increased.max_num_tasks) == (6, 24)
    assert (increased.min_duration, increased.max_duration) == (1, 20)
    assert increased.dependency_depth == 4
    assert increased.max_prerequisites == 4
    increased.validate()

    curriculum.decrement_attr_level("num_tasks")
    decreased = curriculum.generate_configuration(defaults)
    assert (decreased.min_num_tasks, decreased.max_num_tasks) == (6, 12)
    assert decreased.dependency_depth == 4


def test_project_schedule_all_curriculum_level_combinations_are_valid():
    curriculum = ProjectScheduleCurriculum()
    attributes = list(curriculum.attributes)
    levels = [range(len(curriculum.attributes[name].levels)) for name in attributes]
    for combination in product(*levels):
        for name, level in zip(attributes, combination):
            curriculum.set_attr_level(name, level)
        config = curriculum.generate_configuration({"seed": 42, "size": 1})
        config.validate()
        assert config.seed == 42 and config.size == 1
        assert config.dependency_depth <= config.min_num_tasks
