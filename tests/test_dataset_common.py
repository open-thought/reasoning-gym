import reasoning_gym
from reasoning_gym.factory import DATASETS


def test_score_answer_consistency():
    for dataset_name in DATASETS.keys():
        dataset = reasoning_gym.create_dataset(dataset_name, size=10, seed=1234)
        for entry in dataset:
            assert entry["answer"] is None or isinstance(
                entry["answer"], str
            ), f"{dataset_name} answer must be str, is {type(entry['answer'])}"
            if entry["answer"] is not None:
                assert (
                    dataset.score_answer(answer=entry["answer"], entry=entry) == 1.0
                ), f"inconsistent score_answer {dataset_name}"
