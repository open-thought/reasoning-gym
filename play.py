import time
from contextlib import contextmanager

import reasoning_gym


@contextmanager
def timer():
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    elapsed_minutes = (end - start) / 60
    print(f"Time taken: {elapsed_minutes:.2f} minutes")


# Usage
with timer():
    # Your code here
    data = reasoning_gym.create_dataset("knight_swap", size=10000, seed=1)
    for i, x in enumerate(data):
        # print(f"{i}: q={x['question']}, a={x['answer']}")
        # print('metadata:', x['metadata'])
        # use the dataset's `score_answer` method for algorithmic verification
        assert data.score_answer(answer=x["answer"], entry=x) == 1.0
