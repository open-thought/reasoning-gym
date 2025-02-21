import reasoning_gym

data = reasoning_gym.create_dataset("list_functions", size=3, seed=42)
for i, x in enumerate(data):
    print(f"{i}: q={x['question']}, a={x['answer']}")
    print("metadata:", x["metadata"])
