import reasoning_gym

data = reasoning_gym.create_dataset("acre", size=10, seed=42)
for i, x in enumerate(data):
    print(f"examples: {len(x['examples'])}\n")
    print(f"question: {x['question']}\n")

print(f"{len(data)} is processed")
