import reasoning_gym
data = reasoning_gym.create_dataset('polynomial_equations', size=3, seed=42)
for i, x in enumerate(data):
    print(f"{i}: question={x['question']}\n")
    print(f"{i}: answer={x['answer']}\n")
    print('metadata:', x['metadata'])