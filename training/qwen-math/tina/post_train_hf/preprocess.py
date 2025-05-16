
def make_conv_for_grpo(example, system_prompt):
    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["problem"]},
        ]
    }
