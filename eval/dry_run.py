import argparse

from eval_config import EvalConfig

import reasoning_gym


def main():
    argparser = argparse.ArgumentParser(description="Evaluate reasoning gym datasets.")
    argparser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    args = argparser.parse_args()

    config_path = args.config
    if config_path.endswith(".yaml") or config_path.endswith(".yml"):
        config = EvalConfig.from_yaml(config_path)
    elif config_path.endswith(".json"):
        config = EvalConfig.from_json(config_path)
    else:
        print("Error: Configuration file must be YAML or JSON")
        return 1

    for category in config.categories:
        for dataset in category.datasets:
            rg_dataset = reasoning_gym.create_dataset(dataset.dataset, size=10, seed=42, **dataset.params)
            print(rg_dataset)


if __name__ == "__main__":
    main()
