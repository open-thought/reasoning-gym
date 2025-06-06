# problem/question, (solution), answer
RL_POST_TRAIN_DATASET_MAP = {
    # Main datasets
    "curated_deepscaler": "agentica-org/DeepScaleR-Preview-Dataset",  # 40.3k
    "curated_still": "RUC-AIBOX/STILL-3-Preview-RL-Data",  # 33k
    "curated_open_rs3": "knoveleng/open-rs",  # 7k
    "curated_open_rs2": "knoveleng/open-rs",  # 7k
    "curated_open_rs1": "knoveleng/open-s1",  # 18.6k
    # Extra datasets
    "curated_limr": "GAIR/LIMR",  # 1.39k
    "curated_open_r1": "open-r1/OpenR1-Math-220k",  # default split 93.7k
    "curated_thoughts": "bethgelab/CuratedThoughts",  # default split 66.1k
    # Ablation
    "curated_limr_large_lr_ablation": "GAIR/LIMR",
    "curated_limr_small_lr_ablation": "GAIR/LIMR",
    "curated_limr_large_rank_ablation": "GAIR/LIMR",
    "curated_limr_medium_rank_ablation": "GAIR/LIMR",
    "curated_limr_small_rank_ablation": "GAIR/LIMR",
    "curated_limr_tiny_rank_ablation": "GAIR/LIMR",
    "curated_open_rs3_drgrpo_ablation": "knoveleng/open-rs",
    # Reasoning gym
    "curated_rg_math": "starzmustdie/rg-math",
}
