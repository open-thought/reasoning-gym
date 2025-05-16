import os

from huggingface_hub import snapshot_download

if __name__ == "__main__":
    CKPT_DIR = os.environ["CKPT_DIR"]

    print("Downloading model ...")
    snapshot_download(repo_id="Qwen/Qwen2.5-3B-Instruct", local_dir=f"{CKPT_DIR}/models/Qwen2.5-3B-Instruct/base")
