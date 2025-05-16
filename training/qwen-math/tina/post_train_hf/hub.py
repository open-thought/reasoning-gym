from concurrent.futures import Future
from huggingface_hub import create_branch, create_repo, list_repo_commits, upload_folder
import logging

logger = logging.getLogger(__name__)


def push_to_hub_revision(training_args, extra_ignore_patterns=[]) -> Future:
    """Pushes the model to branch on a Hub repo."""

    # Create a repo if it doesn't exist yet
    repo_url = create_repo(repo_id=training_args.hub_model_id, private=True, exist_ok=True)
    # Get initial commit to branch from
    initial_commit = list_repo_commits(training_args.hub_model_id)[-1]
    # Now create the branch we'll be pushing to
    create_branch(
        repo_id=training_args.hub_model_id,
        branch=training_args.hub_model_revision,
        # checkpoint=training_args.checkpoint,
        revision=initial_commit.commit_id,
        exist_ok=True,
    )
    logger.info(f"Created target repo at {repo_url}")
    logger.info(f"Pushing to the Hub revision {training_args.hub_model_revision} with checkpoint {training_args.checkpoint}")
    ignore_patterns = ["checkpoint-*", "*.pth"]
    ignore_patterns.extend(extra_ignore_patterns)
    future = upload_folder(
        repo_id=training_args.hub_model_id,
        folder_path=training_args.output_dir,
        revision=training_args.hub_model_revision,
        # commit_message=f"Add {training_args.hub_model_revision} checkpoint {training_args.dataset_name}",
        commit_message=f"Add {training_args.checkpoint} checkpoint post-trained on {training_args.dataset_name}",
        ignore_patterns=ignore_patterns,
        run_as_future=True,
    )

    logger.info(f"Pushed to {repo_url} revision {training_args.hub_model_revision} with checkpoint {training_args.checkpoint} successfully!")

    return future
