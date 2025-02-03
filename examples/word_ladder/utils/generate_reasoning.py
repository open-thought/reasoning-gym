"""
generate_reasoning.py – Reads the JSONL file containing ladder examples,
creates batch requests of chain-of-thought prompts split into batches of 2,500,
calls Anthropic's Message Batches API for each batch, and writes separate batch metadata
files for later retrieval of the responses.

*** WARNING ***: Running large batches of requests via the Anthropic API (especially in generate_reasoning.py)
can incur significant costs in Anthropic credits. Please review and understand your API quota and budgeting
before running the API call. If you are testing or working with a demo dataset, adjust the batch size or dataset
size appropriately to avoid unexpected charges.

Using Anthropic's Message Batches API with caching enabled for system prompt.
In our informal testing, Sonnet was deemed best performance value.
You can swap out to another API, but this will need a rewrite to remove anthropic-specific code.
"""

import json
import os
import time
import uuid
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from tqdm import tqdm

# Updated default output directory to use the parent directory.
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

# Add default constants at the top with other constants
DEFAULT_INPUT_JSONL = "output/word_ladder_examples.jsonl"
DEFAULT_SYSTEM_PROMPT = Path(__file__).resolve().parent.parent / "system_prompt.txt"
BATCH_SIZE = 2500
COMMON_UUID = uuid.uuid4().hex[:8]

# Set up the Anthropic client (ensure the API key is set in the environment)
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def submit_reasoning_batches(
    input_path: str = DEFAULT_INPUT_JSONL,
    batch_metadata_prefix: str = "batch_metadata",
    system_prompt_path: str = DEFAULT_SYSTEM_PROMPT,
) -> None:
    """
    Reads the input JSONL file of word ladder examples, builds batch requests for any example that
    does not have reasoning, splits them into groups of BATCH_SIZE, and submits each batch using
    Anthropic's Message Batches API.

    Args:
        input_path: Path to input JSONL file
        batch_metadata_prefix: Prefix for batch metadata files
        system_prompt_path: Path to system prompt file
    """
    # Create output directory if it doesn't exist
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(exist_ok=True)

    # Read the system prompt from file (used as a preamble for every request)
    with open(system_prompt_path, "r", encoding="utf-8") as sys_file:
        system_message = [
            {
                "type": "text",
                "text": sys_file.read(),
                "cache_control": {"type": "ephemeral"},  # Enable anthropic prompt caching
            }
        ]
    batch_requests = []
    custom_ids = []  # List of custom_ids for the current batch
    batch_num = 0

    # Get the total number of lines in advance for tqdm progress bar.
    total_lines = sum(1 for _ in open(input_path))

    with open(input_path, "r", encoding="utf-8") as infile:
        for idx, line in tqdm(enumerate(infile), desc="Preparing batch requests", total=total_lines):
            data = json.loads(line)

            # Skip example if 'reasoning' already exists.
            if not data.get("reasoning"):
                # Build a custom id. Here we use the row position and the start/end words:
                metadata = data.get("metadata", {})
                start = metadata.get("start_word", "unknown")
                end = metadata.get("end_word", "unknown")
                custom_id = f"{start}_{end}_{idx}"
                custom_ids.append(custom_id)

                # Build the prompt text exactly as before.
                prompt = f"{data['question']}. The correct solution is {data['answer']}. "

                # Build the request payload using Request and MessageCreateParamsNonStreaming.
                request_payload = Request(
                    custom_id=custom_id,
                    params=MessageCreateParamsNonStreaming(
                        model="claude-3-5-sonnet-20241022",  # Or choose the appropriate model version
                        max_tokens=8192,
                        temperature=0.5,
                        system=system_message,
                        messages=[{"role": "user", "content": prompt}],
                    ),
                )
                # Instead of wrapping in SimpleNamespace, simply ensure custom_id is set.
                if isinstance(request_payload, dict):
                    request_payload["custom_id"] = custom_id
                else:
                    request_payload.custom_id = custom_id
                batch_requests.append(request_payload)

                # If we have reached our batch size limit, submit the current batch.
                if len(batch_requests) >= BATCH_SIZE:
                    _submit_single_batch(batch_requests, custom_ids, batch_num, batch_metadata_prefix, input_path)
                    batch_num += 1
                    # Reset for the next batch
                    batch_requests = []
                    custom_ids = []

    # Submit any remaining requests that didn't complete a full batch.
    if batch_requests:
        _submit_single_batch(batch_requests, custom_ids, batch_num, batch_metadata_prefix, input_path)


def _submit_single_batch(batch_requests, custom_ids, batch_num, batch_metadata_prefix, input_path):
    """
    Helper function to submit a single batch request, track the full API response,
    and write out the corresponding metadata including the list of custom_ids.
    """
    # Use the default output directory
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(exist_ok=True)

    def serialize_datetime(dt):
        """
        Convert a datetime object to ISO formatted string.
        If dt is None, returns None.
        """
        if dt is None:
            return None
        iso_str = dt.isoformat()  # e.g. "2024-08-20T18:37:24.100435+00:00"

        if dt.tzinfo is not None and dt.tzinfo.utcoffset(dt) is not None:
            iso_str = iso_str.replace("+00:00", "Z")
        return iso_str

    def extract_custom_id(req):
        # Safely extract the custom_id attribute whether req is an object or a dict.
        return req.custom_id if hasattr(req, "custom_id") else req.get("custom_id")

    max_attempts = 2
    attempt = 0
    last_exception = None
    message_batch = None
    while attempt < max_attempts:
        try:
            print(f"Submitting batch {batch_num} with {len(batch_requests)} requests... (attempt {attempt+1})")
            message_batch = client.messages.batches.create(requests=batch_requests)
            time.sleep(1)
            print(f"Batch {batch_num} submitted with ID: {message_batch.id}")
            break  # Success: exit the loop.
        except Exception as e:
            last_exception = e
            attempt += 1
            print(f"Error submitting batch {batch_num} on attempt {attempt}: {e}")
            if attempt < max_attempts:
                print("Retrying...")
                time.sleep(1)

    if message_batch is None:
        error_filename = output_dir / f"{COMMON_UUID}_failed_batches.jsonl"
        error_msg = (
            f"{str(last_exception)} after {max_attempts} attempts"
            if last_exception
            else f"Failed after {max_attempts} attempts"
        )
        failed_info = {
            "batch_number": batch_num,
            "error": error_msg,
            "batch_requests": [extract_custom_id(req) for req in batch_requests],
            "input_file": input_path,
        }
        with open(error_filename, "a", encoding="utf-8") as error_file:
            error_file.write(json.dumps(failed_info) + "\n")
        print(f"Batch {batch_num} permanently failed. Logged to {error_filename}.")
        return

    # Build a dictionary of the expected response fields.
    api_response = {
        "id": message_batch.id,
        "type": message_batch.type,
        "processing_status": message_batch.processing_status,
        "request_counts": vars(message_batch.request_counts),
        "ended_at": serialize_datetime(message_batch.ended_at),
        "created_at": serialize_datetime(message_batch.created_at),
        "expires_at": serialize_datetime(message_batch.expires_at),
        "cancel_initiated_at": serialize_datetime(message_batch.cancel_initiated_at),
        "results_url": message_batch.results_url,
    }

    batch_metadata = {
        "batch_id": message_batch.id,
        "api_response": api_response,
        "custom_ids": custom_ids,
        "input_file": os.path.basename(input_path),
    }
    metadata_filename = output_dir / f"{COMMON_UUID}_{batch_metadata_prefix}.jsonl"
    with open(metadata_filename, "a", encoding="utf-8") as meta_file:
        meta_file.write(json.dumps(batch_metadata) + "\n")

    print(f"Batch metadata for batch {batch_num} appended to {metadata_filename}.")


if __name__ == "__main__":
    # When running this module directly, submit the reasoning batches.
    submit_reasoning_batches()
