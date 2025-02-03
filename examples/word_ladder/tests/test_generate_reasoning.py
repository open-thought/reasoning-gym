import datetime
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.word_ladder import generate_reasoning

# We alias the functions and globals for easier usage in our tests.
submit_reasoning_batches = generate_reasoning.submit_reasoning_batches
_submit_single_batch = generate_reasoning._submit_single_batch
DEFAULT_INPUT_JSONL = generate_reasoning.DEFAULT_INPUT_JSONL
COMMON_UUID = generate_reasoning.COMMON_UUID
BATCH_SIZE = generate_reasoning.BATCH_SIZE
client = generate_reasoning.client


# Define a mock batch response class mimicking Anthropic's API response.
class MockBatchResponse:
    def __init__(self, batch_id="msgbatch_mock", processing_status="in_progress", fail=False):
        self.id = batch_id
        self.type = "message_batch"
        self.processing_status = processing_status
        # Make request_counts a SimpleNamespace object with the required attributes
        self.request_counts = SimpleNamespace(processing=0, succeeded=0, errored=0, canceled=0, expired=0)
        self.ended_at = None
        # Use datetime objects so that isoformat() is available
        self.created_at = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
        self.expires_at = self.created_at + datetime.timedelta(seconds=86400)
        self.cancel_initiated_at = None
        self.results_url = None


# Helper: Create a temporary system prompt file.
@pytest.fixture
def system_prompt_file(tmp_path, monkeypatch):
    prompt_text = "This is a system prompt."
    sys_file = tmp_path / "system_prompt.txt"
    sys_file.write_text(prompt_text, encoding="utf-8")

    # Monkeypatch the system prompt path
    monkeypatch.setattr(generate_reasoning, "DEFAULT_SYSTEM_PROMPT", str(sys_file))
    return sys_file


# Helper: Create necessary directories using a temporary location.
@pytest.fixture
def setup_directories(tmp_path, monkeypatch):
    # Create output directory in temporary path
    output_dir = tmp_path / "output"
    output_dir.mkdir(exist_ok=True)

    # Monkeypatch the DEFAULT_OUTPUT_DIR to be a Path (temporary directory).
    monkeypatch.setattr(generate_reasoning, "DEFAULT_OUTPUT_DIR", output_dir)

    # Ensure we're working in the temporary directory
    monkeypatch.chdir(tmp_path)
    return output_dir


# Helper: Create a temporary input JSONL file with given entries.
@pytest.fixture
def input_jsonl_file(tmp_path, setup_directories, monkeypatch):
    # Create input file in temporary directory
    file_path = setup_directories / "word_ladder_examples.jsonl"
    entries = [
        {
            "question": "Transform 'A' to 'B'",
            "answer": "A,X,B",
            "reasoning": None,
            "metadata": {"start_word": "A", "end_word": "B", "word_length": 1, "chain_length": 3},
        },
        {
            "question": "Transform 'C' to 'D'",
            "answer": "C,Y,D",
            "reasoning": "Some reasoning",
            "metadata": {"start_word": "C", "end_word": "D", "word_length": 1, "chain_length": 3},
        },
        {
            "question": "Transform 'E' to 'F'",
            "answer": "E,Z,F",
            "reasoning": None,
            "metadata": {"start_word": "E", "end_word": "F", "word_length": 1, "chain_length": 3},
        },
    ]
    with file_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    # Monkeypatch DEFAULT_INPUT_JSONL to point to our temporary test file.
    monkeypatch.setattr(generate_reasoning, "DEFAULT_INPUT_JSONL", str(file_path))
    return file_path


# Test that submit_reasoning_batches builds a batch skipping entries with existing reasoning.
def test_submit_batches_success(system_prompt_file, input_jsonl_file, setup_directories, monkeypatch):
    def fake_create(requests):
        for req in requests:
            # Handle the case where req is a dictionary
            if isinstance(req, dict):
                params = req.get("params", {})
                custom_id = req.get("custom_id")
                # Check if params itself is a dictionary
                if isinstance(params, dict):
                    model = params.get("model")
                    temperature = params.get("temperature")
                else:
                    model = params.model
                    temperature = params.temperature
            else:
                # Else, req is an object with attributes.
                params = req.params
                custom_id = req.custom_id
                if isinstance(params, dict):
                    model = params.get("model")
                    temperature = params.get("temperature")
                else:
                    model = params.model
                    temperature = params.temperature
            assert model == "claude-3-5-sonnet-20241022", "Incorrect model version"
            assert temperature == 0.5, "Incorrect temperature"
            assert "C_D_" not in custom_id
        return MockBatchResponse(batch_id="msgbatch_test_success")

    monkeypatch.setattr(client.messages.batches, "create", fake_create)

    batch_metadata_prefix = "test_metadata"
    submit_reasoning_batches(input_path=str(input_jsonl_file), batch_metadata_prefix=batch_metadata_prefix)

    metadata_filename = f"{COMMON_UUID}_{batch_metadata_prefix}.jsonl"
    meta_file_path = setup_directories / metadata_filename
    assert meta_file_path.exists(), "Metadata file was not created as expected."

    with meta_file_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
        # Expecting only those entries that did not already have a reasoning.
        # (From our test input, 2 out of 3 entries qualify.)
        assert len(lines) > 0
        for line in lines:
            metadata = json.loads(line)
            api_response = metadata["api_response"]
            assert api_response["id"] == "msgbatch_test_success"
            assert api_response["processing_status"] == "in_progress"
            custom_ids = metadata["custom_ids"]
            assert len(custom_ids) == 2


# Test that _submit_single_batch retries once and eventually succeeds.
def test_retry_logic(system_prompt_file, setup_directories, monkeypatch):
    call_count = {"count": 0}

    def fake_create_retry(requests):
        if call_count["count"] == 0:
            call_count["count"] += 1
            raise Exception("Temporary failure")
        return MockBatchResponse(batch_id="msgbatch_retry_success")

    monkeypatch.setattr(client.messages.batches, "create", fake_create_retry)

    dummy_request = type("DummyRequest", (), {"custom_id": "dummy_1"})()
    batch_requests = [dummy_request]
    custom_ids = ["dummy_1"]

    _submit_single_batch(batch_requests, custom_ids, 0, "test_retry", "dummy_input.jsonl")

    metadata_filename = f"{COMMON_UUID}_test_retry.jsonl"
    meta_file_path = setup_directories / metadata_filename
    assert meta_file_path.exists(), "Retry metadata file was not created."

    with meta_file_path.open("r", encoding="utf-8") as f:
        metadata = json.loads(f.read())
        assert metadata["api_response"]["id"] == "msgbatch_retry_success"

    assert call_count["count"] == 1


# Test that when all attempts to submit a batch fail, the error is logged to the failed file.
def test_failed_batch(system_prompt_file, setup_directories, monkeypatch):
    def fake_create_fail(requests):
        raise Exception("Permanent failure")

    monkeypatch.setattr(client.messages.batches, "create", fake_create_fail)

    dummy_request = type("DummyRequest", (), {"custom_id": "dummy_fail"})()
    batch_requests = [dummy_request]
    custom_ids = ["dummy_fail"]

    _submit_single_batch(batch_requests, custom_ids, 0, "test_failed", "dummy_input.jsonl")

    error_filename = f"{COMMON_UUID}_failed_batches.jsonl"
    error_file_path = setup_directories / error_filename
    assert error_file_path.exists(), "Failed batch log file was not created."

    with error_file_path.open("r", encoding="utf-8") as f:
        error_entry = json.loads(f.readline())
        assert error_entry["batch_number"] == 0
        assert "Permanent failure" in error_entry["error"]
        assert error_entry["batch_requests"] == ["dummy_fail"]


# Test batching behavior when multiple batches are needed.
def test_multiple_batches(system_prompt_file, setup_directories, monkeypatch):
    test_batch_size = 2
    monkeypatch.setattr(generate_reasoning, "BATCH_SIZE", test_batch_size)

    # Create input file
    input_file = setup_directories / "word_ladder_examples.jsonl"
    entries = [
        {
            "question": f"Transform word ladder {idx}",
            "answer": f"start,mid,end_{idx}",
            "reasoning": None,
            "metadata": {"start_word": f"start_{idx}", "end_word": f"end_{idx}"},
        }
        for idx in range(5)
    ]

    with input_file.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    # Monkeypatch DEFAULT_INPUT_JSONL.
    monkeypatch.setattr(generate_reasoning, "DEFAULT_INPUT_JSONL", str(input_file))

    batch_ids = []

    def fake_create(requests):
        new_id = f"msgbatch_batch_{len(batch_ids)}"
        batch_ids.append(new_id)
        return MockBatchResponse(batch_id=new_id)

    monkeypatch.setattr(client.messages.batches, "create", fake_create)

    batch_metadata_prefix = "test_multi"
    submit_reasoning_batches(input_path=str(input_file), batch_metadata_prefix=batch_metadata_prefix)

    metadata_filename = f"{COMMON_UUID}_{batch_metadata_prefix}.jsonl"
    meta_file_path = setup_directories / metadata_filename
    assert meta_file_path.exists(), "Multiple batch metadata file was not created."

    with meta_file_path.open("r", encoding="utf-8") as f:
        metadata_lines = f.readlines()
    # With 5 qualifying entries and a batch size of 2 we expect 3 batches.
    assert len(metadata_lines) == 3

    seen_custom_ids = []
    for line in metadata_lines:
        metadata = json.loads(line)
        seen_custom_ids.extend(metadata["custom_ids"])
        assert metadata["api_response"]["id"] in batch_ids
    assert len(seen_custom_ids) == 5
