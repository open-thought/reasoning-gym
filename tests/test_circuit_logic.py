import pytest

from reasoning_gym.logic import CircuitLogicConfig, CircuitLogicDataset


def test_circuit_logic_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = CircuitLogicConfig(min_inputs=3, max_inputs=2)
        config.validate()

    with pytest.raises(AssertionError):
        config = CircuitLogicConfig(num_terms=0)
        config.validate()

    with pytest.raises(AssertionError):
        config = CircuitLogicConfig(neg_prob=-0.1)
        config.validate()

    with pytest.raises(AssertionError):
        config = CircuitLogicConfig(neg_prob=1.1)
        config.validate()


def test_circuit_logic_deterministic():
    """Test that dataset generates same items with same seed"""
    config = CircuitLogicConfig(seed=42, size=10)
    dataset1 = CircuitLogicDataset(config)
    dataset2 = CircuitLogicDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_circuit_logic_items():
    """Test basic properties of generated items"""
    config = CircuitLogicConfig(num_terms=3, min_inputs=2, max_inputs=3, neg_prob=0.3, size=50, seed=42)
    dataset = CircuitLogicDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Verify metadata contents
        metadata = item["metadata"]
        assert "expression" in metadata
        assert "assignments" in metadata
        assert "final_gate" in metadata
        assert "inputs" in metadata

        # Verify answer is binary
        assert item["answer"] in ("0", "1")

        # Verify assignments are binary
        for input_name, value in metadata["assignments"].items():
            assert value in (0, 1)

        # Verify final gate is valid
        assert metadata["final_gate"] in ("OR", "NOR", "XOR", "AND")

        # Verify inputs list matches assignments
        assert set(metadata["inputs"]) == set(metadata["assignments"].keys())


def test_circuit_logic_expression_validity():
    """Test that generated expressions follow logical circuit rules"""
    config = CircuitLogicConfig(
        num_terms=2, min_inputs=2, max_inputs=2, neg_prob=0.0, size=20, seed=42  # Disable negation for simpler testing
    )
    dataset = CircuitLogicDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        metadata = item["metadata"]

        # Expression should contain valid operators
        expr = metadata["expression"]
        assert any(op in expr for op in ("&", "↑", "⊕", "+", "↓"))

        # Input names should be valid Excel-style names
        for input_name in metadata["inputs"]:
            assert input_name.isalpha()
            assert input_name.isupper()


def test_circuit_logic_answer_verification():
    """Test that answers match logical evaluation of circuits"""
    config = CircuitLogicConfig(num_terms=2, min_inputs=2, max_inputs=2, size=20, seed=42)
    dataset = CircuitLogicDataset(config)

    def evaluate_term(term: str, assignments: dict) -> int:
        """Evaluate a single term with given assignments"""
        if "↑" in term:  # NAND
            parts = term.split("↑")
            values = []
            for p in parts:
                if p.endswith("'"):
                    values.append(1 - assignments[p[:-1]])
                else:
                    values.append(assignments[p])
            return 0 if all(v == 1 for v in values) else 1
        elif "&" in term:  # AND
            parts = term.split("&")
            values = []
            for p in parts:
                if p.endswith("'"):
                    values.append(1 - assignments[p[:-1]])
                else:
                    values.append(assignments[p])
            return 1 if all(v == 1 for v in values) else 0
        elif "⊕" in term:  # XOR
            parts = term.split("⊕")
            values = []
            for p in parts:
                if p.endswith("'"):
                    values.append(1 - assignments[p[:-1]])
                else:
                    values.append(assignments[p])
            return sum(values) % 2
        else:
            raise ValueError(f"Unknown operator in term: {term}")

    def evaluate_final_gate(gate_type: str, term_values: list) -> int:
        """Evaluate the final gate with given term values"""
        if gate_type == "AND":
            return 1 if all(v == 1 for v in term_values) else 0
        elif gate_type == "OR":
            return 1 if any(v == 1 for v in term_values) else 0
        elif gate_type == "XOR":
            return sum(term_values) % 2
        elif gate_type == "NOR":
            return 0 if any(v == 1 for v in term_values) else 1
        else:
            raise ValueError(f"Unknown gate type: {gate_type}")

    for i in range(len(dataset)):
        item = dataset[i]
        metadata = item["metadata"]
        assignments = metadata["assignments"]
        final_gate = metadata["final_gate"]
        term_strings = metadata["term_strings"]

        # First evaluate each term
        term_values = [evaluate_term(term, assignments) for term in term_strings]

        # Then combine terms with final gate
        expected = evaluate_final_gate(final_gate, term_values)

        # Compare with actual result
        result = int(item["answer"])
        assert (
            result == expected
        ), f"Item {i}: Expected {expected} but got {result} for terms {term_strings} with assignments {assignments} and final gate {final_gate}"


def test_circuit_logic_ascii_diagram():
    """Test properties of the ASCII circuit diagram"""
    config = CircuitLogicConfig(num_terms=2, min_inputs=2, max_inputs=2, size=10, seed=42)
    dataset = CircuitLogicDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]

        # Split question to get diagram
        parts = item["question"].split("\n")
        diagram_start = parts.index("Below is a randomly generated logic circuit.") + 2
        diagram_end = parts.index("", diagram_start)
        diagram = parts[diagram_start:diagram_end]

        # Basic diagram validation
        assert len(diagram) > 0
        assert all(len(row) > 0 for row in diagram)

        # Check for required circuit elements
        diagram_str = "\n".join(diagram)
        assert "OUT" in diagram_str
        assert any(gate in diagram_str for gate in ("&", "↑", "⊕"))

        # Verify input labels
        for input_name in item["metadata"]["inputs"]:
            assert f"{input_name}:" in diagram_str


def test_circuit_logic_scoring():
    """Test the answer scoring mechanism"""
    config = CircuitLogicConfig(size=5, seed=42)
    dataset = CircuitLogicDataset(config)

    item = dataset[0]

    # Correct answer should score 1.0
    assert dataset.score_answer(item["answer"], item) == 1.0

    # Wrong answer should score lower
    wrong_answer = "1" if item["answer"] == "0" else "0"
    assert dataset.score_answer(wrong_answer, item) < 1.0

    # None or empty answer should score 0.0
    assert dataset.score_answer(None, item) == 0.0
    assert dataset.score_answer("", item) == 0.0  # Empty string should score 0.0 like None


def test_circuit_logic_iteration():
    """Test that iteration works correctly"""
    config = CircuitLogicConfig(size=5, seed=42)
    dataset = CircuitLogicDataset(config)

    # Test manual iteration
    items = []
    for item in dataset:
        items.append(item)
    assert len(items) == config.size

    # Test list conversion
    items = list(dataset)
    assert len(items) == config.size

    # Test multiple iterations yield same items
    first_items = list(dataset)
    second_items = list(dataset)
    assert first_items == second_items
