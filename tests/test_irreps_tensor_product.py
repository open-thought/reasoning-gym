import sys
import types
from dataclasses import FrozenInstanceError

import pytest


def _install_optional_dependency_stubs() -> None:
    if "cellpylib" not in sys.modules:
        sys.modules["cellpylib"] = types.ModuleType("cellpylib")

    if "zss" not in sys.modules:
        sys.modules["zss"] = types.ModuleType("zss")

    if "bfi" not in sys.modules:
        bfi = types.ModuleType("bfi")
        bfi.interpret = lambda *args, **kwargs: ""
        sys.modules["bfi"] = bfi

    if "arckit" not in sys.modules:
        arckit = types.ModuleType("arckit")
        arckit.load_data = lambda: ([], [])
        sys.modules["arckit"] = arckit

    if "pyfiglet" not in sys.modules:
        pyfiglet = types.ModuleType("pyfiglet")

        class _FigletFont:
            @staticmethod
            def getFonts():
                return ["standard"]

        pyfiglet.FigletFont = _FigletFont
        pyfiglet.figlet_format = lambda text, font=None: text
        sys.modules["pyfiglet"] = pyfiglet

    if "magiccube" not in sys.modules:
        magiccube = types.ModuleType("magiccube")
        cube = types.ModuleType("magiccube.cube")
        solver = types.ModuleType("magiccube.solver")
        basic = types.ModuleType("magiccube.solver.basic")
        basic_solver = types.ModuleType("magiccube.solver.basic.basic_solver")

        cube.Cube = type("Cube", (), {})
        cube.CubeMove = type("CubeMove", (), {})
        cube.CubeMoveType = type("CubeMoveType", (), {})
        basic_solver.BasicSolver = type("BasicSolver", (), {})

        sys.modules["magiccube"] = magiccube
        sys.modules["magiccube.cube"] = cube
        sys.modules["magiccube.solver"] = solver
        sys.modules["magiccube.solver.basic"] = basic
        sys.modules["magiccube.solver.basic.basic_solver"] = basic_solver


_install_optional_dependency_stubs()

import reasoning_gym.algebra.irreps_tensor_product as irreps_module
from reasoning_gym import create_dataset, get_score_answer_fn
from reasoning_gym.algebra.irreps_tensor_product import (
    DATASET_NAME,
    Irrep,
    IrrepsTensorProductConfig,
    IrrepsTensorProductDataset,
    decompose_tensor_product,
    format_decomposition,
    format_irrep,
    irrep_dimension,
    multiply_parity,
    parse_irrep,
)


def test_module_all_exports_public_api():
    assert set(irreps_module.__all__) == {
        "DATASET_NAME",
        "Irrep",
        "Parity",
        "IrrepsTensorProductConfig",
        "IrrepsTensorProductDataset",
        "parse_irrep",
        "format_irrep",
        "multiply_parity",
        "decompose_tensor_product",
        "format_decomposition",
        "irrep_dimension",
    }


def test_irrep_is_immutable():
    irrep = Irrep(2, "e")

    with pytest.raises(FrozenInstanceError):
        irrep.l = 3


def test_irrep_structural_equality():
    assert Irrep(2, "e") == Irrep(2, "e")
    assert Irrep(2, "e") != Irrep(3, "e")
    assert Irrep(2, "e") != Irrep(2, "o")


def test_irrep_is_hashable():
    seen = {Irrep(2, "e")}
    assert Irrep(2, "e") in seen


def test_irrep_requires_exact_int_degree():
    with pytest.raises(TypeError):
        Irrep(True, "e")

    with pytest.raises(TypeError):
        Irrep(2.0, "e")


def test_irrep_validates_degree_and_parity_values():
    with pytest.raises(ValueError):
        Irrep(-1, "e")

    with pytest.raises(ValueError, match="parity"):
        Irrep(2, "E")

    with pytest.raises(ValueError, match="parity"):
        Irrep(2, "x")


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("0e", "0e"),
        ("1o", "1o"),
        ("2E", "2e"),
        ("3O", "3o"),
        (" 4e ", "4e"),
    ],
)
def test_parse_then_format_round_trip_to_canonical(token, expected):
    irrep = parse_irrep(token)
    assert format_irrep(irrep) == expected


def test_format_parse_round_trip_on_irrep():
    irrep = Irrep(2, "e")
    assert parse_irrep(format_irrep(irrep)) == irrep


def test_parse_irrep_rejects_non_string_input():
    with pytest.raises(TypeError):
        parse_irrep(2)


@pytest.mark.parametrize(
    "token",
    [
        "",
        " ",
        "2",
        "2 e",
        "2even",
        "2x",
        "-1e",
        "01e",
        "00o",
        "1.5e",
        "1e extra",
        "1e + 2e",
    ],
)
def test_parse_irrep_rejects_invalid_inputs(token):
    with pytest.raises(ValueError):
        parse_irrep(token)


@pytest.mark.parametrize(
    "token",
    [
        "2ο",
        "２e",
    ],
)
def test_parse_irrep_rejects_unicode_lookalikes(token):
    with pytest.raises(ValueError):
        parse_irrep(token)


def test_format_irrep_requires_irrep():
    with pytest.raises(TypeError):
        format_irrep("2e")


def test_multiply_parity_table():
    assert multiply_parity("e", "e") == "e"
    assert multiply_parity("e", "o") == "o"
    assert multiply_parity("o", "e") == "o"
    assert multiply_parity("o", "o") == "e"


def test_multiply_parity_rejects_invalid_values():
    with pytest.raises(ValueError, match="parity"):
        multiply_parity("E", "o")

    with pytest.raises(TypeError):
        multiply_parity("e", None)


def test_irrep_dimension():
    assert irrep_dimension(Irrep(0, "e")) == 1
    assert irrep_dimension(Irrep(4, "o")) == 9


def test_irrep_dimension_requires_irrep():
    with pytest.raises(TypeError):
        irrep_dimension("2e")


def test_decompose_tensor_product_examples():
    assert decompose_tensor_product(Irrep(2, "e"), Irrep(3, "o")) == (
        Irrep(1, "o"),
        Irrep(2, "o"),
        Irrep(3, "o"),
        Irrep(4, "o"),
        Irrep(5, "o"),
    )
    assert decompose_tensor_product(Irrep(1, "o"), Irrep(1, "o")) == (
        Irrep(0, "e"),
        Irrep(1, "e"),
        Irrep(2, "e"),
    )
    assert decompose_tensor_product(Irrep(0, "e"), Irrep(4, "o")) == (Irrep(4, "o"),)


def test_decompose_tensor_product_is_commutative():
    expected = (
        Irrep(1, "o"),
        Irrep(2, "o"),
        Irrep(3, "o"),
        Irrep(4, "o"),
        Irrep(5, "o"),
    )

    assert decompose_tensor_product(Irrep(2, "e"), Irrep(3, "o")) == expected
    assert decompose_tensor_product(Irrep(3, "o"), Irrep(2, "e")) == expected


def test_decompose_tensor_product_safe_large_one_term_case():
    assert decompose_tensor_product(Irrep(0, "e"), Irrep(10_000_000, "o")) == (Irrep(10_000_000, "o"),)


def test_decompose_tensor_product_requires_irrep_operands():
    with pytest.raises(TypeError):
        decompose_tensor_product("2e", Irrep(1, "o"))


def test_decompose_tensor_product_preserves_dimension():
    left = Irrep(2, "e")
    right = Irrep(3, "o")
    terms = decompose_tensor_product(left, right)

    assert sum(irrep_dimension(term) for term in terms) == irrep_dimension(left) * irrep_dimension(right)


def test_format_decomposition():
    terms = (
        Irrep(1, "o"),
        Irrep(2, "o"),
        Irrep(3, "o"),
        Irrep(4, "o"),
        Irrep(5, "o"),
    )
    assert format_decomposition(terms) == "1o + 2o + 3o + 4o + 5o"


def test_format_decomposition_requires_tuple():
    with pytest.raises(TypeError):
        format_decomposition([Irrep(1, "o")])


def test_format_decomposition_rejects_empty_or_non_irrep_terms():
    with pytest.raises(ValueError):
        format_decomposition(())

    with pytest.raises(TypeError):
        format_decomposition((Irrep(1, "o"), "2o"))


def test_format_decomposition_rejects_non_canonical_term_sequences():
    with pytest.raises(ValueError):
        format_decomposition((Irrep(1, "o"), Irrep(3, "o")))

    with pytest.raises(ValueError):
        format_decomposition((Irrep(2, "o"), Irrep(1, "o")))

    with pytest.raises(ValueError):
        format_decomposition((Irrep(1, "o"), Irrep(1, "o")))

    with pytest.raises(ValueError):
        format_decomposition((Irrep(1, "o"), Irrep(2, "e")))


def test_config_validation():
    with pytest.raises(AssertionError):
        IrrepsTensorProductConfig(min_l=-1).validate()

    with pytest.raises(AssertionError):
        IrrepsTensorProductConfig(min_l=3, max_l=2).validate()

    with pytest.raises(AssertionError):
        IrrepsTensorProductConfig(include_even=False, include_odd=False).validate()

    with pytest.raises(AssertionError):
        IrrepsTensorProductConfig(size=0).validate()


def test_dataset_registration_and_creation():
    dataset = create_dataset(DATASET_NAME, min_l=1, max_l=3, include_even=True, include_odd=False, seed=7, size=5)

    assert isinstance(dataset, IrrepsTensorProductDataset)
    assert len(dataset) == 5


def test_dataset_item_shape_and_metadata():
    dataset = IrrepsTensorProductDataset(IrrepsTensorProductConfig(min_l=1, max_l=3, seed=11, size=2))
    entry = dataset[0]
    metadata = entry["metadata"]

    assert "question" in entry
    assert "answer" in entry
    assert "metadata" in entry
    assert entry["answer"] == format_decomposition(tuple(parse_irrep(term) for term in metadata["decomposition_terms"]))
    assert metadata["source_dataset"] == DATASET_NAME
    assert metadata["source_index"] == 0
    assert metadata["left_irrep"] == f"{metadata['l1']}{metadata['parity_1']}"
    assert metadata["right_irrep"] == f"{metadata['l2']}{metadata['parity_2']}"
    assert metadata["parity_case"] == f"{metadata['parity_1']}{metadata['parity_2']}"
    assert metadata["term_count"] == len(metadata["decomposition_terms"])
    assert entry["question"].count("e*e=e") == 1
    assert entry["question"].count("o*o=e") == 1


def test_dataset_is_deterministic():
    config = IrrepsTensorProductConfig(min_l=0, max_l=4, seed=19, size=5)
    dataset_1 = IrrepsTensorProductDataset(config)
    dataset_2 = IrrepsTensorProductDataset(config)

    for index in range(len(dataset_1)):
        assert dataset_1[index] == dataset_2[index]


def test_dataset_iteration_is_repeatable():
    dataset = IrrepsTensorProductDataset(IrrepsTensorProductConfig(seed=23, size=4))
    first_pass = list(dataset)
    second_pass = list(dataset)

    assert len(first_pass) == 4
    assert first_pass == second_pass


def test_dataset_scores_oracle_answers():
    dataset = IrrepsTensorProductDataset(IrrepsTensorProductConfig(seed=29, size=5))

    for entry in dataset:
        assert dataset.score_answer(entry["answer"], entry) == 1.0


def test_score_answer_accepts_harmless_format_variants():
    dataset = IrrepsTensorProductDataset(IrrepsTensorProductConfig())
    entry = {"metadata": {"decomposition_terms": ["1o", "2o", "3o", "4o", "5o"]}}

    assert dataset.score_answer("1o + 2o + 3o + 4o + 5o", entry) == 1.0
    assert dataset.score_answer("1o+2o+3o+4o+5o", entry) == 1.0
    assert dataset.score_answer(" 1O ⊕ 2O ⊕ 3O ⊕ 4O ⊕ 5O ", entry) == 1.0


def test_score_answer_rejects_wrong_or_malformed_answers():
    dataset = IrrepsTensorProductDataset(IrrepsTensorProductConfig())
    entry = {"metadata": {"decomposition_terms": ["1o", "2o", "3o", "4o", "5o"]}}

    assert dataset.score_answer(None, entry) == 0.0
    assert dataset.score_answer("", entry) == 0.0
    assert dataset.score_answer("5o + 4o + 3o + 2o + 1o", entry) == 0.0
    assert dataset.score_answer("1o + 2o + 3o + 4o", entry) == 0.0
    assert dataset.score_answer("1o + 2o + 3o + 4o + 5o + 6o", entry) == 0.0
    assert dataset.score_answer("1o + 2o + 2o + 3o + 4o + 5o", entry) == 0.0
    assert dataset.score_answer("1e + 2e + 3e + 4e + 5e", entry) == 0.0
    assert dataset.score_answer("The answer is 1o + 2o + 3o + 4o + 5o", entry) == 0.0


def test_get_score_answer_fn_uses_dataset_scorer():
    scorer = get_score_answer_fn(DATASET_NAME)
    entry = {"metadata": {"decomposition_terms": ["0e", "1e", "2e"]}}

    assert scorer("0e + 1e + 2e", entry) == 1.0
    assert scorer("2e + 1e + 0e", entry) == 0.0
