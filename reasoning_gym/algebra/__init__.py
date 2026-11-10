from .complex_arithmetic import ComplexArithmeticConfig, ComplexArithmeticCurriculum, ComplexArithmeticDataset
from .intermediate_integration import IntermediateIntegrationConfig, IntermediateIntegrationDataset
from .irreps_tensor_product import DATASET_NAME as IRREPS_TENSOR_PRODUCT_DATASET_NAME
from .irreps_tensor_product import IrrepsTensorProductConfig, IrrepsTensorProductDataset
from .polynomial_equations import PolynomialEquationsConfig, PolynomialEquationsCurriculum, PolynomialEquationsDataset
from .polynomial_multiplication import (
    PolynomialMultiplicationConfig,
    PolynomialMultiplicationCurriculum,
    PolynomialMultiplicationDataset,
)
from .simple_equations import SimpleEquationsConfig, SimpleEquationsCurriculum, SimpleEquationsDataset
from .simple_integration import SimpleIntegrationConfig, SimpleIntegrationCurriculum, SimpleIntegrationDataset

__all__ = [
    "ComplexArithmeticConfig",
    "ComplexArithmeticDataset",
    "ComplexArithmeticCurriculum",
    "IntermediateIntegrationConfig",
    "IntermediateIntegrationDataset",
    "IRREPS_TENSOR_PRODUCT_DATASET_NAME",
    "IrrepsTensorProductConfig",
    "IrrepsTensorProductDataset",
    "PolynomialEquationsConfig",
    "PolynomialEquationsDataset",
    "PolynomialEquationsCurriculum",
    "SimpleEquationsDataset",
    "SimpleEquationsConfig",
    "SimpleEquationsCurriculum",
    "SimpleIntegrationConfig",
    "SimpleIntegrationCurriculum",
    "SimpleIntegrationDataset",
    "PolynomialMultiplicationConfig",
    "PolynomialMultiplicationDataset",
    "PolynomialMultiplicationCurriculum",
]
