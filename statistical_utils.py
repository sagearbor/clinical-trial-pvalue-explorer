# Backwards-compatibility shim: re-export implementations from src package
from src.statistical_utils import (
    validate_statistical_inputs,
    calculate_p_value_from_N_d,
    calculate_power_from_N_d,
    perform_statistical_calculations,
)

__all__ = [
    "validate_statistical_inputs",
    "calculate_p_value_from_N_d",
    "calculate_power_from_N_d",
    "perform_statistical_calculations",
]
