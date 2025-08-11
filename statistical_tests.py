# Backwards-compatibility shim: re-export the factory and helpers from src
from src.statistical_tests import *

# Explicit exports for clarity
__all__ = [
    'StatisticalTest', 'TwoSampleTTest', 'ChiSquareTest', 'OneWayANOVA', 'CorrelationTest',
    'ANCOVATest', 'FishersExactTest', 'LogisticRegressionTest', 'RepeatedMeasuresANOVA',
    'StatisticalTestFactory', 'get_factory', 'validate_statistical_inputs',
    'calculate_p_value_from_N_d', 'calculate_power_from_N_d'
]
