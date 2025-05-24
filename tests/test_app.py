import math
import pytest
from app import calculate_p_value_from_N_d


def test_p_value_below_0_05():
    p_value, msg = calculate_p_value_from_N_d(100, 0.5)
    assert p_value is not None
    assert p_value < 0.05


def test_p_value_zero_effect():
    p_value, msg = calculate_p_value_from_N_d(10, 0)
    assert p_value == 1.0
