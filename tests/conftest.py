"""Pytest configuration and fixtures."""

import pytest
import sys
from pathlib import Path
from fastapi.testclient import TestClient
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.api.main import app
from backend.statistical.frequentist import StatisticalTestFactory
from backend.statistical.visualizations import TrialVisualizer


@pytest.fixture
def client():
    """Create FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def test_factory():
    """Create statistical test factory instance."""
    return StatisticalTestFactory()


@pytest.fixture
def visualizer():
    """Create trial visualizer instance."""
    return TrialVisualizer()


@pytest.fixture
def sample_trial_params():
    """Sample parameters for two-sample t-test."""
    return {
        "n_total": 100,
        "effect_size": 0.5,
        "alpha": 0.05,
        "power": 0.8
    }


@pytest.fixture
def sample_chi_square_params():
    """Sample parameters for chi-square test."""
    return {
        "n_total": 200,
        "n_groups": 2,
        "n_categories": 2,
        "effect_size": 0.3,
        "alpha": 0.05
    }


@pytest.fixture
def sample_anova_params():
    """Sample parameters for one-way ANOVA."""
    return {
        "n_total": 150,
        "n_groups": 3,
        "effect_size": 0.25,
        "alpha": 0.05
    }


@pytest.fixture
def sample_correlation_params():
    """Sample parameters for correlation test."""
    return {
        "n": 100,
        "correlation": 0.3,
        "method": "pearson",
        "alpha": 0.05
    }


@pytest.fixture
def sample_study_description():
    """Sample study description for AI analysis."""
    return (
        "We want to test if a new diabetes drug reduces HbA1c levels "
        "compared to placebo in a randomized controlled trial with 200 patients."
    )


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    return {
        "suggested_study_type": "two_sample_t_test",
        "rationale": "This is a two-group comparison of a continuous outcome",
        "parameters": {
            "total_n": 200,
            "cohens_d": 0.5
        },
        "alternative_tests": ["welch_t_test", "mann_whitney"],
        "confidence_level": 0.85
    }


@pytest.fixture
def sample_visualization_data():
    """Sample data for visualization tests."""
    return {
        "n_range": list(range(10, 101, 10)),
        "effect_sizes": [0.2, 0.5, 0.8],
        "alpha": 0.05,
        "power_values": np.random.uniform(0.5, 0.95, 10).tolist()
    }


@pytest.fixture
def sample_scenarios():
    """Sample scenarios for comparison."""
    return [
        {
            "name": "Conservative",
            "p_value": 0.08,
            "power": 0.70,
            "sample_size": 80
        },
        {
            "name": "Moderate",
            "p_value": 0.04,
            "power": 0.80,
            "sample_size": 100
        },
        {
            "name": "Optimistic",
            "p_value": 0.01,
            "power": 0.90,
            "sample_size": 150
        }
    ]


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed for reproducibility."""
    np.random.seed(42)
    yield
    np.random.seed(None)