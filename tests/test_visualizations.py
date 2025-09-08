"""Tests for visualization components."""

import pytest
import json
import numpy as np
from backend.statistical.visualizations import TrialVisualizer


class TestTrialVisualizer:
    """Test suite for TrialVisualizer class."""
    
    def test_visualizer_initialization(self, visualizer):
        """Test visualizer initialization."""
        assert visualizer.theme == "plotly_white"
        assert len(visualizer.default_colors) == 5
    
    def test_power_curve_generation(self, visualizer):
        """Test power curve generation."""
        n_range = list(range(20, 101, 20))
        effect_sizes = [0.2, 0.5, 0.8]
        
        plot_json = visualizer.create_power_curve(
            n_range=n_range,
            effect_sizes=effect_sizes,
            alpha=0.05,
            test_type="two_sample_t_test"
        )
        
        # Verify JSON structure
        plot_data = json.loads(plot_json)
        assert 'data' in plot_data
        assert 'layout' in plot_data
        assert len(plot_data['data']) == len(effect_sizes)
        
        # Verify trace properties
        for i, trace in enumerate(plot_data['data']):
            assert 'x' in trace
            assert 'y' in trace
            assert len(trace['x']) == len(n_range)
            assert all(0 <= y <= 1 for y in trace['y'] if y is not None)
    
    def test_p_value_distribution(self, visualizer):
        """Test p-value distribution visualization."""
        plot_json = visualizer.create_p_value_distribution(
            n=100,
            effect_size=0.5,
            alpha=0.05,
            simulations=1000,  # Reduced for testing
            test_type="two_sample_t_test"
        )
        
        plot_data = json.loads(plot_json)
        assert 'data' in plot_data
        assert 'layout' in plot_data
        
        # Should have histograms for null and alternative
        assert len(plot_data['data']) >= 2
    
    def test_effect_size_sensitivity(self, visualizer):
        """Test effect size sensitivity analysis."""
        plot_json = visualizer.create_effect_size_sensitivity(
            base_effect=0.5,
            n=100,
            alpha=0.05,
            variation_range=0.3,
            test_type="two_sample_t_test"
        )
        
        plot_data = json.loads(plot_json)
        assert 'data' in plot_data
        assert 'layout' in plot_data
        
        # Should have power and p-value curves
        assert len(plot_data['data']) >= 2
    
    def test_sample_size_optimization(self, visualizer):
        """Test sample size optimization visualization."""
        plot_json = visualizer.create_sample_size_optimization(
            effect_size=0.5,
            desired_power=0.8,
            alpha=0.05,
            max_n=500,
            test_type="two_sample_t_test"
        )
        
        plot_data = json.loads(plot_json)
        assert 'data' in plot_data
        assert 'layout' in plot_data
        
        # Should have bar chart
        assert len(plot_data['data']) >= 1
        assert plot_data['data'][0]['type'] == 'bar'
    
    def test_confidence_interval_plot(self, visualizer):
        """Test confidence interval visualization."""
        plot_json = visualizer.create_confidence_interval_plot(
            estimate=0.5,
            ci_lower=0.2,
            ci_upper=0.8,
            effect_name="Test Effect",
            comparison_value=0.3
        )
        
        plot_data = json.loads(plot_json)
        assert 'data' in plot_data
        assert 'layout' in plot_data
        
        # Should have CI line and markers
        assert len(plot_data['data']) >= 3
    
    def test_scenario_comparison(self, visualizer, sample_scenarios):
        """Test scenario comparison visualization."""
        plot_json = visualizer.create_scenario_comparison(sample_scenarios)
        
        plot_data = json.loads(plot_json)
        assert 'data' in plot_data
        assert 'layout' in plot_data
        
        # Should have bars for each metric
        assert len(plot_data['data']) >= 3
    
    def test_power_calculation_accuracy(self, visualizer):
        """Test accuracy of power calculations."""
        # Known case: n=64, d=0.5, alpha=0.05 should give ~0.8 power
        power = visualizer._calculate_power(
            n=64,
            effect_size=0.5,
            alpha=0.05,
            test_type="two_sample_t_test"
        )
        
        assert 0.75 <= power <= 0.85, f"Power {power} not in expected range"
    
    def test_p_value_calculation(self, visualizer):
        """Test p-value calculation."""
        # Large effect should give small p-value
        p_val = visualizer._calculate_p_value(
            n=100,
            effect_size=1.0,
            test_type="two_sample_t_test"
        )
        
        assert p_val < 0.001, f"P-value {p_val} not as expected for large effect"
    
    def test_required_n_calculation(self, visualizer):
        """Test required sample size calculation."""
        required_n = visualizer._find_required_n(
            effect_size=0.5,
            target_power=0.8,
            alpha=0.05,
            max_n=1000,
            test_type="two_sample_t_test"
        )
        
        # Should be around 64 for d=0.5, power=0.8
        assert 50 <= required_n <= 80, f"Required n={required_n} not in expected range"
    
    def test_simulation_reproducibility(self, visualizer):
        """Test that simulations are reproducible with seed."""
        p_values_1 = visualizer._simulate_p_values(
            n=50,
            effect_size=0.5,
            simulations=100,
            test_type="two_sample_t_test"
        )
        
        p_values_2 = visualizer._simulate_p_values(
            n=50,
            effect_size=0.5,
            simulations=100,
            test_type="two_sample_t_test"
        )
        
        # With fixed seed in method, should be identical
        assert p_values_1 == p_values_2
    
    def test_edge_cases(self, visualizer):
        """Test edge cases in visualizations."""
        # Very small sample size
        power_small = visualizer._calculate_power(
            n=10,
            effect_size=0.5,
            alpha=0.05,
            test_type="two_sample_t_test"
        )
        assert 0 <= power_small <= 1
        
        # Very large effect size
        power_large = visualizer._calculate_power(
            n=100,
            effect_size=3.0,
            alpha=0.05,
            test_type="two_sample_t_test"
        )
        assert power_large > 0.99
        
        # Zero effect size
        power_zero = visualizer._calculate_power(
            n=100,
            effect_size=0.0,
            alpha=0.05,
            test_type="two_sample_t_test"
        )
        assert power_zero <= 0.05  # Should be close to alpha