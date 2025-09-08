"""Tests for non-parametric statistical tests."""

import pytest
import numpy as np
from backend.statistical.nonparametric import (
    MannWhitneyUTest,
    KruskalWallisTest,
    WilcoxonSignedRankTest,
    SpearmanCorrelationTest,
    NonParametricTestFactory
)


class TestMannWhitneyU:
    """Test suite for Mann-Whitney U test."""
    
    def test_mann_whitney_basic(self):
        """Test basic Mann-Whitney U test functionality."""
        test = MannWhitneyUTest()
        
        # Create test data with known difference
        data1 = np.array([1, 2, 3, 4, 5])
        data2 = np.array([6, 7, 8, 9, 10])
        
        result = test.calculate(data1=data1, data2=data2)
        
        assert 'p_value' in result
        assert 'statistic' in result
        assert 'rank_biserial' in result
        assert result['p_value'] < 0.05  # Should detect significant difference
    
    def test_mann_whitney_no_difference(self):
        """Test Mann-Whitney U with no difference between groups."""
        test = MannWhitneyUTest()
        
        np.random.seed(42)
        data1 = np.random.normal(0, 1, 30)
        data2 = np.random.normal(0, 1, 30)
        
        result = test.calculate(data1=data1, data2=data2)
        
        assert result['p_value'] > 0.05  # Should not detect difference
        assert abs(result['rank_biserial']) < 0.3  # Small effect size
    
    def test_mann_whitney_simulation(self):
        """Test Mann-Whitney U with simulated data."""
        test = MannWhitneyUTest()
        
        result = test.calculate(n1=50, n2=50, effect_size=0.8)
        
        assert 'p_value' in result
        assert 'rank_biserial' in result
        assert result['n1'] == 50
        assert result['n2'] == 50
    
    def test_mann_whitney_power(self):
        """Test power calculation for Mann-Whitney U."""
        test = MannWhitneyUTest()
        
        power = test.calculate_power(
            n1=30,
            n2=30,
            effect_size=0.5,
            alpha=0.05
        )
        
        assert 0 <= power <= 1
        assert 0.3 <= power <= 0.7  # Moderate power for moderate effect
    
    def test_mann_whitney_validation(self):
        """Test parameter validation."""
        test = MannWhitneyUTest()
        
        # Valid parameters
        valid, error = test.validate_parameters(data1=[1, 2], data2=[3, 4])
        assert valid
        assert error is None
        
        # Invalid parameters (too few observations)
        valid, error = test.validate_parameters(data1=[1], data2=[2])
        assert not valid
        assert "at least 2 observations" in error


class TestKruskalWallis:
    """Test suite for Kruskal-Wallis test."""
    
    def test_kruskal_wallis_basic(self):
        """Test basic Kruskal-Wallis functionality."""
        test = KruskalWallisTest()
        
        # Create test data with different medians
        group1 = np.array([1, 2, 3, 4, 5])
        group2 = np.array([4, 5, 6, 7, 8])
        group3 = np.array([7, 8, 9, 10, 11])
        
        result = test.calculate(groups=[group1, group2, group3])
        
        assert 'p_value' in result
        assert 'statistic' in result
        assert 'epsilon_squared' in result
        assert result['p_value'] < 0.05  # Should detect difference
        assert result['n_groups'] == 3
    
    def test_kruskal_wallis_no_difference(self):
        """Test Kruskal-Wallis with no difference between groups."""
        test = KruskalWallisTest()
        
        np.random.seed(42)
        groups = [np.random.normal(0, 1, 20) for _ in range(3)]
        
        result = test.calculate(groups=groups)
        
        assert result['p_value'] > 0.05  # Should not detect difference
        assert result['epsilon_squared'] < 0.06  # Small effect
    
    def test_kruskal_wallis_simulation(self):
        """Test Kruskal-Wallis with simulated data."""
        test = KruskalWallisTest()
        
        result = test.calculate(n_groups=4, n_per_group=25, effect_size=0.3)
        
        assert 'p_value' in result
        assert result['n_groups'] == 4
        assert result['n_total'] == 100
    
    def test_kruskal_wallis_power(self):
        """Test power calculation for Kruskal-Wallis."""
        test = KruskalWallisTest()
        
        power = test.calculate_power(
            n_groups=3,
            n_per_group=30,
            effect_size=0.1,
            alpha=0.05
        )
        
        assert 0 <= power <= 1
        assert power > 0.5  # Should have decent power
    
    def test_kruskal_wallis_post_hoc(self):
        """Test post-hoc suggestion."""
        test = KruskalWallisTest()
        
        # Create data with clear differences
        groups = [
            np.array([1, 2, 3]),
            np.array([5, 6, 7]),
            np.array([9, 10, 11])
        ]
        
        result = test.calculate(groups=groups)
        
        if result['p_value'] < 0.05:
            assert result['post_hoc_suggestion'] is not None
            assert "Dunn" in result['post_hoc_suggestion']


class TestWilcoxonSignedRank:
    """Test suite for Wilcoxon signed-rank test."""
    
    def test_wilcoxon_basic(self):
        """Test basic Wilcoxon signed-rank functionality."""
        test = WilcoxonSignedRankTest()
        
        # Paired data with systematic difference
        data1 = np.array([1, 2, 3, 4, 5])
        data2 = np.array([2, 3, 4, 5, 6])
        
        result = test.calculate(data1=data1, data2=data2)
        
        assert 'p_value' in result
        assert 'statistic' in result
        assert 'median_difference' in result
        assert 'hodges_lehmann_estimate' in result
        assert result['n_pairs'] == 5
    
    def test_wilcoxon_differences(self):
        """Test Wilcoxon with direct differences."""
        test = WilcoxonSignedRankTest()
        
        differences = np.array([1, 2, 1.5, 0.5, 2.5, 1, 1.5])
        
        result = test.calculate(differences=differences)
        
        assert result['p_value'] < 0.05  # All positive differences
        assert result['median_difference'] > 0
    
    def test_wilcoxon_no_difference(self):
        """Test Wilcoxon with no systematic difference."""
        test = WilcoxonSignedRankTest()
        
        np.random.seed(42)
        n = 20
        data1 = np.random.normal(0, 1, n)
        data2 = data1 + np.random.normal(0, 0.1, n)  # Small random differences
        
        result = test.calculate(data1=data1, data2=data2)
        
        assert result['p_value'] > 0.05  # Should not detect difference
    
    def test_wilcoxon_power(self):
        """Test power calculation for Wilcoxon signed-rank."""
        test = WilcoxonSignedRankTest()
        
        power = test.calculate_power(
            n=20,
            effect_size=0.5,
            alpha=0.05
        )
        
        assert 0 <= power <= 1
        assert 0.3 <= power <= 0.8  # Reasonable power


class TestSpearmanCorrelation:
    """Test suite for Spearman correlation test."""
    
    def test_spearman_basic(self):
        """Test basic Spearman correlation functionality."""
        test = SpearmanCorrelationTest()
        
        # Create perfectly correlated ranks
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])
        
        result = test.calculate(x=x, y=y)
        
        assert 'correlation' in result
        assert 'p_value' in result
        assert 'ci_lower' in result
        assert 'ci_upper' in result
        assert abs(result['correlation'] - 1.0) < 0.01  # Perfect correlation
    
    def test_spearman_negative_correlation(self):
        """Test Spearman with negative correlation."""
        test = SpearmanCorrelationTest()
        
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([5, 4, 3, 2, 1])
        
        result = test.calculate(x=x, y=y)
        
        assert result['correlation'] < -0.9  # Strong negative correlation
        assert result['p_value'] < 0.05
    
    def test_spearman_no_correlation(self):
        """Test Spearman with no correlation."""
        test = SpearmanCorrelationTest()
        
        np.random.seed(42)
        x = np.random.normal(0, 1, 50)
        y = np.random.normal(0, 1, 50)
        
        result = test.calculate(x=x, y=y)
        
        assert abs(result['correlation']) < 0.3  # Weak correlation
        assert result['p_value'] > 0.05
    
    def test_spearman_confidence_interval(self):
        """Test confidence interval calculation."""
        test = SpearmanCorrelationTest()
        
        np.random.seed(42)
        n = 100
        # Create correlated data
        mean = [0, 0]
        cov = [[1, 0.5], [0.5, 1]]
        data = np.random.multivariate_normal(mean, cov, n)
        
        result = test.calculate(x=data[:, 0], y=data[:, 1])
        
        # CI should contain the correlation
        assert result['ci_lower'] < result['correlation'] < result['ci_upper']
        # CI should be reasonable width
        ci_width = result['ci_upper'] - result['ci_lower']
        assert 0.2 < ci_width < 0.6
    
    def test_spearman_power(self):
        """Test power calculation for Spearman correlation."""
        test = SpearmanCorrelationTest()
        
        power = test.calculate_power(
            n=50,
            correlation=0.4,
            alpha=0.05
        )
        
        assert 0 <= power <= 1
        assert 0.5 <= power <= 0.9  # Reasonable power


class TestNonParametricFactory:
    """Test suite for NonParametricTestFactory."""
    
    def test_factory_get_test(self):
        """Test getting tests from factory."""
        factory = NonParametricTestFactory()
        
        # Test getting Mann-Whitney U
        test = factory.get_test('mann_whitney')
        assert isinstance(test, MannWhitneyUTest)
        
        # Test alias
        test = factory.get_test('wilcoxon_ranksum')
        assert isinstance(test, MannWhitneyUTest)
        
        # Test Kruskal-Wallis
        test = factory.get_test('kruskal_wallis')
        assert isinstance(test, KruskalWallisTest)
        
        # Test Wilcoxon signed-rank
        test = factory.get_test('wilcoxon_signed')
        assert isinstance(test, WilcoxonSignedRankTest)
        
        # Test Spearman
        test = factory.get_test('spearman')
        assert isinstance(test, SpearmanCorrelationTest)
    
    def test_factory_list_tests(self):
        """Test listing available tests."""
        factory = NonParametricTestFactory()
        
        tests = factory.list_tests()
        
        assert 'mann_whitney' in tests
        assert 'kruskal_wallis' in tests
        assert 'wilcoxon_signed' in tests
        assert 'spearman' in tests
        assert len(tests) >= 6  # Including aliases
    
    def test_factory_invalid_test(self):
        """Test getting invalid test from factory."""
        factory = NonParametricTestFactory()
        
        test = factory.get_test('invalid_test')
        assert test is None