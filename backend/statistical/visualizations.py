"""Interactive visualization engine using Plotly."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Any, Optional, Tuple
from scipy import stats
import json


class TrialVisualizer:
    """Create interactive visualizations for clinical trial analysis."""
    
    def __init__(self, theme: str = "plotly_white"):
        """Initialize visualizer with theme."""
        self.theme = theme
        self.default_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    
    def create_power_curve(
        self,
        n_range: List[int],
        effect_sizes: List[float],
        alpha: float = 0.05,
        test_type: str = "two_sample_t_test"
    ) -> str:
        """
        Generate interactive power curve for different effect sizes.
        
        Args:
            n_range: Range of sample sizes to evaluate
            effect_sizes: List of effect sizes to compare
            alpha: Significance level
            test_type: Type of statistical test
            
        Returns:
            JSON string of Plotly figure
        """
        fig = go.Figure()
        
        for i, effect_size in enumerate(effect_sizes):
            power_values = []
            for n in n_range:
                power = self._calculate_power(n, effect_size, alpha, test_type)
                power_values.append(power)
            
            fig.add_trace(go.Scatter(
                x=n_range,
                y=power_values,
                mode='lines+markers',
                name=f'd={effect_size}',
                line=dict(color=self.default_colors[i % len(self.default_colors)], width=2),
                marker=dict(size=6),
                hovertemplate='<b>Effect Size: %{customdata}</b><br>' +
                             'Sample Size: %{x}<br>' +
                             'Power: %{y:.3f}<br>' +
                             '<extra></extra>',
                customdata=[effect_size] * len(n_range)
            ))
        
        # Add 80% power reference line
        fig.add_hline(
            y=0.8,
            line_dash="dash",
            line_color="gray",
            annotation_text="80% Power",
            annotation_position="right"
        )
        
        fig.update_layout(
            title="Statistical Power Analysis",
            xaxis_title="Total Sample Size",
            yaxis_title="Statistical Power",
            template=self.theme,
            hovermode='x unified',
            legend=dict(
                title="Effect Size",
                yanchor="bottom",
                y=0.01,
                xanchor="right",
                x=0.99
            ),
            yaxis=dict(range=[0, 1.05], tickformat='.0%'),
            height=500
        )
        
        return fig.to_json()
    
    def create_p_value_distribution(
        self,
        n: int,
        effect_size: float,
        alpha: float = 0.05,
        simulations: int = 10000,
        test_type: str = "two_sample_t_test"
    ) -> str:
        """
        Visualize p-value distribution under null and alternative hypotheses.
        
        Args:
            n: Sample size
            effect_size: Effect size
            alpha: Significance level
            simulations: Number of simulations
            test_type: Type of statistical test
            
        Returns:
            JSON string of Plotly figure
        """
        # Simulate p-values under null hypothesis
        p_values_null = self._simulate_p_values(n, 0, simulations, test_type)
        
        # Simulate p-values under alternative hypothesis
        p_values_alt = self._simulate_p_values(n, effect_size, simulations, test_type)
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Null Hypothesis (d=0)", f"Alternative Hypothesis (d={effect_size})"),
            horizontal_spacing=0.12
        )
        
        # Null hypothesis histogram
        fig.add_trace(
            go.Histogram(
                x=p_values_null,
                nbinsx=50,
                name="H₀",
                marker_color='lightblue',
                opacity=0.7,
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Alternative hypothesis histogram
        fig.add_trace(
            go.Histogram(
                x=p_values_alt,
                nbinsx=50,
                name="H₁",
                marker_color='lightcoral',
                opacity=0.7,
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Add significance threshold line
        for col in [1, 2]:
            fig.add_vline(
                x=alpha,
                line_dash="dash",
                line_color="red",
                annotation_text=f"α={alpha}",
                row=1, col=col
            )
        
        fig.update_xaxes(title_text="P-value", range=[0, 1], row=1, col=1)
        fig.update_xaxes(title_text="P-value", range=[0, 1], row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        
        fig.update_layout(
            title=f"P-value Distribution (n={n}, {simulations:,} simulations)",
            template=self.theme,
            height=400
        )
        
        return fig.to_json()
    
    def create_effect_size_sensitivity(
        self,
        base_effect: float,
        n: int,
        alpha: float = 0.05,
        variation_range: float = 0.5,
        test_type: str = "two_sample_t_test"
    ) -> str:
        """
        Create sensitivity analysis for effect size variations.
        
        Args:
            base_effect: Central effect size estimate
            n: Sample size
            alpha: Significance level
            variation_range: Range of variation (± from base)
            test_type: Type of statistical test
            
        Returns:
            JSON string of Plotly figure
        """
        effect_range = np.linspace(
            max(0, base_effect - variation_range),
            base_effect + variation_range,
            50
        )
        
        powers = []
        p_values = []
        
        for effect in effect_range:
            power = self._calculate_power(n, effect, alpha, test_type)
            p_val = self._calculate_p_value(n, effect, test_type)
            powers.append(power)
            p_values.append(p_val)
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Power curve
        fig.add_trace(
            go.Scatter(
                x=effect_range,
                y=powers,
                name="Power",
                line=dict(color='blue', width=2),
                mode='lines',
                hovertemplate='Effect: %{x:.3f}<br>Power: %{y:.3f}<extra></extra>'
            ),
            secondary_y=False
        )
        
        # P-value curve
        fig.add_trace(
            go.Scatter(
                x=effect_range,
                y=p_values,
                name="P-value",
                line=dict(color='red', width=2, dash='dash'),
                mode='lines',
                hovertemplate='Effect: %{x:.3f}<br>P-value: %{y:.4f}<extra></extra>'
            ),
            secondary_y=True
        )
        
        # Add reference lines
        fig.add_hline(y=0.8, line_dash="dot", line_color="blue", 
                     annotation_text="80% Power", secondary_y=False)
        fig.add_hline(y=alpha, line_dash="dot", line_color="red", 
                     annotation_text=f"α={alpha}", secondary_y=True)
        fig.add_vline(x=base_effect, line_dash="dash", line_color="gray",
                     annotation_text="Base Effect")
        
        fig.update_xaxes(title_text="Effect Size")
        fig.update_yaxes(title_text="Statistical Power", secondary_y=False, range=[0, 1.05])
        fig.update_yaxes(title_text="P-value", secondary_y=True, range=[0, 1], type="log")
        
        fig.update_layout(
            title=f"Effect Size Sensitivity Analysis (n={n})",
            template=self.theme,
            hovermode='x unified',
            height=500
        )
        
        return fig.to_json()
    
    def create_sample_size_optimization(
        self,
        effect_size: float,
        desired_power: float = 0.8,
        alpha: float = 0.05,
        max_n: int = 1000,
        test_type: str = "two_sample_t_test"
    ) -> str:
        """
        Visualize sample size requirements for different power levels.
        
        Args:
            effect_size: Effect size
            desired_power: Target power level
            alpha: Significance level
            max_n: Maximum sample size to consider
            test_type: Type of statistical test
            
        Returns:
            JSON string of Plotly figure
        """
        power_levels = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        required_n = []
        
        for power_target in power_levels:
            n = self._find_required_n(effect_size, power_target, alpha, max_n, test_type)
            required_n.append(n)
        
        fig = go.Figure()
        
        # Bar chart of required sample sizes
        fig.add_trace(go.Bar(
            x=[f"{p:.0%}" for p in power_levels],
            y=required_n,
            text=required_n,
            textposition='auto',
            marker_color=['lightcoral' if p < desired_power else 'lightgreen' 
                         for p in power_levels],
            hovertemplate='Power: %{x}<br>Required n: %{y}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"Sample Size Requirements (d={effect_size}, α={alpha})",
            xaxis_title="Target Power",
            yaxis_title="Total Sample Size Required",
            template=self.theme,
            height=400,
            showlegend=False
        )
        
        return fig.to_json()
    
    def create_confidence_interval_plot(
        self,
        estimate: float,
        ci_lower: float,
        ci_upper: float,
        effect_name: str = "Effect",
        comparison_value: Optional[float] = None
    ) -> str:
        """
        Create confidence interval visualization.
        
        Args:
            estimate: Point estimate
            ci_lower: Lower CI bound
            ci_upper: Upper CI bound
            effect_name: Name of the effect
            comparison_value: Optional reference value
            
        Returns:
            JSON string of Plotly figure
        """
        fig = go.Figure()
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=[ci_lower, ci_upper],
            y=[effect_name, effect_name],
            mode='lines',
            line=dict(color='blue', width=3),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Point estimate
        fig.add_trace(go.Scatter(
            x=[estimate],
            y=[effect_name],
            mode='markers',
            marker=dict(size=12, color='blue'),
            name='Point Estimate',
            hovertemplate=f'Estimate: {estimate:.3f}<extra></extra>'
        ))
        
        # CI endpoints
        fig.add_trace(go.Scatter(
            x=[ci_lower, ci_upper],
            y=[effect_name, effect_name],
            mode='markers',
            marker=dict(size=8, color='blue'),
            name='95% CI',
            hovertemplate='CI: %{x:.3f}<extra></extra>'
        ))
        
        # Add reference line if provided
        if comparison_value is not None:
            fig.add_vline(
                x=comparison_value,
                line_dash="dash",
                line_color="red",
                annotation_text="Reference"
            )
        
        # Add zero line
        fig.add_vline(x=0, line_dash="dot", line_color="gray")
        
        fig.update_layout(
            title=f"{effect_name}: {estimate:.3f} (95% CI: {ci_lower:.3f}, {ci_upper:.3f})",
            xaxis_title="Effect Size",
            template=self.theme,
            height=200,
            showlegend=True,
            yaxis=dict(visible=False)
        )
        
        return fig.to_json()
    
    def create_scenario_comparison(
        self,
        scenarios: List[Dict[str, Any]]
    ) -> str:
        """
        Create comparison visualization for multiple scenarios.
        
        Args:
            scenarios: List of scenario dictionaries with keys:
                       'name', 'p_value', 'power', 'sample_size'
                       
        Returns:
            JSON string of Plotly figure
        """
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=("P-values", "Statistical Power", "Sample Sizes"),
            horizontal_spacing=0.15
        )
        
        names = [s['name'] for s in scenarios]
        p_values = [s.get('p_value', 0) for s in scenarios]
        powers = [s.get('power', 0) for s in scenarios]
        sample_sizes = [s.get('sample_size', 0) for s in scenarios]
        
        # P-values
        fig.add_trace(
            go.Bar(
                x=names,
                y=p_values,
                marker_color=['red' if p < 0.05 else 'gray' for p in p_values],
                text=[f"{p:.4f}" for p in p_values],
                textposition='auto',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Power
        fig.add_trace(
            go.Bar(
                x=names,
                y=powers,
                marker_color=['green' if p >= 0.8 else 'orange' for p in powers],
                text=[f"{p:.2%}" for p in powers],
                textposition='auto',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Sample sizes
        fig.add_trace(
            go.Bar(
                x=names,
                y=sample_sizes,
                marker_color='lightblue',
                text=sample_sizes,
                textposition='auto',
                showlegend=False
            ),
            row=1, col=3
        )
        
        # Add reference lines
        fig.add_hline(y=0.05, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=0.8, line_dash="dash", line_color="green", row=1, col=2)
        
        fig.update_yaxes(title_text="P-value", row=1, col=1)
        fig.update_yaxes(title_text="Power", row=1, col=2)
        fig.update_yaxes(title_text="Sample Size", row=1, col=3)
        
        fig.update_layout(
            title="Scenario Comparison",
            template=self.theme,
            height=400,
            showlegend=False
        )
        
        return fig.to_json()
    
    # Helper methods
    def _calculate_power(
        self,
        n: int,
        effect_size: float,
        alpha: float,
        test_type: str
    ) -> float:
        """Calculate statistical power for given parameters."""
        if test_type == "two_sample_t_test":
            # Proper power calculation for t-test
            from statsmodels.stats.power import ttest_power
            try:
                # statsmodels uses n per group, not total n
                n_per_group = n / 2
                power = ttest_power(effect_size, n_per_group, alpha, alternative='two-sided')
                return float(power)
            except:
                # Fallback calculation
                n_per_group = n / 2
                df = n - 2
                # Correct non-centrality parameter
                nc = effect_size * np.sqrt(n_per_group * n_per_group / (n_per_group + n_per_group))
                critical_t = stats.t.ppf(1 - alpha/2, df)
                if np.isnan(nc) or np.isinf(nc):
                    return alpha  # Return alpha for zero effect
                power = 1 - stats.nct.cdf(critical_t, df, nc) + stats.nct.cdf(-critical_t, df, nc)
                return max(0, min(1, power))  # Ensure in [0, 1]
        # Add other test types as needed
        return 0.8  # Default
    
    def _calculate_p_value(
        self,
        n: int,
        effect_size: float,
        test_type: str
    ) -> float:
        """Calculate expected p-value for given parameters."""
        if test_type == "two_sample_t_test":
            # Simplified p-value calculation
            t_stat = effect_size * np.sqrt(n / 2)
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
            return p_value
        return 0.05  # Default
    
    def _simulate_p_values(
        self,
        n: int,
        effect_size: float,
        simulations: int,
        test_type: str
    ) -> List[float]:
        """Simulate p-values for given parameters."""
        np.random.seed(42)  # For reproducibility
        p_values = []
        
        for _ in range(simulations):
            if test_type == "two_sample_t_test":
                # Simulate two groups
                group1 = np.random.normal(0, 1, n // 2)
                group2 = np.random.normal(effect_size, 1, n // 2)
                _, p_val = stats.ttest_ind(group1, group2)
                p_values.append(p_val)
        
        return p_values
    
    def _find_required_n(
        self,
        effect_size: float,
        target_power: float,
        alpha: float,
        max_n: int,
        test_type: str
    ) -> int:
        """Find required sample size for target power."""
        for n in range(10, max_n, 2):
            power = self._calculate_power(n, effect_size, alpha, test_type)
            if power >= target_power:
                return n
        return max_n