"""Visualization components for Streamlit interface."""

import streamlit as st
import plotly.graph_objects as go
import json
from typing import Dict, Any, List, Optional
import requests
import pandas as pd


class InteractivePlots:
    """Manages interactive Plotly visualizations."""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        """Initialize with API base URL."""
        self.api_base_url = api_base_url
    
    def render_power_curve(
        self,
        effect_sizes: List[float],
        alpha: float = 0.05,
        test_type: str = "two_sample_t_test",
        n_min: int = 10,
        n_max: int = 500,
        n_step: int = 10
    ) -> None:
        """
        Render interactive power curve.
        
        Args:
            effect_sizes: List of effect sizes to display
            alpha: Significance level
            test_type: Type of statistical test
            n_min: Minimum sample size
            n_max: Maximum sample size
            n_step: Step size for sample size range
        """
        try:
            # Call API to get plot data
            response = requests.post(
                f"{self.api_base_url}/api/visualize/power-curve",
                json={
                    "n_min": n_min,
                    "n_max": n_max,
                    "n_step": n_step,
                    "effect_sizes": effect_sizes,
                    "alpha": alpha,
                    "test_type": test_type
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                fig = go.Figure(json.loads(data['plot']))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"Failed to generate power curve: {response.text}")
        except Exception as e:
            # Fallback to local generation if API fails
            self._render_power_curve_local(effect_sizes, alpha, test_type, n_min, n_max, n_step)
    
    def _render_power_curve_local(
        self,
        effect_sizes: List[float],
        alpha: float,
        test_type: str,
        n_min: int,
        n_max: int,
        n_step: int
    ) -> None:
        """Local fallback for power curve generation."""
        import numpy as np
        from scipy import stats
        
        fig = go.Figure()
        n_range = list(range(n_min, n_max + 1, n_step))
        
        for effect_size in effect_sizes:
            powers = []
            for n in n_range:
                # Simplified power calculation
                if test_type == "two_sample_t_test":
                    n_per_group = n / 2
                    df = n - 2
                    nc = effect_size * np.sqrt(n_per_group / 2)
                    critical_t = stats.t.ppf(1 - alpha/2, df)
                    power = 1 - stats.nct.cdf(critical_t, df, nc) + stats.nct.cdf(-critical_t, df, nc)
                else:
                    power = 0.8  # Default
                powers.append(power)
            
            fig.add_trace(go.Scatter(
                x=n_range,
                y=powers,
                mode='lines+markers',
                name=f'd={effect_size}'
            ))
        
        fig.add_hline(y=0.8, line_dash="dash", line_color="gray",
                     annotation_text="80% Power")
        
        fig.update_layout(
            title="Statistical Power Analysis",
            xaxis_title="Total Sample Size",
            yaxis_title="Statistical Power",
            yaxis=dict(range=[0, 1.05], tickformat='.0%'),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_p_value_distribution(
        self,
        n: int,
        effect_size: float,
        alpha: float = 0.05,
        simulations: int = 10000,
        test_type: str = "two_sample_t_test"
    ) -> None:
        """
        Render p-value distribution visualization.
        
        Args:
            n: Sample size
            effect_size: Effect size
            alpha: Significance level
            simulations: Number of simulations
            test_type: Type of statistical test
        """
        try:
            response = requests.post(
                f"{self.api_base_url}/api/visualize/p-value-distribution",
                json={
                    "n": n,
                    "effect_size": effect_size,
                    "alpha": alpha,
                    "simulations": simulations,
                    "test_type": test_type
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                fig = go.Figure(json.loads(data['plot']))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"Failed to generate p-value distribution: {response.text}")
        except:
            self._render_p_value_distribution_local(n, effect_size, alpha, simulations, test_type)
    
    def _render_p_value_distribution_local(
        self,
        n: int,
        effect_size: float,
        alpha: float,
        simulations: int,
        test_type: str
    ) -> None:
        """Local fallback for p-value distribution."""
        import numpy as np
        from scipy import stats
        
        np.random.seed(42)
        
        # Simulate p-values
        p_values_null = []
        p_values_alt = []
        
        for _ in range(min(simulations, 1000)):  # Limit for performance
            # Null hypothesis
            group1 = np.random.normal(0, 1, n // 2)
            group2 = np.random.normal(0, 1, n // 2)
            _, p_null = stats.ttest_ind(group1, group2)
            p_values_null.append(p_null)
            
            # Alternative hypothesis
            group1 = np.random.normal(0, 1, n // 2)
            group2 = np.random.normal(effect_size, 1, n // 2)
            _, p_alt = stats.ttest_ind(group1, group2)
            p_values_alt.append(p_alt)
        
        # Create figure
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=p_values_null,
            name="Null Hypothesis",
            opacity=0.7,
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Histogram(
            x=p_values_alt,
            name="Alternative Hypothesis",
            opacity=0.7,
            marker_color='lightcoral'
        ))
        
        fig.add_vline(x=alpha, line_dash="dash", line_color="red",
                     annotation_text=f"α={alpha}")
        
        fig.update_layout(
            title=f"P-value Distribution (n={n})",
            xaxis_title="P-value",
            yaxis_title="Frequency",
            barmode='overlay'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_effect_sensitivity(
        self,
        base_effect: float,
        n: int,
        alpha: float = 0.05,
        variation_range: float = 0.5,
        test_type: str = "two_sample_t_test"
    ) -> None:
        """
        Render effect size sensitivity analysis.
        
        Args:
            base_effect: Central effect size estimate
            n: Sample size
            alpha: Significance level
            variation_range: Range of variation
            test_type: Type of statistical test
        """
        try:
            response = requests.post(
                f"{self.api_base_url}/api/visualize/effect-size-sensitivity",
                json={
                    "base_effect": base_effect,
                    "n": n,
                    "alpha": alpha,
                    "variation_range": variation_range,
                    "test_type": test_type
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                fig = go.Figure(json.loads(data['plot']))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"Failed to generate sensitivity analysis: {response.text}")
        except:
            st.warning("Sensitivity analysis visualization not available offline")
    
    def render_sample_size_optimization(
        self,
        effect_size: float,
        desired_power: float = 0.8,
        alpha: float = 0.05,
        max_n: int = 1000,
        test_type: str = "two_sample_t_test"
    ) -> None:
        """
        Render sample size optimization visualization.
        
        Args:
            effect_size: Effect size
            desired_power: Target power level
            alpha: Significance level
            max_n: Maximum sample size
            test_type: Type of statistical test
        """
        try:
            response = requests.post(
                f"{self.api_base_url}/api/visualize/sample-size-optimization",
                json={
                    "effect_size": effect_size,
                    "desired_power": desired_power,
                    "alpha": alpha,
                    "max_n": max_n,
                    "test_type": test_type
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                fig = go.Figure(json.loads(data['plot']))
                st.plotly_chart(fig, use_container_width=True)
                
                # Display required sample size
                if 'required_n' in data:
                    st.success(f"Required sample size for {desired_power:.0%} power: **{data['required_n']}**")
            else:
                st.error(f"Failed to generate optimization: {response.text}")
        except:
            st.warning("Sample size optimization not available offline")
    
    def render_confidence_interval(
        self,
        estimate: float,
        ci_lower: float,
        ci_upper: float,
        effect_name: str = "Effect",
        comparison_value: Optional[float] = None
    ) -> None:
        """
        Render confidence interval visualization.
        
        Args:
            estimate: Point estimate
            ci_lower: Lower CI bound
            ci_upper: Upper CI bound
            effect_name: Name of the effect
            comparison_value: Optional reference value
        """
        fig = go.Figure()
        
        # CI line
        fig.add_trace(go.Scatter(
            x=[ci_lower, ci_upper],
            y=[1, 1],
            mode='lines',
            line=dict(color='blue', width=4),
            showlegend=False
        ))
        
        # Point estimate
        fig.add_trace(go.Scatter(
            x=[estimate],
            y=[1],
            mode='markers',
            marker=dict(size=12, color='blue'),
            name='Estimate'
        ))
        
        # CI endpoints
        fig.add_trace(go.Scatter(
            x=[ci_lower, ci_upper],
            y=[1, 1],
            mode='markers',
            marker=dict(size=8, color='blue', symbol='line-ns'),
            name='95% CI'
        ))
        
        # Reference lines
        fig.add_vline(x=0, line_dash="dot", line_color="gray")
        if comparison_value is not None:
            fig.add_vline(x=comparison_value, line_dash="dash", line_color="red",
                         annotation_text="Reference")
        
        fig.update_layout(
            title=f"{effect_name}: {estimate:.3f} (95% CI: {ci_lower:.3f}, {ci_upper:.3f})",
            xaxis_title="Value",
            yaxis=dict(visible=False, range=[0.5, 1.5]),
            height=200,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretation
        if ci_lower > 0:
            st.success("✅ Confidence interval excludes zero (statistically significant)")
        elif ci_upper < 0:
            st.success("✅ Confidence interval excludes zero (statistically significant)")
        else:
            st.info("ℹ️ Confidence interval includes zero (not statistically significant)")
    
    def render_scenario_comparison(
        self,
        scenarios: List[Dict[str, Any]]
    ) -> None:
        """
        Render scenario comparison visualization.
        
        Args:
            scenarios: List of scenario dictionaries
        """
        if not scenarios:
            st.warning("No scenarios to compare")
            return
        
        # Create DataFrame for easier manipulation
        df = pd.DataFrame(scenarios)
        
        # Create subplots
        fig = go.Figure()
        
        # Add traces for each metric
        metrics = ['p_value', 'power', 'sample_size']
        colors = ['red', 'green', 'blue']
        
        for i, metric in enumerate(metrics):
            if metric in df.columns:
                fig.add_trace(go.Bar(
                    name=metric.replace('_', ' ').title(),
                    x=df.get('name', df.index),
                    y=df[metric],
                    marker_color=colors[i],
                    yaxis=f'y{i+1}' if i > 0 else 'y'
                ))
        
        # Update layout with multiple y-axes
        fig.update_layout(
            title="Scenario Comparison",
            xaxis=dict(title="Scenario"),
            yaxis=dict(title="P-value", side="left"),
            yaxis2=dict(title="Power", overlaying="y", side="right"),
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary table
        st.markdown("#### Scenario Summary")
        summary_df = df[['name', 'p_value', 'power', 'sample_size']].round(4)
        st.dataframe(
            summary_df,
            use_container_width=True,
            column_config={
                "p_value": st.column_config.NumberColumn("P-value", format="%.4f"),
                "power": st.column_config.NumberColumn("Power", format="%.2f"),
                "sample_size": st.column_config.NumberColumn("Sample Size", format="%d")
            }
        )


class ResultsVisualizer:
    """Manages comprehensive results visualization."""
    
    def render_results_dashboard(
        self,
        results: Dict[str, Any],
        display_options: Dict[str, bool]
    ) -> None:
        """
        Render comprehensive results dashboard.
        
        Args:
            results: Analysis results dictionary
            display_options: Display configuration flags
        """
        # Main metrics
        self._render_main_metrics(results)
        
        # Visualizations
        if display_options.get('show_visualizations', True):
            self._render_visualizations(results)
        
        # Effect sizes
        if display_options.get('show_effect_sizes', True):
            self._render_effect_sizes(results)
        
        # Confidence intervals
        if display_options.get('show_confidence_intervals', True):
            self._render_confidence_intervals(results)
        
        # Interpretations
        if display_options.get('show_interpretations', True):
            self._render_interpretations(results)
    
    def _render_main_metrics(self, results: Dict[str, Any]) -> None:
        """Render main statistical metrics."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            p_value = results.get('calculated_p_value', 'N/A')
            if isinstance(p_value, (int, float)):
                st.metric(
                    "P-value",
                    f"{p_value:.4f}",
                    delta="Significant" if p_value < 0.05 else "Not significant",
                    delta_color="normal" if p_value < 0.05 else "off"
                )
            else:
                st.metric("P-value", p_value)
        
        with col2:
            power = results.get('calculated_power', 'N/A')
            if isinstance(power, (int, float)):
                st.metric(
                    "Statistical Power",
                    f"{power:.2%}",
                    delta="Adequate" if power >= 0.8 else "Low",
                    delta_color="normal" if power >= 0.8 else "inverse"
                )
            else:
                st.metric("Statistical Power", power)
        
        with col3:
            test_used = results.get('statistical_test_used', 'N/A')
            st.metric("Test Used", test_used)
        
        with col4:
            confidence = results.get('confidence_level', 'N/A')
            if isinstance(confidence, (int, float)):
                st.metric("AI Confidence", f"{confidence:.1%}")
            else:
                st.metric("AI Confidence", confidence)
    
    def _render_visualizations(self, results: Dict[str, Any]) -> None:
        """Render visualization section."""
        st.markdown("### 📊 Visualizations")
        
        plots = InteractivePlots()
        
        # Create tabs for different visualizations
        tabs = st.tabs(["Power Curve", "P-value Distribution", "Sensitivity", "Sample Size"])
        
        with tabs[0]:
            if 'parameters' in results:
                params = results['parameters']
                effect_size = params.get('cohens_d', params.get('effect_size', 0.5))
                plots.render_power_curve(
                    effect_sizes=[effect_size * 0.5, effect_size, effect_size * 1.5],
                    alpha=params.get('alpha', 0.05),
                    test_type=results.get('suggested_study_type', 'two_sample_t_test')
                )
        
        with tabs[1]:
            if 'parameters' in results:
                params = results['parameters']
                plots.render_p_value_distribution(
                    n=params.get('total_n', params.get('n_total', 100)),
                    effect_size=params.get('cohens_d', params.get('effect_size', 0.5)),
                    alpha=params.get('alpha', 0.05),
                    test_type=results.get('suggested_study_type', 'two_sample_t_test')
                )
        
        with tabs[2]:
            if 'parameters' in results:
                params = results['parameters']
                plots.render_effect_sensitivity(
                    base_effect=params.get('cohens_d', params.get('effect_size', 0.5)),
                    n=params.get('total_n', params.get('n_total', 100)),
                    alpha=params.get('alpha', 0.05),
                    test_type=results.get('suggested_study_type', 'two_sample_t_test')
                )
        
        with tabs[3]:
            if 'parameters' in results:
                params = results['parameters']
                plots.render_sample_size_optimization(
                    effect_size=params.get('cohens_d', params.get('effect_size', 0.5)),
                    desired_power=0.8,
                    alpha=params.get('alpha', 0.05),
                    test_type=results.get('suggested_study_type', 'two_sample_t_test')
                )
    
    def _render_effect_sizes(self, results: Dict[str, Any]) -> None:
        """Render effect size information."""
        st.markdown("### 📏 Effect Sizes")
        
        if 'parameters' in results:
            params = results['parameters']
            
            # Determine effect size type and value
            if 'cohens_d' in params:
                effect_type = "Cohen's d"
                effect_value = params['cohens_d']
                interpretation = self._interpret_cohens_d(effect_value)
            elif 'effect_size' in params:
                effect_type = "Effect Size"
                effect_value = params['effect_size']
                interpretation = self._interpret_generic_effect(effect_value)
            else:
                st.info("No effect size information available")
                return
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(effect_type, f"{effect_value:.3f}")
            
            with col2:
                st.info(f"**Interpretation**: {interpretation}")
            
            # Additional context
            st.markdown("""
            **Effect Size Guidelines**:
            - Small: 0.2 - 0.5
            - Medium: 0.5 - 0.8
            - Large: > 0.8
            """)
    
    def _render_confidence_intervals(self, results: Dict[str, Any]) -> None:
        """Render confidence interval information."""
        st.markdown("### 📊 Confidence Intervals")
        
        # Check if we have CI data
        if 'confidence_interval' in results:
            ci = results['confidence_interval']
            plots = InteractivePlots()
            plots.render_confidence_interval(
                estimate=ci['estimate'],
                ci_lower=ci['lower'],
                ci_upper=ci['upper'],
                effect_name=ci.get('name', 'Effect')
            )
        else:
            # Calculate CI from available data
            if 'parameters' in results and 'calculated_p_value' in results:
                st.info("Confidence intervals will be calculated based on the analysis")
    
    def _render_interpretations(self, results: Dict[str, Any]) -> None:
        """Render statistical interpretations."""
        st.markdown("### 💡 Interpretations")
        
        # P-value interpretation
        if 'calculated_p_value' in results:
            p_val = results['calculated_p_value']
            if isinstance(p_val, (int, float)):
                if p_val < 0.001:
                    st.success("**Very strong evidence** against the null hypothesis (p < 0.001)")
                elif p_val < 0.01:
                    st.success("**Strong evidence** against the null hypothesis (p < 0.01)")
                elif p_val < 0.05:
                    st.success("**Moderate evidence** against the null hypothesis (p < 0.05)")
                elif p_val < 0.10:
                    st.warning("**Weak evidence** against the null hypothesis (p < 0.10)")
                else:
                    st.info("**No significant evidence** against the null hypothesis (p ≥ 0.10)")
        
        # Power interpretation
        if 'calculated_power' in results:
            power = results['calculated_power']
            if isinstance(power, (int, float)):
                if power >= 0.9:
                    st.success(f"**Excellent power** ({power:.1%}) - Very high probability of detecting true effects")
                elif power >= 0.8:
                    st.success(f"**Good power** ({power:.1%}) - Adequate probability of detecting true effects")
                elif power >= 0.7:
                    st.warning(f"**Marginal power** ({power:.1%}) - Moderate probability of detecting true effects")
                else:
                    st.error(f"**Insufficient power** ({power:.1%}) - Low probability of detecting true effects")
        
        # Rationale from AI
        if 'rationale' in results:
            st.markdown("**AI Analysis**:")
            st.write(results['rationale'])
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d value."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "Negligible effect"
        elif abs_d < 0.5:
            return "Small effect"
        elif abs_d < 0.8:
            return "Medium effect"
        else:
            return "Large effect"
    
    def _interpret_generic_effect(self, effect: float) -> str:
        """Interpret generic effect size."""
        if effect < 0.1:
            return "Very small effect"
        elif effect < 0.3:
            return "Small effect"
        elif effect < 0.5:
            return "Medium effect"
        else:
            return "Large effect"