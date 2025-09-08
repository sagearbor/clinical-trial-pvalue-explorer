"""API routes for visualization endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from backend.statistical.visualizations import TrialVisualizer

router = APIRouter(prefix="/api/visualize", tags=["visualizations"])
visualizer = TrialVisualizer()


class PowerCurveRequest(BaseModel):
    """Request model for power curve generation."""
    n_min: int = Field(default=10, ge=10, le=10000)
    n_max: int = Field(default=500, ge=10, le=10000)
    n_step: int = Field(default=10, ge=1)
    effect_sizes: List[float] = Field(default=[0.2, 0.5, 0.8])
    alpha: float = Field(default=0.05, ge=0.001, le=0.1)
    test_type: str = Field(default="two_sample_t_test")


class PValueDistributionRequest(BaseModel):
    """Request model for p-value distribution visualization."""
    n: int = Field(default=100, ge=10, le=10000)
    effect_size: float = Field(default=0.5, ge=0, le=3)
    alpha: float = Field(default=0.05, ge=0.001, le=0.1)
    simulations: int = Field(default=10000, ge=100, le=100000)
    test_type: str = Field(default="two_sample_t_test")


class EffectSensitivityRequest(BaseModel):
    """Request model for effect size sensitivity analysis."""
    base_effect: float = Field(default=0.5, ge=0, le=3)
    n: int = Field(default=100, ge=10, le=10000)
    alpha: float = Field(default=0.05, ge=0.001, le=0.1)
    variation_range: float = Field(default=0.5, ge=0.1, le=2)
    test_type: str = Field(default="two_sample_t_test")


class SampleSizeOptimizationRequest(BaseModel):
    """Request model for sample size optimization."""
    effect_size: float = Field(default=0.5, ge=0.1, le=3)
    desired_power: float = Field(default=0.8, ge=0.5, le=0.99)
    alpha: float = Field(default=0.05, ge=0.001, le=0.1)
    max_n: int = Field(default=1000, ge=100, le=10000)
    test_type: str = Field(default="two_sample_t_test")


class ConfidenceIntervalRequest(BaseModel):
    """Request model for confidence interval visualization."""
    estimate: float
    ci_lower: float
    ci_upper: float
    effect_name: str = Field(default="Effect")
    comparison_value: Optional[float] = None


class ScenarioComparisonRequest(BaseModel):
    """Request model for scenario comparison."""
    scenarios: List[Dict[str, Any]]


@router.post("/power-curve")
async def create_power_curve(request: PowerCurveRequest) -> Dict[str, Any]:
    """
    Generate interactive power curve visualization.
    
    Returns Plotly figure as JSON for different effect sizes and sample sizes.
    """
    try:
        n_range = list(range(request.n_min, request.n_max + 1, request.n_step))
        
        plot_json = visualizer.create_power_curve(
            n_range=n_range,
            effect_sizes=request.effect_sizes,
            alpha=request.alpha,
            test_type=request.test_type
        )
        
        return {
            "success": True,
            "plot": plot_json,
            "parameters": {
                "n_range": [min(n_range), max(n_range)],
                "effect_sizes": request.effect_sizes,
                "alpha": request.alpha,
                "test_type": request.test_type
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/p-value-distribution")
async def create_p_value_distribution(request: PValueDistributionRequest) -> Dict[str, Any]:
    """
    Visualize p-value distribution under null and alternative hypotheses.
    
    Shows distribution of p-values from simulated experiments.
    """
    try:
        plot_json = visualizer.create_p_value_distribution(
            n=request.n,
            effect_size=request.effect_size,
            alpha=request.alpha,
            simulations=request.simulations,
            test_type=request.test_type
        )
        
        return {
            "success": True,
            "plot": plot_json,
            "parameters": {
                "n": request.n,
                "effect_size": request.effect_size,
                "alpha": request.alpha,
                "simulations": request.simulations
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/effect-size-sensitivity")
async def create_effect_sensitivity(request: EffectSensitivityRequest) -> Dict[str, Any]:
    """
    Create sensitivity analysis for effect size variations.
    
    Shows how power and p-values change with effect size.
    """
    try:
        plot_json = visualizer.create_effect_size_sensitivity(
            base_effect=request.base_effect,
            n=request.n,
            alpha=request.alpha,
            variation_range=request.variation_range,
            test_type=request.test_type
        )
        
        return {
            "success": True,
            "plot": plot_json,
            "parameters": {
                "base_effect": request.base_effect,
                "n": request.n,
                "variation_range": request.variation_range
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sample-size-optimization")
async def optimize_sample_size(request: SampleSizeOptimizationRequest) -> Dict[str, Any]:
    """
    Visualize sample size requirements for different power levels.
    
    Helps determine optimal sample size for study design.
    """
    try:
        plot_json = visualizer.create_sample_size_optimization(
            effect_size=request.effect_size,
            desired_power=request.desired_power,
            alpha=request.alpha,
            max_n=request.max_n,
            test_type=request.test_type
        )
        
        # Calculate exact required n for desired power
        from backend.statistical.visualizations import TrialVisualizer
        viz = TrialVisualizer()
        required_n = viz._find_required_n(
            request.effect_size,
            request.desired_power,
            request.alpha,
            request.max_n,
            request.test_type
        )
        
        return {
            "success": True,
            "plot": plot_json,
            "required_n": required_n,
            "parameters": {
                "effect_size": request.effect_size,
                "desired_power": request.desired_power,
                "alpha": request.alpha
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/confidence-interval")
async def create_confidence_interval(request: ConfidenceIntervalRequest) -> Dict[str, Any]:
    """
    Create confidence interval visualization.
    
    Shows point estimate with confidence bounds.
    """
    try:
        plot_json = visualizer.create_confidence_interval_plot(
            estimate=request.estimate,
            ci_lower=request.ci_lower,
            ci_upper=request.ci_upper,
            effect_name=request.effect_name,
            comparison_value=request.comparison_value
        )
        
        # Calculate CI width and precision
        ci_width = request.ci_upper - request.ci_lower
        relative_precision = ci_width / abs(request.estimate) if request.estimate != 0 else float('inf')
        
        return {
            "success": True,
            "plot": plot_json,
            "statistics": {
                "ci_width": ci_width,
                "relative_precision": relative_precision,
                "includes_zero": request.ci_lower <= 0 <= request.ci_upper
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scenario-comparison")
async def compare_scenarios(request: ScenarioComparisonRequest) -> Dict[str, Any]:
    """
    Create comparison visualization for multiple scenarios.
    
    Compares p-values, power, and sample sizes across scenarios.
    """
    try:
        plot_json = visualizer.create_scenario_comparison(
            scenarios=request.scenarios
        )
        
        # Calculate summary statistics
        p_values = [s.get('p_value', 0) for s in request.scenarios]
        powers = [s.get('power', 0) for s in request.scenarios]
        
        return {
            "success": True,
            "plot": plot_json,
            "summary": {
                "n_scenarios": len(request.scenarios),
                "n_significant": sum(1 for p in p_values if p < 0.05),
                "n_powered": sum(1 for p in powers if p >= 0.8),
                "min_p_value": min(p_values) if p_values else None,
                "max_power": max(powers) if powers else None
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))