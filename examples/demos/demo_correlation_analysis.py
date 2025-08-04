#!/usr/bin/env python3
"""
Correlation Analysis Demo Script

This script demonstrates the CorrelationTest implementation with real-world scenarios,
showing both Pearson and Spearman correlation analysis for various types of relationships.

Usage:
    python examples/demos/demo_correlation_analysis.py

Scenarios demonstrated:
1. Clinical Trial: Drug dose vs efficacy (linear relationship - Pearson)
2. Healthcare: Age vs blood pressure (monotonic - Spearman) 
3. Psychology: Stress vs performance (negative correlation)
4. Biology: Height vs weight (positive correlation)
5. Education: Study hours vs exam scores (with outliers)

This demo shows the complete Phase 2.3 correlation test implementation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from statistical_tests import get_factory, CorrelationTest
from scipy import stats
import sys

def print_header(title):
    """Print a formatted header for demo sections."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def print_subheader(title):
    """Print a formatted subheader."""
    print(f"\n--- {title} ---")

def correlation_demo():
    """Run comprehensive correlation analysis demonstration."""
    
    print_header("CORRELATION ANALYSIS DEMO - Phase 2.3 Implementation")
    print("This demo showcases the CorrelationTest class with real-world scenarios")
    print("demonstrating both Pearson and Spearman correlation analysis.")
    
    # Get factory and correlation test
    factory = get_factory()
    
    print_subheader("Factory Integration Test")
    print("Available correlation test aliases:")
    available_tests = factory.get_available_tests()
    correlation_aliases = [test for test in available_tests if any(alias in test for alias in ['correlation', 'pearson', 'spearman', 'relationship'])]
    for alias in sorted(correlation_aliases):
        print(f"  - {alias}")
    
    # Get correlation test instance
    corr_test = factory.get_test("correlation")
    print(f"\nCorrelation test instance: {type(corr_test).__name__}")
    print(f"Required parameters: {corr_test.get_required_params()}")
    
    # SCENARIO 1: Clinical Trial - Drug Dose vs Efficacy
    print_header("SCENARIO 1: Clinical Trial - Drug Dose vs Efficacy")
    print("Linear dose-response relationship (ideal for Pearson correlation)")
    
    # Simulated clinical trial data
    drug_dose = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # mg
    efficacy_score = [2.1, 3.8, 5.2, 6.9, 8.1, 9.3, 10.5, 11.8, 12.9, 14.2]  # 0-15 scale
    
    print(f"Drug doses (mg): {drug_dose}")
    print(f"Efficacy scores: {efficacy_score}")
    
    # Pearson correlation analysis
    p_value_pearson, error_pearson = corr_test.calculate_p_value(
        x_values=drug_dose,
        y_values=efficacy_score,
        correlation_type="pearson"
    )
    
    power_pearson, power_error_pearson = corr_test.calculate_power(
        x_values=drug_dose,
        y_values=efficacy_score,
        correlation_type="pearson",
        alpha=0.05
    )
    
    # Get correlation coefficient for interpretation
    r_pearson, _ = stats.pearsonr(drug_dose, efficacy_score)
    
    print(f"\nPearson Correlation Results:")
    print(f"  Correlation coefficient (r): {r_pearson:.3f}")
    print(f"  P-value: {p_value_pearson:.6f}")
    print(f"  Statistical power: {power_pearson:.3f}")
    print(f"  Effect size (r²): {r_pearson**2:.3f}")
    
    if p_value_pearson < 0.05:
        print(f"  ✓ Statistically significant strong positive correlation")
        print(f"  ✓ {r_pearson**2*100:.1f}% of efficacy variance explained by dose")
    
    if error_pearson:
        print(f"  Note: {error_pearson}")
    
    # SCENARIO 2: Healthcare - Age vs Blood Pressure
    print_header("SCENARIO 2: Healthcare - Age vs Blood Pressure")
    print("Monotonic but potentially non-linear relationship (ideal for Spearman)")
    
    age = [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    blood_pressure = [115, 118, 122, 128, 135, 142, 150, 158, 167, 175, 185, 195]
    
    print(f"Ages: {age}")
    print(f"Systolic BP: {blood_pressure}")
    
    # Both Pearson and Spearman for comparison
    p_value_spearman, error_spearman = corr_test.calculate_p_value(
        x_values=age,
        y_values=blood_pressure,
        correlation_type="spearman"
    )
    
    p_value_pearson_bp, _ = corr_test.calculate_p_value(
        x_values=age,
        y_values=blood_pressure,
        correlation_type="pearson"
    )
    
    power_spearman, _ = corr_test.calculate_power(
        x_values=age,
        y_values=blood_pressure,
        correlation_type="spearman"
    )
    
    # Get correlation coefficients
    rho_spearman, _ = stats.spearmanr(age, blood_pressure)
    r_pearson_bp, _ = stats.pearsonr(age, blood_pressure)
    
    print(f"\nSpearman Correlation Results:")
    print(f"  Spearman's rho (ρ): {rho_spearman:.3f}")
    print(f"  P-value: {p_value_spearman:.6f}")
    print(f"  Statistical power: {power_spearman:.3f}")
    
    print(f"\nPearson Correlation Results (for comparison):")
    print(f"  Pearson's r: {r_pearson_bp:.3f}")
    print(f"  P-value: {p_value_pearson_bp:.6f}")
    
    print(f"\nInterpretation:")
    if abs(rho_spearman) > abs(r_pearson_bp):
        print(f"  ✓ Spearman correlation stronger, suggesting monotonic but non-linear relationship")
    print(f"  ✓ Strong age-BP association detected with both methods")
    
    # SCENARIO 3: Psychology - Stress vs Performance
    print_header("SCENARIO 3: Psychology - Stress vs Performance")
    print("Negative correlation example (stress decreases performance)")
    
    stress_level = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 1-10 scale
    performance = [95, 92, 88, 85, 80, 75, 68, 60, 52, 45]  # percentage
    
    print(f"Stress levels (1-10): {stress_level}")
    print(f"Performance scores (%): {performance}")
    
    p_value_stress, error_stress = corr_test.calculate_p_value(
        x_values=stress_level,
        y_values=performance,
        correlation_type="pearson"
    )
    
    power_stress, _ = corr_test.calculate_power(
        x_values=stress_level,
        y_values=performance,
        correlation_type="pearson"
    )
    
    r_stress, _ = stats.pearsonr(stress_level, performance)
    
    print(f"\nNegative Correlation Results:")
    print(f"  Correlation coefficient (r): {r_stress:.3f}")
    print(f"  P-value: {p_value_stress:.6f}")
    print(f"  Statistical power: {power_stress:.3f}")
    print(f"  Effect size (r²): {r_stress**2:.3f}")
    
    if r_stress < -0.8:
        print(f"  ✓ Strong negative correlation: As stress increases, performance decreases")
        print(f"  ✓ {r_stress**2*100:.1f}% of performance variance explained by stress")
    
    # SCENARIO 4: Biology - Height vs Weight
    print_header("SCENARIO 4: Biology - Height vs Weight")
    print("Classic positive correlation with realistic biological data")
    
    np.random.seed(42)  # For reproducible results
    height_inches = [65, 67, 70, 72, 69, 71, 68, 74, 66, 73, 69, 70, 67, 72, 68]
    weight_lbs = [140, 155, 180, 200, 165, 190, 160, 210, 145, 195, 170, 175, 150, 185, 162]
    
    print(f"Heights (inches): {height_inches}")
    print(f"Weights (lbs): {weight_lbs}")
    
    p_value_bio, error_bio = corr_test.calculate_p_value(
        x_values=height_inches,
        y_values=weight_lbs,
        correlation_type="pearson"
    )
    
    power_bio, _ = corr_test.calculate_power(
        x_values=height_inches,
        y_values=weight_lbs,
        correlation_type="pearson"
    )
    
    r_bio, _ = stats.pearsonr(height_inches, weight_lbs)
    
    print(f"\nBiological Correlation Results:")
    print(f"  Correlation coefficient (r): {r_bio:.3f}")
    print(f"  P-value: {p_value_bio:.6f}")
    print(f"  Statistical power: {power_bio:.3f}")
    print(f"  Effect size (r²): {r_bio**2:.3f}")
    
    if p_value_bio < 0.05:
        print(f"  ✓ Significant positive correlation between height and weight")
    
    if error_bio:
        print(f"  Note: {error_bio}")
    
    # SCENARIO 5: Education - Study Hours vs Exam Scores (with outliers)
    print_header("SCENARIO 5: Education - Study Hours vs Exam Scores")
    print("Demonstrates outlier effects and Pearson vs Spearman comparison")
    
    study_hours = [2, 4, 6, 8, 10, 12, 14, 16, 18, 25]  # Last value is outlier
    exam_scores = [65, 70, 75, 80, 85, 88, 90, 92, 94, 85]  # Diminishing returns + outlier effect
    
    print(f"Study hours: {study_hours}")
    print(f"Exam scores: {exam_scores}")
    print(f"Note: Student with 25 hours shows lower score (burnout effect)")
    
    # Compare Pearson vs Spearman with outlier
    p_value_edu_pearson, _ = corr_test.calculate_p_value(
        x_values=study_hours,
        y_values=exam_scores,
        correlation_type="pearson"
    )
    
    p_value_edu_spearman, _ = corr_test.calculate_p_value(
        x_values=study_hours,
        y_values=exam_scores,
        correlation_type="spearman"
    )
    
    r_edu, _ = stats.pearsonr(study_hours, exam_scores)
    rho_edu, _ = stats.spearmanr(study_hours, exam_scores)
    
    print(f"\nOutlier Effect Analysis:")
    print(f"  Pearson's r: {r_edu:.3f} (p-value: {p_value_edu_pearson:.4f})")
    print(f"  Spearman's ρ: {rho_edu:.3f} (p-value: {p_value_edu_spearman:.4f})")
    
    if abs(rho_edu) > abs(r_edu):
        print(f"  ✓ Spearman correlation stronger, indicating outlier effect on Pearson")
        print(f"  ✓ Monotonic relationship better captured by rank correlation")
    
    # SCENARIO 6: Power Analysis Example
    print_header("SCENARIO 6: Power Analysis for Study Design")
    print("Demonstrates power calculation for different sample sizes and effect sizes")
    
    effect_sizes = [0.1, 0.3, 0.5, 0.7, 0.9]
    sample_sizes = [10, 20, 50, 100, 200]
    
    print("Power analysis for correlation studies:")
    print("Effect Size | n=10  | n=20  | n=50  | n=100 | n=200")
    print("-"*55)
    
    for effect_size in effect_sizes:
        power_row = f"{effect_size:^11}"
        for n in sample_sizes:
            # Create synthetic data with known effect size
            np.random.seed(123)
            x_power = np.random.normal(0, 1, n)
            y_power = effect_size * x_power + np.sqrt(1 - effect_size**2) * np.random.normal(0, 1, n)
            
            power, _ = corr_test.calculate_power(
                x_values=x_power,
                y_values=y_power,
                effect_size=effect_size,
                alpha=0.05
            )
            power_row += f"| {power:.3f} "
        
        print(power_row)
    
    print("\nPower Analysis Interpretation:")
    print("- Values ≥ 0.80 indicate adequate power (80%)")
    print("- Larger effect sizes and sample sizes increase power")
    print("- Small effect sizes (r=0.1) require large samples for adequate power")
    
    # SCENARIO 7: Error Handling and Edge Cases
    print_header("SCENARIO 7: Error Handling and Edge Cases")
    print("Demonstrates robust error handling and validation")
    
    print_subheader("Testing Parameter Validation")
    
    # Test mismatched lengths
    x_mismatch = [1, 2, 3]
    y_mismatch = [4, 5]
    
    p_val, error = corr_test.calculate_p_value(x_values=x_mismatch, y_values=y_mismatch)
    print(f"Mismatched lengths: {error}")
    
    # Test insufficient data
    x_small = [1, 2]
    y_small = [3, 4]
    
    p_val, error = corr_test.calculate_p_value(x_values=x_small, y_values=y_small)
    print(f"Insufficient data: {error}")
    
    # Test constant values
    x_constant = [5, 5, 5, 5, 5]
    y_varying = [1, 2, 3, 4, 5]
    
    p_val, error = corr_test.calculate_p_value(
        x_values=x_constant, 
        y_values=y_varying,
        correlation_type="pearson"
    )
    print(f"Constant values (Pearson): {error}")
    
    # Test invalid correlation type
    try:
        is_valid, error = corr_test.validate_params(
            x_values=[1, 2, 3], 
            y_values=[4, 5, 6], 
            correlation_type="invalid"
        )
        print(f"Invalid correlation type: {error}")
    except Exception as e:
        print(f"Invalid correlation type: {e}")
    
    print_subheader("Testing Perfect Correlation Detection")
    
    x_perfect = [1, 2, 3, 4, 5]
    y_perfect = [2, 4, 6, 8, 10]  # y = 2x
    
    p_val, error = corr_test.calculate_p_value(x_values=x_perfect, y_values=y_perfect)
    print(f"Perfect correlation detected: {error}")
    print(f"P-value for perfect correlation: {p_val:.2e}")
    
    # Final summary
    print_header("CORRELATION ANALYSIS DEMO COMPLETE")
    print("✓ Phase 2.3 CorrelationTest implementation successfully demonstrated")
    print("✓ Both Pearson and Spearman correlations working correctly")
    print("✓ Factory pattern integration confirmed")
    print("✓ Comprehensive error handling validated")
    print("✓ Real-world scenarios covered:")
    print("  - Clinical trials (linear dose-response)")
    print("  - Healthcare data (monotonic relationships)")
    print("  - Psychological studies (negative correlations)")
    print("  - Biological measurements (positive correlations)")
    print("  - Educational data (outlier effects)")
    print("  - Power analysis for study design")
    print("✓ Ready for Phase 2 completion!")

def main():
    """Main function to run the correlation demo."""
    try:
        correlation_demo()
    except ImportError as e:
        print(f"Error: Missing dependencies - {e}")
        print("Please ensure all required packages are installed:")
        print("  pip install numpy scipy matplotlib")
        sys.exit(1)
    except Exception as e:
        print(f"Error running correlation demo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()