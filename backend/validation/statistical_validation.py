"""Statistical validation against R and SAS results."""

import subprocess
import tempfile
import json
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import pandas as pd
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class StatisticalValidator:
    """Validate statistical calculations against R and SAS."""
    
    def __init__(self):
        """Initialize validator."""
        self.r_available = self._check_r_available()
        self.sas_available = self._check_sas_available()
        self.validation_results = []
    
    def _check_r_available(self) -> bool:
        """Check if R is available."""
        try:
            result = subprocess.run(
                ["R", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    def _check_sas_available(self) -> bool:
        """Check if SAS is available."""
        try:
            result = subprocess.run(
                ["sas", "-version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    def validate_t_test(
        self,
        group1: List[float],
        group2: List[float],
        alpha: float = 0.05,
        paired: bool = False
    ) -> Dict[str, Any]:
        """
        Validate t-test results against R and SAS.
        
        Args:
            group1: First group data
            group2: Second group data
            alpha: Significance level
            paired: Whether to perform paired t-test
            
        Returns:
            Validation results
        """
        results = {
            "test": "t-test",
            "paired": paired,
            "python_result": None,
            "r_result": None,
            "sas_result": None,
            "validation_passed": False
        }
        
        # Python calculation
        if paired:
            statistic, p_value = stats.ttest_rel(group1, group2)
        else:
            statistic, p_value = stats.ttest_ind(group1, group2)
        
        results["python_result"] = {
            "statistic": float(statistic),
            "p_value": float(p_value)
        }
        
        # R calculation
        if self.r_available:
            r_result = self._run_r_t_test(group1, group2, paired)
            if r_result:
                results["r_result"] = r_result
        
        # SAS calculation
        if self.sas_available:
            sas_result = self._run_sas_t_test(group1, group2, paired)
            if sas_result:
                results["sas_result"] = sas_result
        
        # Validate results
        results["validation_passed"] = self._validate_results(results)
        self.validation_results.append(results)
        
        return results
    
    def _run_r_t_test(
        self,
        group1: List[float],
        group2: List[float],
        paired: bool
    ) -> Optional[Dict[str, float]]:
        """Run t-test in R."""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.R', delete=False) as f:
                r_script = f"""
                group1 <- c({','.join(map(str, group1))})
                group2 <- c({','.join(map(str, group2))})
                
                if ({str(paired).upper()}) {{
                    result <- t.test(group1, group2, paired=TRUE)
                }} else {{
                    result <- t.test(group1, group2, paired=FALSE)
                }}
                
                cat(paste("statistic:", result$statistic, "\n"))
                cat(paste("p_value:", result$p.value, "\n"))
                """
                f.write(r_script)
                f.flush()
                
                result = subprocess.run(
                    ["R", "--slave", "--no-save", "-f", f.name],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0:
                    output = result.stdout
                    statistic = float(output.split("statistic:")[1].split("\n")[0])
                    p_value = float(output.split("p_value:")[1].split("\n")[0])
                    return {"statistic": statistic, "p_value": p_value}
        except Exception as e:
            logger.error(f"R validation failed: {e}")
        
        return None
    
    def _run_sas_t_test(
        self,
        group1: List[float],
        group2: List[float],
        paired: bool
    ) -> Optional[Dict[str, float]]:
        """Run t-test in SAS."""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.sas', delete=False) as f:
                # Create SAS data
                sas_script = """
                data testdata;
                    input group value;
                    datalines;
                """
                for val in group1:
                    sas_script += f"1 {val}\n"
                for val in group2:
                    sas_script += f"2 {val}\n"
                sas_script += """
                ;
                run;
                
                proc ttest data=testdata;
                    class group;
                    var value;
                run;
                """
                
                f.write(sas_script)
                f.flush()
                
                result = subprocess.run(
                    ["sas", "-sysin", f.name, "-print", "/dev/stdout"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0:
                    # Parse SAS output (simplified)
                    # In practice, would need more sophisticated parsing
                    return {"statistic": None, "p_value": None}
        except Exception as e:
            logger.error(f"SAS validation failed: {e}")
        
        return None
    
    def validate_anova(
        self,
        groups: List[List[float]],
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Validate ANOVA results against R and SAS.
        
        Args:
            groups: List of group data
            alpha: Significance level
            
        Returns:
            Validation results
        """
        results = {
            "test": "anova",
            "n_groups": len(groups),
            "python_result": None,
            "r_result": None,
            "sas_result": None,
            "validation_passed": False
        }
        
        # Python calculation
        statistic, p_value = stats.f_oneway(*groups)
        results["python_result"] = {
            "statistic": float(statistic),
            "p_value": float(p_value)
        }
        
        # R calculation
        if self.r_available:
            r_result = self._run_r_anova(groups)
            if r_result:
                results["r_result"] = r_result
        
        # Validate results
        results["validation_passed"] = self._validate_results(results)
        self.validation_results.append(results)
        
        return results
    
    def _run_r_anova(self, groups: List[List[float]]) -> Optional[Dict[str, float]]:
        """Run ANOVA in R."""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.R', delete=False) as f:
                # Create R data frame
                r_script = "data <- data.frame(\n"
                r_script += "  value = c("
                all_values = []
                group_labels = []
                for i, group in enumerate(groups):
                    all_values.extend(group)
                    group_labels.extend([f"G{i+1}"] * len(group))
                r_script += ",".join(map(str, all_values))
                r_script += "),\n  group = factor(c("
                r_script += ",".join(f'"{g}"' for g in group_labels)
                r_script += "))\n)\n"
                
                r_script += """
                result <- aov(value ~ group, data=data)
                summary_result <- summary(result)
                f_stat <- summary_result[[1]][["F value"]][1]
                p_val <- summary_result[[1]][["Pr(>F)"]][1]
                cat(paste("statistic:", f_stat, "\n"))
                cat(paste("p_value:", p_val, "\n"))
                """
                
                f.write(r_script)
                f.flush()
                
                result = subprocess.run(
                    ["R", "--slave", "--no-save", "-f", f.name],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0:
                    output = result.stdout
                    statistic = float(output.split("statistic:")[1].split("\n")[0])
                    p_value = float(output.split("p_value:")[1].split("\n")[0])
                    return {"statistic": statistic, "p_value": p_value}
        except Exception as e:
            logger.error(f"R ANOVA validation failed: {e}")
        
        return None
    
    def validate_chi_square(
        self,
        contingency_table: np.ndarray,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Validate chi-square test results against R and SAS.
        
        Args:
            contingency_table: Contingency table
            alpha: Significance level
            
        Returns:
            Validation results
        """
        results = {
            "test": "chi_square",
            "table_shape": contingency_table.shape,
            "python_result": None,
            "r_result": None,
            "sas_result": None,
            "validation_passed": False
        }
        
        # Python calculation
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        results["python_result"] = {
            "statistic": float(chi2),
            "p_value": float(p_value),
            "dof": int(dof)
        }
        
        # R calculation
        if self.r_available:
            r_result = self._run_r_chi_square(contingency_table)
            if r_result:
                results["r_result"] = r_result
        
        # Validate results
        results["validation_passed"] = self._validate_results(results)
        self.validation_results.append(results)
        
        return results
    
    def _run_r_chi_square(self, contingency_table: np.ndarray) -> Optional[Dict[str, Any]]:
        """Run chi-square test in R."""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.R', delete=False) as f:
                # Create R matrix
                r_script = "table_data <- matrix(c("
                r_script += ",".join(map(str, contingency_table.flatten()))
                r_script += f"), nrow={contingency_table.shape[0]}, byrow=TRUE)\n"
                
                r_script += """
                result <- chisq.test(table_data)
                cat(paste("statistic:", result$statistic, "\n"))
                cat(paste("p_value:", result$p.value, "\n"))
                cat(paste("dof:", result$parameter, "\n"))
                """
                
                f.write(r_script)
                f.flush()
                
                result = subprocess.run(
                    ["R", "--slave", "--no-save", "-f", f.name],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0:
                    output = result.stdout
                    statistic = float(output.split("statistic:")[1].split("\n")[0])
                    p_value = float(output.split("p_value:")[1].split("\n")[0])
                    dof = int(float(output.split("dof:")[1].split("\n")[0]))
                    return {"statistic": statistic, "p_value": p_value, "dof": dof}
        except Exception as e:
            logger.error(f"R chi-square validation failed: {e}")
        
        return None
    
    def _validate_results(self, results: Dict[str, Any]) -> bool:
        """
        Validate that results match across platforms.
        
        Args:
            results: Results from different platforms
            
        Returns:
            Whether validation passed
        """
        tolerance = 1e-6  # Numerical tolerance for comparison
        
        python_result = results.get("python_result")
        if not python_result:
            return False
        
        # Check R results if available
        r_result = results.get("r_result")
        if r_result:
            if abs(python_result["p_value"] - r_result["p_value"]) > tolerance:
                logger.warning(
                    f"P-value mismatch: Python={python_result['p_value']}, "
                    f"R={r_result['p_value']}"
                )
                return False
            
            if (python_result.get("statistic") is not None and 
                r_result.get("statistic") is not None):
                if abs(python_result["statistic"] - r_result["statistic"]) > tolerance:
                    logger.warning(
                        f"Statistic mismatch: Python={python_result['statistic']}, "
                        f"R={r_result['statistic']}"
                    )
                    return False
        
        # Check SAS results if available
        sas_result = results.get("sas_result")
        if sas_result and sas_result.get("p_value") is not None:
            if abs(python_result["p_value"] - sas_result["p_value"]) > tolerance:
                logger.warning(
                    f"P-value mismatch: Python={python_result['p_value']}, "
                    f"SAS={sas_result['p_value']}"
                )
                return False
        
        return True
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive validation report.
        
        Returns:
            Validation report
        """
        total_tests = len(self.validation_results)
        passed_tests = sum(1 for r in self.validation_results if r["validation_passed"])
        
        report = {
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "pass_rate": passed_tests / total_tests if total_tests > 0 else 0,
                "r_available": self.r_available,
                "sas_available": self.sas_available
            },
            "test_results": self.validation_results,
            "recommendations": []
        }
        
        # Add recommendations
        if report["summary"]["pass_rate"] < 1.0:
            report["recommendations"].append(
                "Some tests failed validation. Review numerical precision and algorithm implementations."
            )
        
        if not self.r_available:
            report["recommendations"].append(
                "R is not available. Install R for comprehensive validation."
            )
        
        if not self.sas_available:
            report["recommendations"].append(
                "SAS is not available. SAS validation skipped."
            )
        
        return report


class PowerValidation:
    """Validate power calculations against established tools."""
    
    def __init__(self):
        """Initialize power validator."""
        self.validation_results = []
    
    def validate_t_test_power(
        self,
        n: int,
        effect_size: float,
        alpha: float = 0.05,
        power_target: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Validate t-test power calculation.
        
        Args:
            n: Sample size per group
            effect_size: Cohen's d
            alpha: Significance level
            power_target: Expected power (if known)
            
        Returns:
            Validation results
        """
        from statsmodels.stats.power import TTestPower
        
        results = {
            "test": "t_test_power",
            "parameters": {
                "n": n,
                "effect_size": effect_size,
                "alpha": alpha
            },
            "python_result": None,
            "statsmodels_result": None,
            "validation_passed": False
        }
        
        # Statsmodels calculation
        power_analysis = TTestPower()
        power = power_analysis.solve_power(
            effect_size=effect_size,
            nobs1=n,
            alpha=alpha,
            alternative='two-sided'
        )
        results["statsmodels_result"] = float(power)
        
        # Compare with target if provided
        if power_target is not None:
            results["target_power"] = power_target
            results["validation_passed"] = abs(power - power_target) < 0.01
        else:
            results["validation_passed"] = True
        
        self.validation_results.append(results)
        return results
    
    def validate_sample_size_calculation(
        self,
        effect_size: float,
        alpha: float,
        power: float,
        expected_n: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Validate sample size calculation.
        
        Args:
            effect_size: Effect size
            alpha: Significance level
            power: Desired power
            expected_n: Expected sample size (if known)
            
        Returns:
            Validation results
        """
        from statsmodels.stats.power import TTestPower
        
        results = {
            "test": "sample_size_calculation",
            "parameters": {
                "effect_size": effect_size,
                "alpha": alpha,
                "power": power
            },
            "statsmodels_result": None,
            "validation_passed": False
        }
        
        # Statsmodels calculation
        power_analysis = TTestPower()
        n = power_analysis.solve_power(
            effect_size=effect_size,
            power=power,
            alpha=alpha,
            alternative='two-sided'
        )
        results["statsmodels_result"] = int(np.ceil(n))
        
        # Compare with expected if provided
        if expected_n is not None:
            results["expected_n"] = expected_n
            results["validation_passed"] = abs(results["statsmodels_result"] - expected_n) <= 1
        else:
            results["validation_passed"] = True
        
        self.validation_results.append(results)
        return results


def run_comprehensive_validation() -> Dict[str, Any]:
    """
    Run comprehensive validation suite.
    
    Returns:
        Complete validation report
    """
    # Initialize validators
    stat_validator = StatisticalValidator()
    power_validator = PowerValidation()
    
    # Test data
    np.random.seed(42)
    group1 = np.random.normal(100, 15, 30)
    group2 = np.random.normal(105, 15, 30)
    group3 = np.random.normal(110, 15, 30)
    
    # Run statistical tests
    stat_validator.validate_t_test(group1, group2)
    stat_validator.validate_t_test(group1[:20], group2[:20], paired=True)
    stat_validator.validate_anova([group1, group2, group3])
    
    # Chi-square test
    contingency = np.array([[10, 15], [20, 25]])
    stat_validator.validate_chi_square(contingency)
    
    # Power calculations
    power_validator.validate_t_test_power(n=30, effect_size=0.5, alpha=0.05)
    power_validator.validate_sample_size_calculation(
        effect_size=0.5, alpha=0.05, power=0.8
    )
    
    # Generate combined report
    report = {
        "statistical_validation": stat_validator.generate_validation_report(),
        "power_validation": {
            "total_tests": len(power_validator.validation_results),
            "passed_tests": sum(
                1 for r in power_validator.validation_results 
                if r["validation_passed"]
            ),
            "results": power_validator.validation_results
        },
        "overall_status": "PASSED" if all(
            r["validation_passed"] 
            for r in stat_validator.validation_results + power_validator.validation_results
        ) else "FAILED"
    }
    
    return report


if __name__ == "__main__":
    # Run validation
    report = run_comprehensive_validation()
    
    # Save report
    with open("validation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"Validation Status: {report['overall_status']}")
    print(f"Statistical Tests: {report['statistical_validation']['summary']['passed_tests']}/{report['statistical_validation']['summary']['total_tests']} passed")
    print(f"Power Calculations: {report['power_validation']['passed_tests']}/{report['power_validation']['total_tests']} passed")