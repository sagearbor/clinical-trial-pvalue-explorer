"""Additional statistical tests to complete the suite."""

import numpy as np
from scipy import stats
from typing import Tuple, Optional, Dict, Any, List
from abc import ABC, abstractmethod
import statsmodels.api as sm
from statsmodels.stats.power import ttest_power


class PairedTTest:
    """
    Paired t-test for dependent samples.
    
    Used for within-subjects designs, pre-post comparisons.
    """
    
    def calculate(
        self,
        data1: Optional[np.ndarray] = None,
        data2: Optional[np.ndarray] = None,
        differences: Optional[np.ndarray] = None,
        n_pairs: Optional[int] = None,
        mean_diff: Optional[float] = None,
        std_diff: Optional[float] = None,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Calculate paired t-test.
        
        Args:
            data1, data2: Paired observations
            differences: Direct differences (if data1/data2 not provided)
            n_pairs: Number of pairs (for simulation)
            mean_diff: Mean difference (for power calculation)
            std_diff: Standard deviation of differences
            alpha: Significance level
            
        Returns:
            Dictionary with test results
        """
        # Handle different input formats
        if data1 is not None and data2 is not None:
            data1 = np.array(data1)
            data2 = np.array(data2)
            if len(data1) != len(data2):
                return {"error": "Paired samples must have equal length"}
            differences = data1 - data2
            n_pairs = len(data1)
        elif differences is not None:
            differences = np.array(differences)
            n_pairs = len(differences)
        elif n_pairs and mean_diff is not None and std_diff is not None:
            # Simulate data
            differences = np.random.normal(mean_diff, std_diff, n_pairs)
        else:
            return {"error": "Insufficient data provided"}
        
        # Perform paired t-test
        t_stat, p_value = stats.ttest_1samp(differences, 0)
        
        # Calculate effect size (Cohen's d for paired samples)
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0
        
        # Confidence interval for mean difference
        se = std_diff / np.sqrt(n_pairs)
        ci_lower = mean_diff - stats.t.ppf(1 - alpha/2, n_pairs - 1) * se
        ci_upper = mean_diff + stats.t.ppf(1 - alpha/2, n_pairs - 1) * se
        
        # Calculate power
        if std_diff > 0:
            power = ttest_power(cohens_d, n_pairs, alpha, alternative='two-sided')
        else:
            power = alpha
        
        return {
            "test_name": "Paired t-Test",
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "mean_difference": float(mean_diff),
            "std_difference": float(std_diff),
            "cohens_d": float(cohens_d),
            "confidence_interval": {
                "lower": float(ci_lower),
                "upper": float(ci_upper),
                "level": 1 - alpha
            },
            "n_pairs": n_pairs,
            "df": n_pairs - 1,
            "power": float(power),
            "interpretation": self._interpret_results(p_value, cohens_d, mean_diff)
        }
    
    def calculate_power(
        self,
        n_pairs: int,
        effect_size: float,
        alpha: float = 0.05,
        alternative: str = 'two-sided'
    ) -> float:
        """Calculate statistical power for paired t-test."""
        return ttest_power(effect_size, n_pairs, alpha, alternative=alternative)
    
    def calculate_sample_size(
        self,
        effect_size: float,
        power: float = 0.8,
        alpha: float = 0.05,
        alternative: str = 'two-sided'
    ) -> int:
        """Calculate required sample size for paired t-test."""
        from statsmodels.stats.power import tt_solve_power
        n = tt_solve_power(effect_size=effect_size, power=power, alpha=alpha,
                          alternative=alternative)
        return int(np.ceil(n))
    
    def _interpret_results(self, p_value: float, cohens_d: float, mean_diff: float) -> str:
        """Interpret paired t-test results."""
        sig = "significant" if p_value < 0.05 else "not significant"
        
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            magnitude = "negligible"
        elif abs_d < 0.5:
            magnitude = "small"
        elif abs_d < 0.8:
            magnitude = "medium"
        else:
            magnitude = "large"
        
        direction = "increase" if mean_diff > 0 else "decrease"
        
        return f"Paired differences are {sig} (p={p_value:.4f}) with {magnitude} effect (d={cohens_d:.3f}), mean {direction} of {abs(mean_diff):.3f}"


class WelchTTest:
    """
    Welch's t-test for unequal variances.
    
    More robust alternative to Student's t-test when variances differ.
    """
    
    def calculate(
        self,
        group1: Optional[np.ndarray] = None,
        group2: Optional[np.ndarray] = None,
        n1: Optional[int] = None,
        n2: Optional[int] = None,
        mean1: Optional[float] = None,
        mean2: Optional[float] = None,
        std1: Optional[float] = None,
        std2: Optional[float] = None,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Calculate Welch's t-test.
        
        Args:
            group1, group2: Sample data
            n1, n2: Sample sizes
            mean1, mean2: Sample means
            std1, std2: Sample standard deviations
            alpha: Significance level
            
        Returns:
            Dictionary with test results
        """
        # Handle different input formats
        if group1 is not None and group2 is not None:
            group1 = np.array(group1)
            group2 = np.array(group2)
            n1, n2 = len(group1), len(group2)
            mean1, mean2 = np.mean(group1), np.mean(group2)
            std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
            
            # Perform Welch's t-test
            t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
        
        elif all(x is not None for x in [n1, n2, mean1, mean2, std1, std2]):
            # Calculate from summary statistics
            var1, var2 = std1**2, std2**2
            se = np.sqrt(var1/n1 + var2/n2)
            t_stat = (mean2 - mean1) / se if se > 0 else 0
            
            # Welch-Satterthwaite degrees of freedom
            df = ((var1/n1 + var2/n2)**2) / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
            
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        
        else:
            return {"error": "Insufficient data provided"}
        
        # Calculate effect size (Cohen's d with pooled SD)
        pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
        cohens_d = (mean2 - mean1) / pooled_std if pooled_std > 0 else 0
        
        # Glass's delta (uses control group SD)
        glass_delta = (mean2 - mean1) / std1 if std1 > 0 else 0
        
        # Confidence interval for mean difference
        mean_diff = mean2 - mean1
        se = np.sqrt(std1**2/n1 + std2**2/n2)
        
        # Welch-Satterthwaite df
        df = ((std1**2/n1 + std2**2/n2)**2) / \
             ((std1**2/n1)**2/(n1-1) + (std2**2/n2)**2/(n2-1))
        
        ci_lower = mean_diff - stats.t.ppf(1 - alpha/2, df) * se
        ci_upper = mean_diff + stats.t.ppf(1 - alpha/2, df) * se
        
        # Calculate power (approximate)
        nc = cohens_d * np.sqrt(n1*n2/(n1+n2))
        power = 1 - stats.nct.cdf(stats.t.ppf(1-alpha/2, df), df, nc) + \
                stats.nct.cdf(-stats.t.ppf(1-alpha/2, df), df, nc)
        
        # Levene's test for equality of variances
        if group1 is not None and group2 is not None:
            levene_stat, levene_p = stats.levene(group1, group2)
        else:
            levene_stat, levene_p = None, None
        
        return {
            "test_name": "Welch's t-Test",
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "mean_difference": float(mean_diff),
            "cohens_d": float(cohens_d),
            "glass_delta": float(glass_delta),
            "confidence_interval": {
                "lower": float(ci_lower),
                "upper": float(ci_upper),
                "level": 1 - alpha
            },
            "df": float(df),
            "n1": n1,
            "n2": n2,
            "variance_ratio": float(std2**2 / std1**2) if std1 > 0 else None,
            "levene_test": {
                "statistic": float(levene_stat) if levene_stat else None,
                "p_value": float(levene_p) if levene_p else None
            },
            "power": float(power),
            "interpretation": self._interpret_results(p_value, cohens_d, levene_p)
        }
    
    def _interpret_results(self, p_value: float, cohens_d: float, levene_p: Optional[float]) -> str:
        """Interpret Welch's t-test results."""
        sig = "significant" if p_value < 0.05 else "not significant"
        
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            magnitude = "negligible"
        elif abs_d < 0.5:
            magnitude = "small"
        elif abs_d < 0.8:
            magnitude = "medium"
        else:
            magnitude = "large"
        
        variance_comment = ""
        if levene_p is not None:
            if levene_p < 0.05:
                variance_comment = " (unequal variances confirmed, Welch's test appropriate)"
            else:
                variance_comment = " (equal variances, Student's t-test could be used)"
        
        return f"Group differences are {sig} (p={p_value:.4f}) with {magnitude} effect (d={cohens_d:.3f}){variance_comment}"


class LinearRegression:
    """
    Simple and multiple linear regression analysis.
    """
    
    def calculate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        alpha: float = 0.05,
        include_intercept: bool = True
    ) -> Dict[str, Any]:
        """
        Perform linear regression analysis.
        
        Args:
            X: Independent variable(s) - can be 1D or 2D array
            y: Dependent variable
            feature_names: Names for features
            alpha: Significance level
            include_intercept: Whether to include intercept term
            
        Returns:
            Dictionary with regression results
        """
        # Ensure arrays
        X = np.array(X)
        y = np.array(y)
        
        # Handle 1D case (simple regression)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        # Add intercept if requested
        if include_intercept:
            X = sm.add_constant(X)
            if feature_names:
                feature_names = ['Intercept'] + feature_names
            else:
                feature_names = ['Intercept'] + [f'X{i+1}' for i in range(n_features)]
        else:
            if not feature_names:
                feature_names = [f'X{i+1}' for i in range(n_features)]
        
        # Fit model
        model = sm.OLS(y, X)
        results = model.fit()
        
        # Extract key statistics
        coefficients = results.params
        std_errors = results.bse
        t_values = results.tvalues
        p_values = results.pvalues
        conf_int = results.conf_int(alpha)
        
        # Model statistics
        r_squared = results.rsquared
        adj_r_squared = results.rsquared_adj
        f_statistic = results.fvalue
        f_p_value = results.f_pvalue
        aic = results.aic
        bic = results.bic
        
        # Residual analysis
        residuals = results.resid
        mse = np.mean(residuals**2)
        rmse = np.sqrt(mse)
        
        # Durbin-Watson test for autocorrelation
        from statsmodels.stats.stattools import durbin_watson
        dw_stat = durbin_watson(residuals)
        
        # VIF for multicollinearity (if multiple predictors)
        vif_values = {}
        if n_features > 1 and include_intercept:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            for i in range(1, X.shape[1]):  # Skip intercept
                vif = variance_inflation_factor(X, i)
                vif_values[feature_names[i]] = float(vif)
        
        # Prepare coefficient details
        coef_details = []
        for i, name in enumerate(feature_names):
            coef_details.append({
                'name': name,
                'coefficient': float(coefficients[i]),
                'std_error': float(std_errors[i]),
                't_value': float(t_values[i]),
                'p_value': float(p_values[i]),
                'ci_lower': float(conf_int[i, 0]),
                'ci_upper': float(conf_int[i, 1])
            })
        
        return {
            "test_name": "Linear Regression",
            "n_samples": n_samples,
            "n_features": n_features,
            "coefficients": coef_details,
            "r_squared": float(r_squared),
            "adj_r_squared": float(adj_r_squared),
            "f_statistic": float(f_statistic),
            "f_p_value": float(f_p_value),
            "aic": float(aic),
            "bic": float(bic),
            "rmse": float(rmse),
            "durbin_watson": float(dw_stat),
            "vif": vif_values,
            "interpretation": self._interpret_results(r_squared, f_p_value, coef_details)
        }
    
    def predict(
        self,
        model_params: Dict[str, Any],
        X_new: np.ndarray,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Make predictions with fitted model.
        
        Args:
            model_params: Parameters from fitted model
            X_new: New data for prediction
            confidence_level: Confidence level for intervals
            
        Returns:
            Predictions with confidence and prediction intervals
        """
        X_new = np.array(X_new)
        if X_new.ndim == 1:
            X_new = X_new.reshape(-1, 1)
        
        # Add intercept if model has one
        if model_params['coefficients'][0]['name'] == 'Intercept':
            X_new = sm.add_constant(X_new)
        
        # Calculate predictions
        coeffs = [c['coefficient'] for c in model_params['coefficients']]
        y_pred = X_new @ coeffs
        
        # Approximate confidence intervals
        # (simplified - actual implementation would need full model)
        rmse = model_params['rmse']
        n = model_params['n_samples']
        
        # Standard error of prediction
        se_pred = rmse * np.sqrt(1 + 1/n)
        
        # Confidence intervals
        alpha = 1 - confidence_level
        t_crit = stats.t.ppf(1 - alpha/2, n - len(coeffs))
        
        ci_lower = y_pred - t_crit * se_pred
        ci_upper = y_pred + t_crit * se_pred
        
        return {
            "predictions": y_pred.tolist(),
            "confidence_interval": {
                "lower": ci_lower.tolist(),
                "upper": ci_upper.tolist(),
                "level": confidence_level
            }
        }
    
    def _interpret_results(
        self,
        r_squared: float,
        f_p_value: float,
        coef_details: List[Dict]
    ) -> str:
        """Interpret regression results."""
        # Model significance
        model_sig = "significant" if f_p_value < 0.05 else "not significant"
        
        # R-squared interpretation
        if r_squared < 0.3:
            fit_quality = "weak"
        elif r_squared < 0.5:
            fit_quality = "moderate"
        elif r_squared < 0.7:
            fit_quality = "good"
        else:
            fit_quality = "strong"
        
        # Count significant predictors
        sig_predictors = sum(1 for c in coef_details 
                           if c['p_value'] < 0.05 and c['name'] != 'Intercept')
        
        return f"Model is {model_sig} (F p={f_p_value:.4f}) with {fit_quality} fit (R²={r_squared:.3f}). {sig_predictors} significant predictor(s)."


class LogisticRegressionEnhanced:
    """
    Enhanced logistic regression with additional diagnostics.
    """
    
    def calculate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Perform logistic regression analysis.
        
        Args:
            X: Independent variables
            y: Binary dependent variable (0/1)
            feature_names: Names for features
            alpha: Significance level
            
        Returns:
            Dictionary with regression results
        """
        # Ensure arrays
        X = np.array(X)
        y = np.array(y)
        
        # Handle 1D case
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        # Add intercept
        X = sm.add_constant(X)
        if feature_names:
            feature_names = ['Intercept'] + feature_names
        else:
            feature_names = ['Intercept'] + [f'X{i+1}' for i in range(n_features)]
        
        # Fit model
        model = sm.Logit(y, X)
        results = model.fit(disp=0)
        
        # Extract statistics
        coefficients = results.params
        std_errors = results.bse
        z_values = results.tvalues
        p_values = results.pvalues
        conf_int = results.conf_int(alpha)
        
        # Odds ratios
        odds_ratios = np.exp(coefficients)
        or_conf_int = np.exp(conf_int)
        
        # Model statistics
        log_likelihood = results.llf
        aic = results.aic
        bic = results.bic
        pseudo_r2 = results.prsquared
        
        # Predictions and classification metrics
        y_pred_prob = results.predict(X)
        y_pred = (y_pred_prob >= 0.5).astype(int)
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        # ROC AUC
        from sklearn.metrics import roc_auc_score, roc_curve
        auc = roc_auc_score(y, y_pred_prob)
        fpr, tpr, thresholds = roc_curve(y, y_pred_prob)
        
        # Hosmer-Lemeshow test
        hl_stat, hl_p = self._hosmer_lemeshow_test(y, y_pred_prob)
        
        # Prepare coefficient details
        coef_details = []
        for i, name in enumerate(feature_names):
            coef_details.append({
                'name': name,
                'coefficient': float(coefficients[i]),
                'std_error': float(std_errors[i]),
                'z_value': float(z_values[i]),
                'p_value': float(p_values[i]),
                'odds_ratio': float(odds_ratios[i]),
                'or_ci_lower': float(or_conf_int[i, 0]),
                'or_ci_upper': float(or_conf_int[i, 1])
            })
        
        return {
            "test_name": "Logistic Regression",
            "n_samples": n_samples,
            "n_features": n_features,
            "coefficients": coef_details,
            "log_likelihood": float(log_likelihood),
            "aic": float(aic),
            "bic": float(bic),
            "pseudo_r2": float(pseudo_r2),
            "classification_metrics": {
                "accuracy": float(accuracy),
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
                "ppv": float(ppv),
                "npv": float(npv),
                "auc": float(auc)
            },
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp)
            },
            "hosmer_lemeshow": {
                "statistic": float(hl_stat),
                "p_value": float(hl_p)
            },
            "interpretation": self._interpret_results(pseudo_r2, auc, coef_details)
        }
    
    def _hosmer_lemeshow_test(self, y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> Tuple[float, float]:
        """Perform Hosmer-Lemeshow goodness of fit test."""
        # Sort by predicted probabilities
        order = np.argsort(y_pred)
        y_true_sorted = y_true[order]
        y_pred_sorted = y_pred[order]
        
        # Create bins
        bin_size = len(y_true) // n_bins
        observed = []
        expected = []
        
        for i in range(n_bins):
            start = i * bin_size
            end = start + bin_size if i < n_bins - 1 else len(y_true)
            
            obs = np.sum(y_true_sorted[start:end])
            exp = np.sum(y_pred_sorted[start:end])
            
            observed.append(obs)
            expected.append(exp)
        
        # Calculate statistic
        observed = np.array(observed)
        expected = np.array(expected)
        
        # Avoid division by zero
        expected = np.maximum(expected, 1e-10)
        
        hl_stat = np.sum((observed - expected)**2 / expected)
        hl_p = 1 - stats.chi2.cdf(hl_stat, n_bins - 2)
        
        return hl_stat, hl_p
    
    def _interpret_results(self, pseudo_r2: float, auc: float, coef_details: List[Dict]) -> str:
        """Interpret logistic regression results."""
        # Model fit
        if pseudo_r2 < 0.2:
            fit = "poor"
        elif pseudo_r2 < 0.4:
            fit = "fair"
        elif pseudo_r2 < 0.6:
            fit = "good"
        else:
            fit = "excellent"
        
        # Discrimination
        if auc < 0.6:
            disc = "poor"
        elif auc < 0.7:
            disc = "fair"
        elif auc < 0.8:
            disc = "good"
        elif auc < 0.9:
            disc = "very good"
        else:
            disc = "excellent"
        
        # Significant predictors
        sig_predictors = sum(1 for c in coef_details 
                           if c['p_value'] < 0.05 and c['name'] != 'Intercept')
        
        return f"Model shows {fit} fit (Pseudo R²={pseudo_r2:.3f}) with {disc} discrimination (AUC={auc:.3f}). {sig_predictors} significant predictor(s)."