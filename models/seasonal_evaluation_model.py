"""
Advanced evaluation module for seasonal time series models.
Provides specialized metrics and statistical tests for seasonal models.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Union, Optional, Tuple, Any
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

# Setup logging
logger = logging.getLogger(__name__)


def calculate_seasonal_decomposition_metrics(
    y_true: pd.Series, 
    y_pred: pd.Series,
    date_series: pd.Series,
    period: int = 12
) -> Dict[str, Any]:
    """
    Analyze model residuals for remaining seasonal patterns.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        date_series: Dates corresponding to the values
        period: Seasonal period for decomposition
        
    Returns:
        Dictionary with metrics of seasonal decomposition
    """
    # Ensure Series have the same index
    if len(y_true) != len(y_pred) or len(y_true) != len(date_series):
        raise ValueError("All input series must have the same length")
    
    # Calculate residuals
    residuals = y_true - y_pred
    
    # Create Series with datetime index for seasonal decomposition
    residual_series = pd.Series(residuals.values, index=pd.to_datetime(date_series))
    
    # Check if we have enough data for decomposition
    if len(residual_series) < period * 2:
        logger.warning(f"Not enough data for seasonal decomposition (need at least {period * 2} points)")
        return {
            'error': f"Not enough data for decomposition (need at least {period * 2} points)"
        }
    
    # Handle any NaN values by interpolation
    if np.isnan(residual_series).any():
        residual_series = residual_series.interpolate()
    
    try:
        # Perform seasonal decomposition
        decomposition = seasonal_decompose(
            residual_series, 
            model='additive', 
            period=period, 
            extrapolate_trend='freq'
        )
        
        # Extract components
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid
        
        # Calculate variance of each component
        var_total = np.var(residual_series)
        var_trend = np.var(trend.dropna())
        var_seasonal = np.var(seasonal.dropna())
        var_residual = np.var(residual.dropna())
        
        # Calculate component strengths
        # Formula from Hyndman's "Forecasting: Principles and Practice"
        trend_strength = max(0, 1 - var_residual / (var_trend + var_residual))
        seasonal_strength = max(0, 1 - var_residual / (var_seasonal + var_residual))
        
        # Calculate seasonal contribution to overall variance
        seasonal_contribution = var_seasonal / var_total if var_total > 0 else 0
        
        # Perform autocorrelation test on residuals
        residual_autocorr = acf(residual.dropna(), nlags=period, fft=True)
        
        # Test for significant autocorrelation at seasonal lags
        ljung_box = acorr_ljungbox(
            residual.dropna(), 
            lags=[period], 
            boxpierce=True, 
            return_df=True
        )
        
        lb_pvalue = ljung_box['lb_pvalue'].iloc[0]
        bp_pvalue = ljung_box['bp_pvalue'].iloc[0]
        
        # Prepare result
        result = {
            'variance': {
                'total': var_total,
                'trend': var_trend,
                'seasonal': var_seasonal,
                'residual': var_residual
            },
            'strength': {
                'trend': trend_strength,
                'seasonal': seasonal_strength
            },
            'seasonal_contribution': seasonal_contribution,
            'autocorrelation': {
                'seasonal_lag': residual_autocorr[period] if period < len(residual_autocorr) else None,
                'ljung_box_pvalue': lb_pvalue,
                'box_pierce_pvalue': bp_pvalue
            },
            'components': {
                'trend': trend,
                'seasonal': seasonal,
                'residual': residual
            }
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error in seasonal decomposition: {e}")
        return {'error': str(e)}


def calculate_seasonally_adjusted_metrics(
    y_true: pd.Series, 
    y_pred: pd.Series,
    seasonal_components: pd.Series,
    date_series: pd.Series
) -> Dict[str, float]:
    """
    Evaluate metrics on deseasonalized data.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        seasonal_components: Seasonal components for the time series
        date_series: Dates corresponding to the values
        
    Returns:
        Dictionary with metrics of deseasonalized predictions
    """
    # Ensure Series have the same index
    if len(y_true) != len(y_pred) or len(y_true) != len(date_series):
        raise ValueError("All input series must have the same length")
    
    # Convert to pandas Series with datetime index
    y_true = pd.Series(y_true.values, index=pd.to_datetime(date_series))
    y_pred = pd.Series(y_pred.values, index=pd.to_datetime(date_series))
    
    if len(seasonal_components) != len(y_true):
        raise ValueError("Seasonal components must have the same length as y_true and y_pred")
    
    # Remove seasonal components
    y_true_adj = y_true - seasonal_components
    y_pred_adj = y_pred - seasonal_components
    
    # Calculate metrics on adjusted data
    errors_adj = y_true_adj - y_pred_adj
    abs_errors_adj = np.abs(errors_adj)
    squared_errors_adj = errors_adj ** 2
    
    # Calculate original errors for comparison
    errors = y_true - y_pred
    abs_errors = np.abs(errors)
    squared_errors = errors ** 2
    
    # Calculate metrics
    metrics = {
        'rmse_adj': np.sqrt(np.mean(squared_errors_adj)),
        'mae_adj': np.mean(abs_errors_adj),
        'mean_error_adj': np.mean(errors_adj),
        'median_ae_adj': np.median(abs_errors_adj),
        'rmse_original': np.sqrt(np.mean(squared_errors)),
        'mae_original': np.mean(abs_errors),
        'mean_error_original': np.mean(errors),
        'median_ae_original': np.median(abs_errors)
    }
    
    # Calculate R² for adjusted and original
    ss_total_adj = np.sum((y_true_adj - np.mean(y_true_adj)) ** 2)
    ss_residual_adj = np.sum(squared_errors_adj)
    
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum(squared_errors)
    
    metrics['r_squared_adj'] = 1 - (ss_residual_adj / ss_total_adj) if ss_total_adj > 0 else 0
    metrics['r_squared_original'] = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
    
    # Calculate improvement percentages
    metrics['rmse_improvement'] = (metrics['rmse_original'] - metrics['rmse_adj']) / metrics['rmse_original'] * 100
    metrics['mae_improvement'] = (metrics['mae_original'] - metrics['mae_adj']) / metrics['mae_original'] * 100
    
    logger.info("Calculated seasonally adjusted metrics")
    return metrics


def test_seasonality_significance(model: Any) -> Dict[str, Any]:
    """
    Test statistical significance of seasonal components in a model.
    
    Args:
        model: Fitted model object with accessible coefficients
        
    Returns:
        Dictionary with significance test results
    """
    result = {
        'significant_components': [],
        'insignificant_components': [],
        'overall_significance': None,
        'joint_test': None
    }
    
    try:
        # Check if model has a statsmodels summary method
        model_summary = getattr(getattr(model, 'model', None), 'summary', None)
        
        if model_summary is not None and callable(model_summary):
            # Get model summary and extract p-values
            summary = model_summary()
            
            # Check if we can access the p-values
            p_values = getattr(getattr(model, 'model', None), 'pvalues', None)
            
            if p_values is not None:
                # Identify seasonal components based on variable names
                seasonal_vars = []
                
                # First approach: Check variable names
                for var_name in p_values.index:
                    if ('month_' in var_name or 'quarter_' in var_name or 
                        'sin_' in var_name or 'cos_' in var_name or 
                        'Q' in var_name and var_name != 'const'):
                        seasonal_vars.append(var_name)
                
                # If we found seasonal variables
                if seasonal_vars:
                    # Check individual significance
                    for var in seasonal_vars:
                        p_val = p_values.get(var, 1.0)
                        
                        if p_val < 0.05:
                            result['significant_components'].append({
                                'variable': var,
                                'p_value': p_val,
                                'coefficient': getattr(model, 'model', None).params.get(var, np.nan)
                            })
                        else:
                            result['insignificant_components'].append({
                                'variable': var,
                                'p_value': p_val,
                                'coefficient': getattr(model, 'model', None).params.get(var, np.nan)
                            })
                    
                    # Calculate overall significance based on proportion
                    sig_count = len(result['significant_components'])
                    total_count = len(seasonal_vars)
                    
                    if total_count > 0:
                        sig_proportion = sig_count / total_count
                        
                        if sig_proportion >= 0.8:
                            result['overall_significance'] = 'High'
                        elif sig_proportion >= 0.5:
                            result['overall_significance'] = 'Medium'
                        else:
                            result['overall_significance'] = 'Low'
                    
                    # Try to perform joint test if model has a method
                    f_test = getattr(getattr(model, 'model', None), 'f_test', None)
                    
                    if f_test is not None and callable(f_test):
                        try:
                            # Create restriction matrix for seasonal variables
                            seasonal_indices = [list(p_values.index).index(var) for var in seasonal_vars]
                            
                            if seasonal_indices:
                                k = len(p_values)
                                r = len(seasonal_indices)
                                R = np.zeros((r, k))
                                
                                for i, idx in enumerate(seasonal_indices):
                                    R[i, idx] = 1
                                
                                # Test restriction that all seasonal coefficients are zero
                                f_result = f_test(R)
                                
                                result['joint_test'] = {
                                    'f_value': f_result.fvalue,
                                    'p_value': f_result.pvalue,
                                    'df_num': f_result.df_num,
                                    'df_denom': f_result.df_denom,
                                    'significant': f_result.pvalue < 0.05
                                }
                        except Exception as e:
                            logger.warning(f"Error in joint F-test: {e}")
        
        # If we couldn't get the p-values from the model, try from seasonal_components attribute
        if not result['significant_components'] and not result['insignificant_components']:
            seasonal_components = getattr(model, 'seasonal_components', None)
            
            if seasonal_components is not None:
                # If we have seasonal components data, use it to assess significance
                # based on magnitude relative to overall prediction
                
                # For dummy variable or Fourier models, we can check if any components
                # are large relative to the mean prediction
                seasonal_values = np.array(list(seasonal_components.values())[0]) if seasonal_components else np.array([])
                
                if len(seasonal_values) > 0:
                    # Calculate mean absolute seasonal component
                    mean_abs_seasonal = np.mean(np.abs(seasonal_values))
                    
                    # Compare to overall mean of true values (if available)
                    if hasattr(model, 'train_mean_'):
                        train_mean = model.train_mean_
                    else:
                        # Approximate using y values if passed to this function
                        train_mean = np.mean(getattr(model, 'y_', 0))
                    
                    if train_mean > 0:
                        seasonal_ratio = mean_abs_seasonal / train_mean
                        
                        if seasonal_ratio >= 0.2:
                            result['overall_significance'] = 'High'
                        elif seasonal_ratio >= 0.1:
                            result['overall_significance'] = 'Medium'
                        else:
                            result['overall_significance'] = 'Low'
                            
                        result['mean_abs_seasonal'] = mean_abs_seasonal
                        result['train_mean'] = train_mean
                        result['seasonal_ratio'] = seasonal_ratio
    
    except Exception as e:
        logger.error(f"Error in seasonality significance test: {e}")
        result['error'] = str(e)
    
    return result


def evaluate_seasonal_component_stability(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    date_col: str = 'date',
    window_size: float = 0.5,
    step_size: float = 0.1
) -> Dict[str, Any]:
    """
    Evaluate stability of seasonal patterns over time using rolling window analysis.
    
    Args:
        model: Model class (not instance) to fit on windows
        X: Feature matrix
        y: Target variable
        date_col: Name of date column
        window_size: Size of window as proportion of data
        step_size: Step size as proportion of data
        
    Returns:
        Dictionary with stability metrics
    """
    # Ensure X is sorted by date
    if date_col in X.columns:
        X = X.sort_values(date_col).reset_index(drop=True)
        y = y.loc[X.index]  # Reindex y to match X
    
    n = len(X)
    window_length = int(n * window_size)
    step_length = int(n * step_size)
    
    if window_length < 10:
        return {'error': 'Window size too small'}
    
    # Create windows
    windows = []
    for start in range(0, n - window_length + 1, step_length):
        end = start + window_length
        windows.append((start, end))
    
    # If we have fewer than 2 windows, adjust step size
    if len(windows) < 2:
        step_length = max(1, window_length // 3)
        windows = []
        for start in range(0, n - window_length + 1, step_length):
            end = start + window_length
            windows.append((start, end))
    
    # Create model instance for each window
    window_results = []
    seasonal_components = []
    
    for i, (start, end) in enumerate(windows):
        X_window = X.iloc[start:end].copy()
        y_window = y.iloc[start:end].copy()
        
        try:
            # Create model instance of the same class as the provided model
            window_model = model.__class__()
            
            # Set same parameters as the original model
            if hasattr(model, 'get_params'):
                params = model.get_params()
                window_model.set_params(**params)
            
            # Fit model on window
            window_model.fit(X_window, y_window)
            
            # Extract seasonal components
            components = window_model.extract_seasonal_components() if hasattr(window_model, 'extract_seasonal_components') else None
            
            if components and 'seasonal' in components:
                seasonal_components.append(components['seasonal'])
                
                # Append result
                window_results.append({
                    'window': i,
                    'start': start,
                    'end': end,
                    'start_date': X_window[date_col].min() if date_col in X_window else None,
                    'end_date': X_window[date_col].max() if date_col in X_window else None,
                    'components': components['seasonal'],
                    'n_samples': len(X_window)
                })
            
        except Exception as e:
            logger.warning(f"Error fitting model on window {i}: {e}")
    
    # If we couldn't extract components from any window
    if not seasonal_components:
        return {'error': 'No seasonal components could be extracted from windows'}
    
    # Calculate stability metrics
    stability_metrics = {}
    
    # Convert seasonal components to numpy arrays for easier math
    seasonal_arrays = np.array(seasonal_components)
    
    # Calculate correlation matrix between all pairs of windows
    n_windows = seasonal_arrays.shape[0]
    correlation_matrix = np.zeros((n_windows, n_windows))
    
    for i in range(n_windows):
        for j in range(i+1, n_windows):
            # Calculate correlation between seasonal patterns from different windows
            corr = np.corrcoef(seasonal_arrays[i], seasonal_arrays[j])[0, 1]
            correlation_matrix[i, j] = corr
            correlation_matrix[j, i] = corr  # Matrix is symmetric
    
    # Calculate mean correlation (excluding diagonal)
    mean_correlation = np.sum(correlation_matrix) / (n_windows * (n_windows - 1))
    
    # Calculate standard deviation of seasonal components
    std_by_period = np.std(seasonal_arrays, axis=0)
    mean_std = np.mean(std_by_period)
    
    # Calculate coefficient of variation (relative variability)
    mean_by_period = np.mean(seasonal_arrays, axis=0)
    cv_by_period = np.zeros_like(mean_by_period)
    
    # Avoid division by zero
    for i, mean_val in enumerate(mean_by_period):
        if abs(mean_val) > 1e-10:
            cv_by_period[i] = std_by_period[i] / abs(mean_val)
        else:
            cv_by_period[i] = 0
    
    mean_cv = np.mean(cv_by_period)
    
    # Store metrics
    stability_metrics['mean_correlation'] = mean_correlation
    stability_metrics['mean_std'] = mean_std
    stability_metrics['mean_cv'] = mean_cv
    stability_metrics['std_by_period'] = std_by_period.tolist()
    stability_metrics['cv_by_period'] = cv_by_period.tolist()
    
    # Assess stability
    if mean_correlation >= 0.9:
        stability_metrics['stability'] = 'High'
    elif mean_correlation >= 0.7:
        stability_metrics['stability'] = 'Medium'
    else:
        stability_metrics['stability'] = 'Low'
    
    # Include window details
    stability_metrics['windows'] = window_results
    stability_metrics['n_windows'] = n_windows
    
    return stability_metrics


def create_evaluation_summary(model: Any, 
                             X_train: pd.DataFrame, y_train: pd.Series,
                             X_test: pd.DataFrame, y_test: pd.Series,
                             date_col: str = 'date',
                             seasonality_type: str = 'monthly') -> Dict[str, Any]:
    """
    Create comprehensive evaluation summary for a seasonal model.
    
    Args:
        model: Fitted model
        X_train: Training feature matrix
        y_train: Training target
        X_test: Test feature matrix
        y_test: Test target
        date_col: Name of date column
        seasonality_type: Type of seasonality
        
    Returns:
        Dictionary with comprehensive evaluation results
    """
    summary = {
        'model_info': {
            'type': type(model).__name__,
            'seasonality_type': seasonality_type,
            'train_size': len(X_train),
            'test_size': len(X_test)
        },
        'performance': {},
        'seasonality': {},
        'recommendations': []
    }
    
    # Generate predictions
    try:
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate overall performance metrics
        train_rmse = np.sqrt(np.mean((y_train - y_pred_train) ** 2))
        train_mae = np.mean(np.abs(y_train - y_pred_train))
        test_rmse = np.sqrt(np.mean((y_test - y_pred_test) ** 2))
        test_mae = np.mean(np.abs(y_test - y_pred_test))
        
        # Calculate R²
        train_ss_total = np.sum((y_train - np.mean(y_train)) ** 2)
        train_ss_residual = np.sum((y_train - y_pred_train) ** 2)
        train_r2 = 1 - (train_ss_residual / train_ss_total) if train_ss_total > 0 else 0
        
        test_ss_total = np.sum((y_test - np.mean(y_test)) ** 2)
        test_ss_residual = np.sum((y_test - y_pred_test) ** 2)
        test_r2 = 1 - (test_ss_residual / test_ss_total) if test_ss_total > 0 else 0
        
        # Store basic metrics
        summary['performance']['train'] = {
            'rmse': train_rmse,
            'mae': train_mae,
            'r_squared': train_r2
        }
        
        summary['performance']['test'] = {
            'rmse': test_rmse,
            'mae': test_mae,
            'r_squared': test_r2
        }
        
        # Calculate seasonal metrics if date column is available
        if date_col in X_test.columns:
            test_dates = X_test[date_col]
            
            # Get seasonal error metrics
            seasonal_metrics = calculate_seasonal_error_metrics(
                y_test, y_pred_test, test_dates, seasonality_type
            )
            summary['performance']['seasonal_metrics'] = seasonal_metrics
            
            # Test for remaining seasonality in residuals
            residuals = y_test - y_pred_test
            
            try:
                # Only proceed if we have enough data
                if len(residuals) >= 24:  # Need at least 2 cycles for monthly data
                    # Create Series with datetime index
                    residual_series = pd.Series(residuals.values, index=pd.to_datetime(test_dates))
                    
                    # Calculate autocorrelation at seasonal lags
                    if seasonality_type == 'monthly':
                        period = 12
                    elif seasonality_type == 'quarterly':
                        period = 4
                    else:
                        period = 12
                    
                    residual_acf = acf(residual_series.interpolate(), nlags=period+1, fft=True)
                    
                    # Check for significant autocorrelation at seasonal lag
                    seasonal_autocorr = residual_acf[period]
                    
                    summary['seasonality']['residual_autocorrelation'] = {
                        'lag': period,
                        'value': seasonal_autocorr,
                        'significant': abs(seasonal_autocorr) > 1.96 / np.sqrt(len(residual_series))
                    }
                    
                    # If there's significant autocorrelation, suggest improvements
                    if abs(seasonal_autocorr) > 1.96 / np.sqrt(len(residual_series)):
                        summary['recommendations'].append(
                            "Significant seasonality remains in the residuals. Consider using more complex seasonal patterns or a different model."
                        )
            except Exception as e:
                logger.warning(f"Error in residual analysis: {e}")
        
        # Test seasonality significance
        significance_results = test_seasonality_significance(model)
        summary['seasonality']['significance'] = significance_results
        
        # Add seasonal components if available
        if hasattr(model, 'extract_seasonal_components'):
            components = model.extract_seasonal_components()
            if components and 'seasonal' in components:
                summary['seasonality']['components'] = components['seasonal'].tolist()
                
                # Calculate metrics about the seasonal pattern
                seasonal_amplitude = np.max(components['seasonal']) - np.min(components['seasonal'])
                seasonal_mean = np.mean(components['seasonal'])
                seasonal_std = np.std(components['seasonal'])
                
                summary['seasonality']['metrics'] = {
                    'amplitude': seasonal_amplitude,
                    'mean': seasonal_mean,
                    'std': seasonal_std,
                    'coefficient_of_variation': seasonal_std / abs(seasonal_mean) if abs(seasonal_mean) > 1e-10 else 0
                }
        
        # Generate recommendations based on results
        generate_recommendations(summary)
        
    except Exception as e:
        logger.error(f"Error in evaluation summary: {e}")
        summary['error'] = str(e)
    
    return summary


def generate_recommendations(summary: Dict[str, Any]) -> None:
    """
    Generate recommendations based on evaluation results.
    Modifies the summary dictionary in place.
    
    Args:
        summary: Evaluation summary dictionary
    """
    recommendations = []
    
    # Check for overfitting
    if 'performance' in summary:
        train_metrics = summary['performance'].get('train', {})
        test_metrics = summary['performance'].get('test', {})
        
        train_r2 = train_metrics.get('r_squared', 0)
        test_r2 = test_metrics.get('r_squared', 0)
        
        if train_r2 > test_r2 + 0.2:
            recommendations.append(
                f"Model shows signs of overfitting (train R² = {train_r2:.3f}, test R² = {test_r2:.3f}). "
                "Consider simplifying the model or using regularization."
            )
        
        # Check if model is underfitting
        if test_r2 < 0.5:
            recommendations.append(
                f"Model has low explanatory power (test R² = {test_r2:.3f}). "
                "Consider adding more features or trying different model types."
            )
        
        # Check forecast bias
        if 'mae' in test_metrics:
            error_mean = test_metrics.get('mean_error', 0)
            mae = test_metrics.get('mae', 1)
            
            if abs(error_mean) > 0.3 * mae:
                bias_direction = "overestimating" if error_mean < 0 else "underestimating"
                recommendations.append(
                    f"Model shows bias ({bias_direction}). "
                    "Consider addressing systematic errors or adding more features."
                )
    
    # Check seasonality significance
    seasonality = summary.get('seasonality', {})
    significance = seasonality.get('significance', {})
    
    if significance.get('overall_significance') == 'Low':
        recommendations.append(
            "Seasonal components have low significance. Consider simplifying the model or using non-seasonal approaches."
        )
    
    # Check for residual autocorrelation
    residual_autocorr = seasonality.get('residual_autocorrelation', {})
    
    if residual_autocorr.get('significant', False):
        lag = residual_autocorr.get('lag', 'seasonal')
        recommendations.append(
            f"Significant autocorrelation at {lag} lag in residuals. "
            "Consider using autoregressive models or more complex seasonal patterns."
        )
    
    # Check seasonal variance ratio
    seasonal_metrics = summary.get('performance', {}).get('seasonal_metrics', {})
    if 'worst_period' in seasonal_metrics and 'best_period' in seasonal_metrics:
        worst_period = seasonal_metrics.get('worst_period', {})
        best_period = seasonal_metrics.get('best_period', {})
        
        if worst_period.get('rmse', 0) > 2 * best_period.get('rmse', 1):
            recommendations.append(
                f"Large performance disparity between best ({best_period.get('period_name', '')}) and worst ({worst_period.get('period_name', '')}) periods. "
                "Consider using different models for different seasons or adding more features to address problematic periods."
            )
    
    # Store recommendations in summary
    summary['recommendations'] = recommendations


def seasonal_forecast_errors_by_horizon(
    y_true: pd.Series,
    forecasts: Dict[str, pd.DataFrame],
    date_col: str = 'date',
    target_col: str = 'forecast',
    max_horizon: int = 12
) -> Dict[str, Any]:
    """
    Analyze how forecast errors change with forecast horizon, with focus on seasonal patterns.
    
    Args:
        y_true: Series of actual values with datetime index
        forecasts: Dictionary mapping model names to forecast DataFrames
        date_col: Name of date column in forecast DataFrames
        target_col: Name of forecast column in forecast DataFrames
        max_horizon: Maximum forecast horizon to analyze
        
    Returns:
        Dictionary with error metrics by horizon and model
    """
    # Ensure y_true has datetime index
    if not isinstance(y_true.index, pd.DatetimeIndex):
        raise ValueError("y_true must have a datetime index")
    
    # Initialize results
    results = {
        'by_horizon': {},
        'by_model': {},
        'by_season': {}
    }
    
    # Create dataframe for true values
    true_df = pd.DataFrame({'actual': y_true})
    true_df.index.name = date_col
    true_df = true_df.reset_index()
    
    # Extract month for seasonal analysis
    true_df['month'] = true_df[date_col].dt.month
    true_df['quarter'] = true_df[date_col].dt.quarter
    
    # Analyze each model's forecasts
    for model_name, forecast_df in forecasts.items():
        # Ensure date column is datetime
        forecast_df[date_col] = pd.to_datetime(forecast_df[date_col])
        
        # Get unique forecast start dates
        start_dates = forecast_df['forecast_start'].unique() if 'forecast_start' in forecast_df.columns else [None]
        
        # Initialize model metrics
        model_metrics = {
            'horizon_rmse': {},
            'horizon_mae': {},
            'season_rmse': {},
            'season_mae': {}
        }
        
        for start_date in start_dates:
            # Filter forecasts for this start date if needed
            if start_date is not None:
                model_forecasts = forecast_df[forecast_df['forecast_start'] == start_date].copy()
            else:
                model_forecasts = forecast_df.copy()
            
            # Skip if no forecasts with this start date
            if len(model_forecasts) == 0:
                continue
            
            # Calculate horizon for each forecast point
            if 'horizon' not in model_forecasts.columns:
                if 'forecast_start' in model_forecasts.columns:
                    # Calculate horizon based on difference from forecast start
                    for i, row in model_forecasts.iterrows():
                        forecast_date = row[date_col]
                        model_forecasts.loc[i, 'horizon'] = (forecast_date - start_date).days // 30 + 1
                else:
                    # Assume forecasts are in order from closest to furthest horizon
                    model_forecasts['horizon'] = range(1, len(model_forecasts) + 1)
            
            # Merge with true values on date
            merged = pd.merge(model_forecasts, true_df, on=date_col, how='inner')
            
            # Skip if no matching dates
            if len(merged) == 0:
                continue
            
            # Calculate errors
            merged['error'] = merged['actual'] - merged[target_col]
            merged['abs_error'] = np.abs(merged['error'])
            merged['squared_error'] = merged['error'] ** 2
            
            # Calculate errors by horizon
            for h in range(1, min(max_horizon + 1, merged['horizon'].max() + 1)):
                horizon_data = merged[merged['horizon'] == h]
                
                if len(horizon_data) > 0:
                    # Calculate RMSE and MAE for this horizon
                    horizon_rmse = np.sqrt(np.mean(horizon_data['squared_error']))
                    horizon_mae = np.mean(horizon_data['abs_error'])
                    
                    # Store in model metrics
                    if h not in model_metrics['horizon_rmse']:
                        model_metrics['horizon_rmse'][h] = []
                        model_metrics['horizon_mae'][h] = []
                    
                    model_metrics['horizon_rmse'][h].append(horizon_rmse)
                    model_metrics['horizon_mae'][h].append(horizon_mae)
            
            # Calculate errors by season (month or quarter)
            # For each month
            for month in range(1, 13):
                month_data = merged[merged['month'] == month]
                
                if len(month_data) > 0:
                    # Calculate RMSE and MAE for this month
                    month_rmse = np.sqrt(np.mean(month_data['squared_error']))
                    month_mae = np.mean(month_data['abs_error'])
                    
                    month_name = pd.Timestamp(2000, month, 1).strftime('%b')
                    
                    # Store in model metrics
                    if month_name not in model_metrics['season_rmse']:
                        model_metrics['season_rmse'][month_name] = []
                        model_metrics['season_mae'][month_name] = []
                    
                    model_metrics['season_rmse'][month_name].append(month_rmse)
                    model_metrics['season_mae'][month_name].append(month_mae)
            
            # For each quarter
            for quarter in range(1, 5):
                quarter_data = merged[merged['quarter'] == quarter]
                
                if len(quarter_data) > 0:
                    # Calculate RMSE and MAE for this quarter
                    quarter_rmse = np.sqrt(np.mean(quarter_data['squared_error']))
                    quarter_mae = np.mean(quarter_data['abs_error'])
                    
                    quarter_name = f"Q{quarter}"
                    
                    # Store in model metrics
                    if quarter_name not in model_metrics['season_rmse']:
                        model_metrics['season_rmse'][quarter_name] = []
                        model_metrics['season_mae'][quarter_name] = []
                    
                    model_metrics['season_rmse'][quarter_name].append(quarter_rmse)
                    model_metrics['season_mae'][quarter_name].append(quarter_mae)
        
        # Average metrics across all forecast start dates
        avg_horizon_rmse = {h: np.mean(values) for h, values in model_metrics['horizon_rmse'].items()}
        avg_horizon_mae = {h: np.mean(values) for h, values in model_metrics['horizon_mae'].items()}
        
        avg_season_rmse = {s: np.mean(values) for s, values in model_metrics['season_rmse'].items()}
        avg_season_mae = {s: np.mean(values) for s, values in model_metrics['season_mae'].items()}
        
        # Store in results
        results['by_model'][model_name] = {
            'horizon_rmse': avg_horizon_rmse,
            'horizon_mae': avg_horizon_mae,
            'season_rmse': avg_season_rmse,
            'season_mae': avg_season_mae
        }
        
        # Aggregate across horizons for overall metrics
        for h in avg_horizon_rmse:
            if h not in results['by_horizon']:
                results['by_horizon'][h] = {
                    'models': {},
                    'avg_rmse': 0,
                    'avg_mae': 0,
                    'count': 0
                }
            
            results['by_horizon'][h]['models'][model_name] = {
                'rmse': avg_horizon_rmse[h],
                'mae': avg_horizon_mae[h]
            }
            
            # Update averages
            results['by_horizon'][h]['avg_rmse'] += avg_horizon_rmse[h]
            results['by_horizon'][h]['avg_mae'] += avg_horizon_mae[h]
            results['by_horizon'][h]['count'] += 1
        
        # Aggregate across seasons
        for season in avg_season_rmse:
            if season not in results['by_season']:
                results['by_season'][season] = {
                    'models': {},
                    'avg_rmse': 0,
                    'avg_mae': 0,
                    'count': 0
                }
            
            results['by_season'][season]['models'][model_name] = {
                'rmse': avg_season_rmse[season],
                'mae': avg_season_mae[season]
            }
            
            # Update averages
            results['by_season'][season]['avg_rmse'] += avg_season_rmse[season]
            results['by_season'][season]['avg_mae'] += avg_season_mae[season]
            results['by_season'][season]['count'] += 1
    
    # Calculate final averages
    for h in results['by_horizon']:
        if results['by_horizon'][h]['count'] > 0:
            results['by_horizon'][h]['avg_rmse'] /= results['by_horizon'][h]['count']
            results['by_horizon'][h]['avg_mae'] /= results['by_horizon'][h]['count']
    
    for season in results['by_season']:
        if results['by_season'][season]['count'] > 0:
            results['by_season'][season]['avg_rmse'] /= results['by_season'][season]['count']
            results['by_season'][season]['avg_mae'] /= results['by_season'][season]['count']
    
    # Find best/worst horizons and seasons
    if results['by_horizon']:
        best_horizon = min(results['by_horizon'].items(), key=lambda x: x[1]['avg_rmse'])
        worst_horizon = max(results['by_horizon'].items(), key=lambda x: x[1]['avg_rmse'])
        
        results['summary'] = {
            'best_horizon': {
                'horizon': best_horizon[0],
                'avg_rmse': best_horizon[1]['avg_rmse'],
                'avg_mae': best_horizon[1]['avg_mae']
            },
            'worst_horizon': {
                'horizon': worst_horizon[0],
                'avg_rmse': worst_horizon[1]['avg_rmse'],
                'avg_mae': worst_horizon[1]['avg_mae']
            }
        }
    
    if results['by_season']:
        best_season = min(results['by_season'].items(), key=lambda x: x[1]['avg_rmse'])
        worst_season = max(results['by_season'].items(), key=lambda x: x[1]['avg_rmse'])
        
        if 'summary' not in results:
            results['summary'] = {}
        
        results['summary']['best_season'] = {
            'season': best_season[0],
            'avg_rmse': best_season[1]['avg_rmse'],
            'avg_mae': best_season[1]['avg_mae']
        }
        
        results['summary']['worst_season'] = {
            'season': worst_season[0],
            'avg_rmse': worst_season[1]['avg_rmse'],
            'avg_mae': worst_season[1]['avg_mae']
        }
    
    return results