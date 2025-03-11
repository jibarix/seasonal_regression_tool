"""
ARIMA diagnostics module.
Provides functions for diagnostic testing and visualization of ARIMA models.
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Union, Optional, Tuple, Any
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.tsa.stattools import adfuller, kpss
from scipy import stats

# Import from project modules
from arima_base import BaseARIMAModel

# Setup logging
logger = logging.getLogger(__name__)


def test_stationarity(series: pd.Series) -> Dict[str, Any]:
    """
    Perform stationarity tests on a time series.
    
    Args:
        series: Time series data
        
    Returns:
        Dictionary with test results
    """
    results = {}
    
    # Remove NaN values
    series = series.dropna()
    
    if len(series) < 10:
        logger.warning("Series too short for reliable stationarity tests")
        return {'error': 'Series too short'}
    
    # Augmented Dickey-Fuller test
    try:
        adf_result = adfuller(series)
        results['adf_test'] = {
            'statistic': adf_result[0],
            'p_value': adf_result[1],
            'critical_values': adf_result[4],
            'stationary': adf_result[1] < 0.05  # Stationary if p-value < 0.05
        }
    except Exception as e:
        logger.error(f"Error in ADF test: {e}")
        results['adf_test'] = {'error': str(e)}
    
    # KPSS test
    try:
        kpss_result = kpss(series)
        results['kpss_test'] = {
            'statistic': kpss_result[0],
            'p_value': kpss_result[1],
            'critical_values': kpss_result[3],
            'stationary': kpss_result[1] > 0.05  # Stationary if p-value > 0.05
        }
    except Exception as e:
        logger.error(f"Error in KPSS test: {e}")
        results['kpss_test'] = {'error': str(e)}
    
    # Overall stationarity assessment
    if 'adf_test' in results and 'kpss_test' in results:
        if 'error' not in results['adf_test'] and 'error' not in results['kpss_test']:
            adf_stationary = results['adf_test']['stationary']
            kpss_stationary = results['kpss_test']['stationary']
            
            if adf_stationary and kpss_stationary:
                results['conclusion'] = 'Stationary'
            elif not adf_stationary and not kpss_stationary:
                results['conclusion'] = 'Non-stationary'
            elif adf_stationary and not kpss_stationary:
                results['conclusion'] = 'Trend stationary'
            else:  # not adf_stationary and kpss_stationary
                results['conclusion'] = 'Difference stationary'
    
    return results


def ljung_box_test(residuals: pd.Series, lags: List[int] = [10, 15, 20, 30]) -> Dict[str, Any]:
    """
    Perform Ljung-Box test for autocorrelation in residuals.
    
    Args:
        residuals: Model residuals
        lags: List of lag values to test
        
    Returns:
        Dictionary with test results
    """
    try:
        lb_test = acorr_ljungbox(residuals, lags=lags)
        
        # Convert to more readable format
        results = {
            'lags': lags,
            'statistics': lb_test.iloc[:, 0].tolist(),
            'p_values': lb_test.iloc[:, 1].tolist(),
            'no_autocorrelation': [p > 0.05 for p in lb_test.iloc[:, 1]],
            'conclusion': 'No significant autocorrelation' 
                         if all(p > 0.05 for p in lb_test.iloc[:, 1]) 
                         else 'Significant autocorrelation detected'
        }
        
        return results
    
    except Exception as e:
        logger.error(f"Error in Ljung-Box test: {e}")
        return {'error': str(e)}


def test_normality(residuals: pd.Series) -> Dict[str, Any]:
    """
    Test for normality of residuals.
    
    Args:
        residuals: Model residuals
        
    Returns:
        Dictionary with test results
    """
    results = {}
    
    # Jarque-Bera test
    try:
        jb_test = stats.jarque_bera(residuals)
        results['jarque_bera'] = {
            'statistic': jb_test[0],
            'p_value': jb_test[1],
            'normal': jb_test[1] > 0.05  # Normal if p-value > 0.05
        }
    except Exception as e:
        logger.error(f"Error in Jarque-Bera test: {e}")
        results['jarque_bera'] = {'error': str(e)}
    
    # Shapiro-Wilk test
    try:
        # For large samples, use a subset
        if len(residuals) > 5000:
            logger.info("Using subset of residuals for Shapiro-Wilk test (n=5000)")
            test_data = residuals.sample(5000)
        else:
            test_data = residuals
            
        sw_test = stats.shapiro(test_data)
        results['shapiro_wilk'] = {
            'statistic': sw_test[0],
            'p_value': sw_test[1],
            'normal': sw_test[1] > 0.05  # Normal if p-value > 0.05
        }
    except Exception as e:
        logger.error(f"Error in Shapiro-Wilk test: {e}")
        results['shapiro_wilk'] = {'error': str(e)}
    
    # Overall normality assessment
    if 'jarque_bera' in results and 'shapiro_wilk' in results:
        if 'error' not in results['jarque_bera'] and 'error' not in results['shapiro_wilk']:
            jb_normal = results['jarque_bera']['normal']
            sw_normal = results['shapiro_wilk']['normal']
            
            if jb_normal and sw_normal:
                results['conclusion'] = 'Residuals appear normally distributed'
            elif not jb_normal and not sw_normal:
                results['conclusion'] = 'Residuals are not normally distributed'
            else:
                results['conclusion'] = 'Tests disagree on normality of residuals'
    
    return results


def test_heteroscedasticity(residuals: pd.Series) -> Dict[str, Any]:
    """
    Test for heteroscedasticity in residuals.
    
    Args:
        residuals: Model residuals
        
    Returns:
        Dictionary with test results
    """
    try:
        # Create a time index
        x = np.arange(len(residuals)).reshape(-1, 1)
        
        # Breusch-Pagan test
        bp_test = het_breuschpagan(residuals, x)
        
        results = {
            'statistic': bp_test[0],
            'p_value': bp_test[1],
            'f_statistic': bp_test[2],
            'f_p_value': bp_test[3],
            'homoscedastic': bp_test[1] > 0.05,  # Homoscedastic if p-value > 0.05
            'conclusion': 'Homoscedastic residuals' 
                         if bp_test[1] > 0.05 
                         else 'Heteroscedastic residuals detected'
        }
        
        return results
    
    except Exception as e:
        logger.error(f"Error in heteroscedasticity test: {e}")
        return {'error': str(e)}


def run_all_diagnostics(model: BaseARIMAModel) -> Dict[str, Any]:
    """
    Run all diagnostic tests on a fitted ARIMA model.
    
    Args:
        model: Fitted ARIMA model
        
    Returns:
        Dictionary with all test results
    """
    if not model.is_fitted or model.residuals is None:
        raise ValueError("Model must be fitted with residuals before diagnostics")
    
    # Get residuals
    residuals = model.residuals
    
    # Initialize results
    diagnostics = {
        'model_type': model.__class__.__name__,
        'model_order': model.order
    }
    
    # Add seasonal order if available
    if hasattr(model, 'seasonal_order'):
        diagnostics['seasonal_order'] = model.seasonal_order
    
    # Run tests
    logger.info("Running autocorrelation test")
    diagnostics['autocorrelation'] = ljung_box_test(residuals)
    
    logger.info("Running normality test")
    diagnostics['normality'] = test_normality(residuals)
    
    logger.info("Running heteroscedasticity test")
    diagnostics['heteroscedasticity'] = test_heteroscedasticity(residuals)
    
    # Add AIC and BIC
    if hasattr(model.model, 'aic'):
        diagnostics['aic'] = model.model.aic
    if hasattr(model.model, 'bic'):
        diagnostics['bic'] = model.model.bic
    
    # Overall assessment
    diagnostics['overall_issues'] = []
    
    if 'autocorrelation' in diagnostics and 'conclusion' in diagnostics['autocorrelation']:
        if diagnostics['autocorrelation']['conclusion'] != 'No significant autocorrelation':
            diagnostics['overall_issues'].append('Residual autocorrelation')
    
    if 'normality' in diagnostics and 'conclusion' in diagnostics['normality']:
        if diagnostics['normality']['conclusion'] != 'Residuals appear normally distributed':
            diagnostics['overall_issues'].append('Non-normal residuals')
    
    if 'heteroscedasticity' in diagnostics and 'conclusion' in diagnostics['heteroscedasticity']:
        if diagnostics['heteroscedasticity']['conclusion'] != 'Homoscedastic residuals':
            diagnostics['overall_issues'].append('Heteroscedastic residuals')
    
    diagnostics['recommendation'] = (
        "Model appears well-specified" if not diagnostics['overall_issues']
        else "Consider model adjustments to address: " + ", ".join(diagnostics['overall_issues'])
    )
    
    return diagnostics


def plot_residual_diagnostics(model: BaseARIMAModel, figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
    """
    Create comprehensive residual diagnostic plots.
    
    Args:
        model: Fitted ARIMA model
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if not model.is_fitted or model.residuals is None:
        raise ValueError("Model must be fitted with residuals before plotting")
    
    # Get residuals
    residuals = model.residuals
    
    # Create figure
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    
    # 1. Time series plot of residuals
    axes[0, 0].plot(residuals)
    axes[0, 0].set_title('Residuals Time Series')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Residual')
    axes[0, 0].axhline(y=0, color='r', linestyle='-')
    
    # 2. Histogram with normal curve
    axes[0, 1].hist(residuals, bins=30, density=True, alpha=0.6)
    
    # Add normal curve
    mu, std = residuals.mean(), residuals.std()
    x = np.linspace(mu - 3*std, mu + 3*std, 100)
    p = stats.norm.pdf(x, mu, std)
    axes[0, 1].plot(x, p, 'k', linewidth=2)
    
    axes[0, 1].set_title('Histogram of Residuals with Normal Curve')
    axes[0, 1].set_xlabel('Residual')
    axes[0, 1].set_ylabel('Density')
    
    # 3. Q-Q plot
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot')
    
    # 4. ACF of residuals
    plot_acf(residuals, lags=30, ax=axes[1, 1])
    axes[1, 1].set_title('Autocorrelation Function (ACF)')
    
    # 5. PACF of residuals
    plot_pacf(residuals, lags=30, ax=axes[2, 0])
    axes[2, 0].set_title('Partial Autocorrelation Function (PACF)')
    
    # 6. Residuals vs. fitted values
    if model.fitted_values is not None:
        axes[2, 1].scatter(model.fitted_values, residuals, alpha=0.5)
        axes[2, 1].set_title('Residuals vs. Fitted Values')
        axes[2, 1].set_xlabel('Fitted Value')
        axes[2, 1].set_ylabel('Residual')
        axes[2, 1].axhline(y=0, color='r', linestyle='-')
    else:
        axes[2, 1].set_title('Residuals vs. Fitted Values (Not Available)')
    
    plt.tight_layout()
    return fig


def plot_forecast_evaluation(model: BaseARIMAModel, test_data: pd.DataFrame, 
                            test_target: pd.Series, steps: Optional[int] = None,
                            figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Plot forecast evaluation against test data.
    
    Args:
        model: Fitted ARIMA model
        test_data: Test feature matrix
        test_target: Test target series
        steps: Number of steps to include (None for all)
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if not model.is_fitted:
        raise ValueError("Model must be fitted before forecast evaluation")
    
    # Generate predictions
    predictions = model.predict(test_data)
    
    # Limit to specified steps if provided
    if steps is not None and steps < len(predictions):
        predictions = predictions[:steps]
        test_target = test_target[:steps]
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    # Extract dates if available
    date_col = model.date_col
    if date_col in test_data.columns:
        dates = pd.to_datetime(test_data[date_col])
    else:
        dates = pd.RangeIndex(len(predictions))
    
    # Plot actual vs. predicted
    axes[0].plot(dates, test_target, 'b-', label='Actual')
    axes[0].plot(dates, predictions, 'r-', label='Predicted')
    axes[0].set_title('Actual vs. Predicted Values')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Value')
    axes[0].legend()
    
    # Format x-axis if dates are used
    if date_col in test_data.columns:
        fig.autofmt_xdate()
    
    # Plot prediction errors
    errors = test_target - predictions
    axes[1].plot(dates, errors, 'g-')
    axes[1].set_title('Prediction Errors')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Error')
    axes[1].axhline(y=0, color='r', linestyle='-')
    
    # Format x-axis if dates are used
    if date_col in test_data.columns:
        fig.autofmt_xdate()
    
    # Calculate error metrics
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(errors))
    
    # Add metrics to figure title
    fig.suptitle(f"Forecast Evaluation - RMSE: {rmse:.4f}, MAE: {mae:.4f}", 
                fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust for suptitle
    
    return fig


def plot_decomposition(model: BaseARIMAModel, figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
    """
    Plot time series decomposition (trend, seasonal, residual).
    
    Args:
        model: Fitted ARIMA model
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if not model.is_fitted or model.original_data is None:
        raise ValueError("Model must be fitted with original data before decomposition")
    
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    # Determine seasonal period
    period = 12  # Default for monthly data
    
    if hasattr(model, 'seasonal_order') and model.seasonal_order is not None:
        period = model.seasonal_order[3]  # Extract s from (P,D,Q,s)
    
    # Create a time series with the proper index
    if hasattr(model, 'date_index_') and model.date_index_ is not None:
        y = pd.Series(model.original_data.values, index=model.date_index_)
    else:
        y = pd.Series(model.original_data.values)
    
    # Perform decomposition
    try:
        decomposition = seasonal_decompose(y, model='additive', period=period)
        
        # Create figure
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
        
        # Plot original series
        axes[0].plot(decomposition.observed)
        axes[0].set_title('Original Series')
        axes[0].set_ylabel('Value')
        
        # Plot trend component
        axes[1].plot(decomposition.trend)
        axes[1].set_title('Trend Component')
        axes[1].set_ylabel('Value')
        
        # Plot seasonal component
        axes[2].plot(decomposition.seasonal)
        axes[2].set_title(f'Seasonal Component (Period={period})')
        axes[2].set_ylabel('Value')
        
        # Plot residual component
        axes[3].plot(decomposition.resid)
        axes[3].set_title('Residual Component')
        axes[3].set_ylabel('Value')
        axes[3].set_xlabel('Time')
        
        # Format x-axis
        fig.autofmt_xdate()
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        logger.error(f"Error in decomposition: {e}")
        # Create a simple figure with error message
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Decomposition failed: {str(e)}", 
               horizontalalignment='center', verticalalignment='center')
        ax.set_title("Decomposition Error")
        return fig


def plot_model_comparison(models: List[BaseARIMAModel], X: pd.DataFrame, y: pd.Series,
                         figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot comparison of multiple model forecasts.
    
    Args:
        models: List of fitted ARIMA models
        X: Feature matrix
        y: Target series
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if not all(model.is_fitted for model in models):
        raise ValueError("All models must be fitted before comparison")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract dates if available
    date_col = models[0].date_col  # Use date column from first model
    if date_col in X.columns:
        dates = pd.to_datetime(X[date_col])
    else:
        dates = pd.RangeIndex(len(y))
    
    # Plot actual data
    ax.plot(dates, y, 'k-', linewidth=2, label='Actual')
    
    # Plot predictions for each model
    colors = plt.cm.tab10.colors
    linestyles = ['-', '--', '-.', ':']
    
    for i, model in enumerate(models):
        predictions = model.predict(X)
        color = colors[i % len(colors)]
        linestyle = linestyles[i % len(linestyles)]
        
        # Create model label with AIC/BIC if available
        label = model.__class__.__name__
        
        if hasattr(model, 'order'):
            label += f" {model.order}"
        if hasattr(model, 'seasonal_order') and model.seasonal_order is not None:
            label += f"x{model.seasonal_order}"
            
        if hasattr(model.model, 'aic'):
            label += f" (AIC: {model.model.aic:.1f})"
        
        ax.plot(dates, predictions, color=color, linestyle=linestyle, 
               alpha=0.7, label=label)
    
    # Add labels and legend
    ax.set_title('Model Comparison: Actual vs. Predicted')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.legend(loc='best')
    
    # Format x-axis if dates are used
    if date_col in X.columns:
        fig.autofmt_xdate()
    
    # Calculate and display error metrics
    model_metrics = {}
    for model in models:
        predictions = model.predict(X)
        errors = y - predictions
        rmse = np.sqrt(np.mean(errors ** 2))
        model_name = model.__class__.__name__
        if hasattr(model, 'order'):
            model_name += f" {model.order}"
        model_metrics[model_name] = rmse
    
    # Add metrics table to plot
    metrics_text = "RMSE Values:\n" + "\n".join([f"{name}: {rmse:.4f}" 
                                               for name, rmse in model_metrics.items()])
    ax.text(0.02, 0.02, metrics_text, transform=ax.transAxes, 
           bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    return fig


def create_diagnostic_report(model: BaseARIMAModel, test_data: Optional[pd.DataFrame] = None,
                            test_target: Optional[pd.Series] = None) -> Dict[str, Any]:
    """
    Create a comprehensive diagnostic report for a fitted ARIMA model.
    
    Args:
        model: Fitted ARIMA model
        test_data: Test feature matrix (optional)
        test_target: Test target series (optional)
        
    Returns:
        Dictionary with diagnostics and plot figures
    """
    if not model.is_fitted:
        raise ValueError("Model must be fitted before creating diagnostic report")
    
    # Initialize report
    report = {
        'model_type': model.__class__.__name__,
        'order': model.order,
        'figures': {},
        'metrics': {}
    }
    
    # Add seasonal order if available
    if hasattr(model, 'seasonal_order'):
        report['seasonal_order'] = model.seasonal_order
    
    # Add information criteria
    if hasattr(model.model, 'aic'):
        report['metrics']['aic'] = model.model.aic
    if hasattr(model.model, 'bic'):
        report['metrics']['bic'] = model.model.bic
    if hasattr(model.model, 'hqic'):
        report['metrics']['hqic'] = model.model.hqic
    
    # Run diagnostic tests
    logger.info("Running diagnostics tests")
    report['diagnostics'] = run_all_diagnostics(model)
    
    # Generate figures
    
    # 1. Residual diagnostics
    logger.info("Creating residual diagnostic plots")
    report['figures']['residual_diagnostics'] = plot_residual_diagnostics(model)
    
    # 2. Time series decomposition
    try:
        logger.info("Creating decomposition plot")
        report['figures']['decomposition'] = plot_decomposition(model)
    except Exception as e:
        logger.error(f"Error creating decomposition plot: {e}")
    
    # 3. Forecast evaluation if test data is provided
    if test_data is not None and test_target is not None:
        try:
            logger.info("Creating forecast evaluation plot")
            report['figures']['forecast_evaluation'] = plot_forecast_evaluation(
                model, test_data, test_target
            )
            
            # Calculate test metrics
            predictions = model.predict(test_data)
            errors = test_target - predictions
            
            report['metrics']['test_rmse'] = np.sqrt(np.mean(errors ** 2))
            report['metrics']['test_mae'] = np.mean(np.abs(errors))
            
            # Calculate MAPE if no zeros in test_target
            if not np.any(test_target == 0):
                report['metrics']['test_mape'] = np.mean(np.abs(errors / test_target)) * 100
            else:
                # Alternative for data with zeros: sMAPE
                with np.errstate(divide='ignore', invalid='ignore'):
                    denominator = np.abs(test_target) + np.abs(predictions)
                    smape = np.mean(2 * np.abs(errors) / denominator) * 100
                    report['metrics']['test_smape'] = np.nan_to_num(smape)
        except Exception as e:
            logger.error(f"Error creating forecast evaluation: {e}")
    
    return report


def plot_model_summary(model: BaseARIMAModel, figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
    """
    Create a summary plot for a fitted ARIMA model showing original data, 
    fitted values, and residuals.
    
    Args:
        model: Fitted ARIMA model
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if not model.is_fitted:
        raise ValueError("Model must be fitted before creating summary plot")
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    # Extract data
    original_data = model.original_data
    fitted_values = model.fitted_values
    residuals = model.residuals
    
    # Use date index if available
    if hasattr(model, 'date_index_') and model.date_index_ is not None:
        x = model.date_index_
    else:
        x = np.arange(len(original_data))
    
    # 1. Original data and fitted values
    axes[0].plot(x, original_data, 'b-', label='Original Data')
    if fitted_values is not None:
        axes[0].plot(x, fitted_values, 'r-', label='Fitted Values')
    
    axes[0].set_title('Original Data and Fitted Values')
    axes[0].set_ylabel('Value')
    axes[0].legend()
    
    # 2. Residuals
    axes[1].plot(x, residuals, 'g-')
    axes[1].set_title('Residuals')
    axes[1].set_ylabel('Residual')
    axes[1].axhline(y=0, color='r', linestyle='-')
    
    # 3. Absolute residuals to check for heteroscedasticity
    axes[2].plot(x, np.abs(residuals), 'b-')
    axes[2].set_title('Absolute Residuals')
    axes[2].set_ylabel('|Residual|')
    axes[2].set_xlabel('Time')
    
    # Add model info to figure title
    model_title = f"{model.__class__.__name__}"
    if hasattr(model, 'order'):
        model_title += f" {model.order}"
    if hasattr(model, 'seasonal_order') and model.seasonal_order is not None:
        model_title += f"x{model.seasonal_order}"
        
    fig.suptitle(model_title, fontsize=14)
    
    # Format x-axis if dates are used
    if hasattr(model, 'date_index_') and model.date_index_ is not None:
        fig.autofmt_xdate()
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust for suptitle
    
    return fig