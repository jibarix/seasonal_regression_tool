"""
Utility functions for seasonal time series analysis.
Provides diagnostic tools, visualization functions, and helper methods.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Union, Optional, Tuple, Any
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm

# Setup logging
logger = logging.getLogger(__name__)


def detect_seasonality(series: pd.Series, max_lag: int = 48) -> Dict[str, Any]:
    """
    Detect seasonality in a time series using autocorrelation.
    
    Args:
        series: Time series data
        max_lag: Maximum lag to consider
        
    Returns:
        Dictionary with seasonality information
    """
    # Handle missing values
    series = series.dropna()
    
    if len(series) < max_lag + 1:
        logger.warning(f"Series length ({len(series)}) is less than max_lag + 1 ({max_lag + 1})")
        max_lag = max(len(series) // 2, 12)
    
    # Calculate autocorrelation
    acf_values = acf(series, nlags=max_lag, fft=True)
    
    # Find peaks in autocorrelation
    peak_indices = []
    for i in range(2, len(acf_values) - 1):
        if acf_values[i] > acf_values[i-1] and acf_values[i] > acf_values[i+1]:
            if acf_values[i] > 0.2:  # Only consider significant peaks
                peak_indices.append(i)
    
    # Sort peaks by autocorrelation value
    sorted_peaks = sorted(peak_indices, key=lambda i: acf_values[i], reverse=True)
    
    # Check specific seasonal periods
    seasonal_periods = {
        'weekly': 7,
        'monthly': 12,
        'quarterly': 4,
        'biannual': 6,
        'annual': 12
    }
    
    detected_periods = {}
    
    for period_name, period in seasonal_periods.items():
        # Look for peaks close to the expected period
        for lag in range(max(1, period - 2), min(max_lag, period + 3)):
            if lag in peak_indices:
                detected_periods[period_name] = {
                    'lag': lag,
                    'strength': acf_values[lag]
                }
                break
    
    # Check if any of the top peaks matches common seasonality
    primary_seasonality = None
    if sorted_peaks:
        top_peak = sorted_peaks[0]
        
        # Check if the top peak is close to a known seasonality
        for period_name, period in seasonal_periods.items():
            if abs(top_peak - period) <= 2:
                primary_seasonality = period_name
                break
        
        if primary_seasonality is None:
            # Custom seasonality
            primary_seasonality = f"custom_{top_peak}"
    
    return {
        'has_seasonality': len(peak_indices) > 0,
        'primary_seasonality': primary_seasonality,
        'detected_periods': detected_periods,
        'peaks': [{'lag': i, 'value': acf_values[i]} for i in sorted_peaks[:3]],
        'acf_values': acf_values.tolist()
    }


def seasonal_strength_test(series: pd.Series, period: int = 12) -> float:
    """
    Calculate the strength of seasonality in a time series.
    Based on Rob J. Hyndman's method in "Forecasting: Principles and Practice"
    
    Args:
        series: Time series data
        period: Seasonal period to test
        
    Returns:
        Seasonality strength (0-1, higher is stronger seasonality)
    """
    # Ensure sufficient data
    if len(series) < period * 2:
        logger.warning(f"Not enough data for seasonal strength test (need at least {period * 2} points)")
        return 0.0
    
    # Handle missing values
    series = series.interpolate().bfill().ffill()
    
    # Decompose the series
    try:
        decomposition = seasonal_decompose(series, period=period, model='additive')
        
        # Get components
        seasonal = decomposition.seasonal
        residual = decomposition.resid
        
        # Calculate variances
        var_seasonal = np.nanvar(seasonal)
        var_residual = np.nanvar(residual)
        
        # Calculate strength
        if var_seasonal + var_residual > 0:
            strength = max(0, 1 - var_residual / (var_seasonal + var_residual))
        else:
            strength = 0
            
        return strength
        
    except Exception as e:
        logger.error(f"Error in seasonal strength test: {e}")
        return 0.0


def plot_seasonal_decomposition(series: pd.Series, period: int = 12, 
                              model: str = 'additive', figsize: Tuple[int, int] = (10, 8)):
    """
    Plot seasonal decomposition of a time series.
    
    Args:
        series: Time series data
        period: Seasonal period
        model: Decomposition model ('additive' or 'multiplicative')
        figsize: Figure size
        
    Returns:
        Matplotlib figure and axes
    """
    # Handle missing values
    series = series.interpolate().bfill().ffill()
    
    try:
        # Perform decomposition
        decomposition = seasonal_decompose(series, period=period, model=model)
        
        # Create figure
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
        
        # Original Series
        series.plot(ax=axes[0], color='blue')
        axes[0].set_ylabel('Original')
        axes[0].set_title(f'Seasonal Decomposition (Period={period}, Model={model})')
        
        # Trend Component
        decomposition.trend.plot(ax=axes[1], color='red')
        axes[1].set_ylabel('Trend')
        
        # Seasonal Component
        decomposition.seasonal.plot(ax=axes[2], color='green')
        axes[2].set_ylabel('Seasonal')
        
        # Residual Component
        decomposition.resid.plot(ax=axes[3], color='purple')
        axes[3].set_ylabel('Residual')
        
        plt.tight_layout()
        
        return fig, axes
        
    except Exception as e:
        logger.error(f"Error in seasonal decomposition plot: {e}")
        return None, None


def plot_seasonal_patterns(series: pd.Series, date_index: pd.DatetimeIndex,
                         figsize: Tuple[int, int] = (12, 8)):
    """
    Plot seasonal patterns in different ways (yearly, monthly, box plots).
    
    Args:
        series: Time series data
        date_index: Datetime index corresponding to series
        figsize: Figure size
        
    Returns:
        Matplotlib figure and axes
    """
    # Create DataFrame with date index
    df = pd.DataFrame({'value': series}, index=date_index)
    
    # Add date components
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['month_name'] = df.index.strftime('%b')
    df['quarter'] = df.index.quarter
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Plot yearly pattern
    sns.lineplot(data=df, x='month', y='value', hue='year', ax=axes[0, 0])
    axes[0, 0].set_title('Yearly Patterns')
    axes[0, 0].set_xticks(range(1, 13))
    axes[0, 0].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    axes[0, 0].set_xlabel('Month')
    axes[0, 0].legend(title='Year', loc='best')
    
    # 2. Plot seasonal subseries
    month_avg = df.groupby('month')['value'].mean()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_avg.index = month_names
    month_avg.plot(ax=axes[0, 1], marker='o', color='red')
    axes[0, 1].set_title('Average Monthly Pattern')
    axes[0, 1].set_xlabel('Month')
    axes[0, 1].set_ylabel('Average Value')
    
    # 3. Box plot by month
    sns.boxplot(data=df, x='month_name', y='value', ax=axes[1, 0])
    axes[1, 0].set_title('Monthly Box Plot')
    axes[1, 0].set_xticklabels(month_names)
    axes[1, 0].set_xlabel('Month')
    
    # 4. Box plot by quarter
    sns.boxplot(data=df, x='quarter', y='value', ax=axes[1, 1])
    axes[1, 1].set_title('Quarterly Box Plot')
    axes[1, 1].set_xlabel('Quarter')
    axes[1, 1].set_xticks(range(1, 5))
    axes[1, 1].set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'])
    
    plt.tight_layout()
    
    return fig, axes


def plot_seasonal_heatmap(series: pd.Series, date_index: pd.DatetimeIndex,
                        figsize: Tuple[int, int] = (12, 8)):
    """
    Create a heatmap showing seasonal patterns across years and months.
    
    Args:
        series: Time series data
        date_index: Datetime index corresponding to series
        figsize: Figure size
        
    Returns:
        Matplotlib figure and axis
    """
    # Create DataFrame with date index
    df = pd.DataFrame({'value': series}, index=date_index)
    
    # Extract year and month
    df['year'] = df.index.year
    df['month'] = df.index.month
    
    # Calculate monthly averages
    monthly_avg = df.groupby(['year', 'month'])['value'].mean().unstack(level='month')
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(monthly_avg, annot=True, fmt='.2f', cmap='YlGnBu',
              cbar_kws={'label': 'Value'}, ax=ax)
    
    # Add labels
    ax.set_title('Seasonal Heatmap (Year vs. Month)')
    ax.set_xlabel('Month')
    ax.set_ylabel('Year')
    
    # Set x-tick labels to month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_xticklabels(month_names)
    
    return fig, ax


def plot_acf_pacf(series: pd.Series, lags: int = 36, figsize: Tuple[int, int] = (12, 6)):
    """
    Plot ACF and PACF with seasonal period markers.
    
    Args:
        series: Time series data
        lags: Maximum number of lags to plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure and axes
    """
    # Handle missing values
    series = series.dropna()
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    try:
        # Plot ACF
        plot_acf(series, lags=lags, ax=axes[0])
        axes[0].set_title('Autocorrelation Function (ACF)')
        
        # Add vertical lines at seasonal periods
        for period, color, label in [(7, 'red', 'Weekly'), 
                                  (12, 'green', 'Monthly/Annual'), 
                                  (4, 'blue', 'Quarterly')]:
            if period < lags:
                axes[0].axvline(x=period, color=color, linestyle='--', alpha=0.5, label=label)
        
        axes[0].legend()
        
        # Plot PACF
        plot_pacf(series, lags=lags, ax=axes[1])
        axes[1].set_title('Partial Autocorrelation Function (PACF)')
        
        # Add vertical lines at seasonal periods
        for period, color, label in [(7, 'red', 'Weekly'), 
                                  (12, 'green', 'Monthly/Annual'), 
                                  (4, 'blue', 'Quarterly')]:
            if period < lags:
                axes[1].axvline(x=period, color=color, linestyle='--', alpha=0.5, label=label)
        
        axes[1].legend()
        
        plt.tight_layout()
        
        return fig, axes
        
    except Exception as e:
        logger.error(f"Error in ACF/PACF plot: {e}")
        return None, None


def compare_seasonal_patterns(models: Dict[str, Any], figsize: Tuple[int, int] = (12, 6)):
    """
    Compare seasonal patterns from different models.
    
    Args:
        models: Dictionary mapping model names to fitted model objects
        figsize: Figure size
        
    Returns:
        Matplotlib figure and axis
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Collect seasonal components
    seasonal_components = {}
    
    for model_name, model in models.items():
        if hasattr(model, 'extract_seasonal_components'):
            components = model.extract_seasonal_components()
            
            if components and 'seasonal' in components:
                seasonal_components[model_name] = components['seasonal']
    
    # If no seasonal components found
    if not seasonal_components:
        ax.text(0.5, 0.5, "No seasonal components found in models",
              ha='center', va='center', fontsize=14)
        return fig, ax
    
    # Determine period (length of seasonal cycle)
    period = len(next(iter(seasonal_components.values())))
    
    # Create x-axis labels based on period
    if period == 12:
        x_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    elif period == 4:
        x_labels = ['Q1', 'Q2', 'Q3', 'Q4']
    else:
        x_labels = [str(i+1) for i in range(period)]
    
    # Plot each model's seasonal component
    for model_name, seasonal in seasonal_components.items():
        ax.plot(range(1, period+1), seasonal, marker='o', label=model_name)
    
    # Add chart elements
    ax.set_title('Comparison of Seasonal Components')
    ax.set_xlabel('Period')
    ax.set_ylabel('Seasonal Effect')
    ax.set_xticks(range(1, period+1))
    ax.set_xticklabels(x_labels)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def plot_model_comparison(comparison_df: pd.DataFrame, metrics: List[str] = ['rmse', 'r_squared'],
                        figsize: Tuple[int, int] = (12, 8)):
    """
    Plot comparison of model performance metrics.
    
    Args:
        comparison_df: DataFrame with model comparison metrics
        metrics: List of metrics to plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure and axes
    """
    if comparison_df.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No comparison data available",
              ha='center', va='center', fontsize=14)
        return fig, ax
    
    # Number of metrics to plot
    n_metrics = len(metrics)
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_metrics, 1, figsize=figsize)
    
    # If only one metric, axes won't be array
    if n_metrics == 1:
        axes = [axes]
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        if metric not in comparison_df.columns:
            axes[i].text(0.5, 0.5, f"Metric '{metric}' not found in data",
                       ha='center', va='center', fontsize=12)
            continue
        
        # Sort by the metric
        ascending = metric not in ['r_squared', 'adj_r_squared']  # Lower is better except for R²
        sorted_df = comparison_df.sort_values(metric, ascending=ascending)
        
        # Create horizontal bar chart
        bars = axes[i].barh(sorted_df['model_name'], sorted_df[metric])
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            axes[i].text(width * 1.05, bar.get_y() + bar.get_height()/2,
                       f'{width:.4f}', va='center')
        
        # Add title and labels
        axes[i].set_title(f'Model Comparison - {metric.upper()}')
        axes[i].set_xlabel(metric)
        axes[i].set_ylabel('Model')
        
        # Add grid lines
        axes[i].grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    return fig, axes


def plot_seasonal_coefficients(model: Any, figsize: Tuple[int, int] = (10, 6)):
    """
    Plot seasonal coefficients from a model.
    
    Args:
        model: Fitted model with statsmodels results
        figsize: Figure size
        
    Returns:
        Matplotlib figure and axis
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Try to get coefficients
    try:
        # Get model params
        params = getattr(getattr(model, 'model', None), 'params', None)
        
        if params is None:
            # Try to get seasonal components directly
            if hasattr(model, 'extract_seasonal_components'):
                components = model.extract_seasonal_components()
                
                if components and 'seasonal' in components:
                    seasonal = components['seasonal']
                    period = len(seasonal)
                    
                    # Create x-axis labels based on period
                    if period == 12:
                        x_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    elif period == 4:
                        x_labels = ['Q1', 'Q2', 'Q3', 'Q4']
                    else:
                        x_labels = [str(i+1) for i in range(period)]
                    
                    # Plot seasonal components
                    ax.bar(range(1, period+1), seasonal, color='steelblue')
                    ax.set_xticks(range(1, period+1))
                    ax.set_xticklabels(x_labels)
                    ax.set_title('Seasonal Components')
                    ax.set_xlabel('Period')
                    ax.set_ylabel('Effect')
                    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
                    
                    return fig, ax
            
            ax.text(0.5, 0.5, "No seasonal coefficients found",
                  ha='center', va='center', fontsize=14)
            return fig, ax
        
        # Identify seasonal coefficients based on parameter names
        seasonal_params = {}
        
        # Check for different types of seasonal parameters
        for name, value in params.items():
            if name.startswith('month_'):
                month = int(name.split('_')[1])
                seasonal_params[month] = value
            elif name.startswith('quarter_'):
                quarter = int(name.split('_')[1])
                seasonal_params[quarter] = value
            elif name.startswith('Q') and len(name) == 2 and name[1].isdigit():
                quarter = int(name[1])
                seasonal_params[quarter] = value
        
        # If no seasonal parameters found directly, check for Fourier terms
        if not seasonal_params:
            sin_params = {name: value for name, value in params.items() if name.startswith('sin_')}
            cos_params = {name: value for name, value in params.items() if name.startswith('cos_')}
            
            if sin_params and cos_params:
                # Reconstruct seasonal pattern from Fourier terms
                period = 12  # Default to monthly
                seasonal = np.zeros(period)
                
                for name, value in sin_params.items():
                    if 'sin_h' in name:
                        harmonic = int(name.split('h')[1])
                        for i in range(period):
                            seasonal[i] += value * np.sin(2 * np.pi * harmonic * (i+1) / period)
                
                for name, value in cos_params.items():
                    if 'cos_h' in name:
                        harmonic = int(name.split('h')[1])
                        for i in range(period):
                            seasonal[i] += value * np.cos(2 * np.pi * harmonic * (i+1) / period)
                
                # Add the intercept effect
                if 'const' in params:
                    seasonal += params['const']
                
                # Plot reconstructed seasonal pattern
                x_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                
                ax.bar(range(1, period+1), seasonal, color='steelblue')
                ax.set_xticks(range(1, period+1))
                ax.set_xticklabels(x_labels)
                ax.set_title('Reconstructed Seasonal Pattern from Fourier Terms')
                ax.set_xlabel('Month')
                ax.set_ylabel('Effect')
                ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
                
                return fig, ax
        
        # Plot seasonal parameters if found
        if seasonal_params:
            # Add base effect (usually the intercept)
            base_effect = params.get('const', 0)
            
            # Sort by period index
            sorted_periods = sorted(seasonal_params.keys())
            
            # Get period labels
            if max(sorted_periods) == 12:
                period_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                title = 'Monthly Seasonal Coefficients'
                xlabel = 'Month'
                
                # If we're missing January (common with dummy variables), add it
                if 1 not in seasonal_params:
                    seasonal_params[1] = base_effect
                    sorted_periods = sorted(seasonal_params.keys())
            elif max(sorted_periods) == 4:
                period_labels = ['Q1', 'Q2', 'Q3', 'Q4']
                title = 'Quarterly Seasonal Coefficients'
                xlabel = 'Quarter'
                
                # If we're missing Q1, add it
                if 1 not in seasonal_params:
                    seasonal_params[1] = base_effect
                    sorted_periods = sorted(seasonal_params.keys())
            else:
                period_labels = [str(p) for p in sorted_periods]
                title = 'Seasonal Coefficients'
                xlabel = 'Period'
            
            # Get coefficient values
            coefficient_values = [seasonal_params[p] for p in sorted_periods]
            
            # Plot
            ax.bar(sorted_periods, coefficient_values, color='steelblue')
            ax.set_xticks(sorted_periods)
            ax.set_xticklabels(period_labels)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Coefficient')
            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            
            return fig, ax
        
        # If we get here, no seasonal coefficients were found
        ax.text(0.5, 0.5, "No seasonal coefficients found",
              ha='center', va='center', fontsize=14)
        
    except Exception as e:
        logger.error(f"Error plotting seasonal coefficients: {e}")
        ax.text(0.5, 0.5, f"Error: {str(e)}",
              ha='center', va='center', fontsize=12)
    
    return fig, ax


def plot_forecast_by_season(forecasts: pd.DataFrame, actuals: Optional[pd.Series] = None,
                           date_col: str = 'date', value_col: str = 'forecast',
                           figsize: Tuple[int, int] = (12, 8)):
    """
    Plot forecasts grouped by season (month).
    
    Args:
        forecasts: DataFrame with forecasts
        actuals: Optional Series with actual values (for comparison)
        date_col: Name of date column
        value_col: Name of forecast value column
        figsize: Figure size
        
    Returns:
        Matplotlib figure and axes
    """
    # Ensure dates are datetime
    forecasts[date_col] = pd.to_datetime(forecasts[date_col])
    
    # Extract month
    forecasts['month'] = forecasts[date_col].dt.month
    forecasts['month_name'] = forecasts[date_col].dt.strftime('%b')
    
    # Create figure with subplots (one for each month)
    fig, axes = plt.subplots(3, 4, figsize=figsize, sharex=True, sharey=True)
    axes = axes.flatten()
    
    # Month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Plot each month
    for i, month in enumerate(range(1, 13), 0):
        month_data = forecasts[forecasts['month'] == month]
        
        if len(month_data) > 0:
            # Plot forecasts
            axes[i].plot(month_data[date_col], month_data[value_col], 
                        marker='o', linestyle='-', color='blue', label='Forecast')
            
            # Add actuals if provided
            if actuals is not None:
                # Match actuals to the same dates
                actuals_df = pd.DataFrame({
                    date_col: actuals.index if isinstance(actuals.index, pd.DatetimeIndex) else pd.to_datetime(actuals.index),
                    'actual': actuals.values
                })
                actuals_df['month'] = actuals_df[date_col].dt.month
                
                # Filter for current month
                month_actuals = actuals_df[actuals_df['month'] == month]
                
                if len(month_actuals) > 0:
                    axes[i].plot(month_actuals[date_col], month_actuals['actual'], 
                                marker='x', linestyle='--', color='red', label='Actual')
            
            # Add labels and formatting
            axes[i].set_title(month_names[i-1])
            axes[i].grid(True, alpha=0.3)
            
            # Format x-axis to show years only
            axes[i].xaxis.set_major_locator(MaxNLocator(nbins=5))
            for tick in axes[i].get_xticklabels():
                tick.set_rotation(45)
        else:
            axes[i].text(0.5, 0.5, f"No data for {month_names[i-1]}",
                        ha='center', va='center', fontsize=12)
    
    # Add common legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=2)
    
    # Add common labels
    fig.text(0.5, 0.01, 'Date', ha='center', va='center')
    fig.text(0.01, 0.5, 'Value', ha='center', va='center', rotation='vertical')
    fig.text(0.5, 0.95, 'Forecast by Month', ha='center', va='center', fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    return fig, axes


def create_seasonal_diagnostic_report(model: Any, X: pd.DataFrame, y: pd.Series,
                                    date_col: str = 'date', figsize: Tuple[int, int] = (12, 10)):
    """
    Create a comprehensive seasonal diagnostic report with multiple plots.
    
    Args:
        model: Fitted model
        X: Feature matrix
        y: Target variable
        date_col: Name of date column
        figsize: Base figure size
        
    Returns:
        Dictionary with generated figures
    """
    figures = {}
    
    # Get date index
    if date_col in X.columns:
        date_index = pd.to_datetime(X[date_col])
    else:
        # Create dummy date index if needed
        date_index = pd.date_range(start='2020-01-01', periods=len(y), freq='MS')
        logger.warning(f"No '{date_col}' column found. Using dummy date index.")
    
    # 1. Original series with seasonal decomposition
    if len(y) >= 24:  # Need at least 2 seasonal cycles
        try:
            # Create time series with date index
            ts = pd.Series(y.values, index=date_index)
            
            # Perform seasonal decomposition
            fig_decomp, _ = plot_seasonal_decomposition(ts)
            figures['decomposition'] = fig_decomp
        except Exception as e:
            logger.error(f"Error in seasonal decomposition: {e}")
    
    # 2. Seasonal patterns visualization
    try:
        fig_patterns, _ = plot_seasonal_patterns(y, date_index)
        figures['patterns'] = fig_patterns
    except Exception as e:
        logger.error(f"Error in seasonal patterns plot: {e}")
    
    # 3. Seasonal heatmap
    try:
        fig_heatmap, _ = plot_seasonal_heatmap(y, date_index)
        figures['heatmap'] = fig_heatmap
    except Exception as e:
        logger.error(f"Error in seasonal heatmap: {e}")
    
    # 4. ACF/PACF plots
    try:
        fig_acf, _ = plot_acf_pacf(y)
        figures['acf_pacf'] = fig_acf
    except Exception as e:
        logger.error(f"Error in ACF/PACF plot: {e}")
    
    # 5. Model seasonal coefficients
    try:
        fig_coef, _ = plot_seasonal_coefficients(model)
        figures['coefficients'] = fig_coef
    except Exception as e:
        logger.error(f"Error in seasonal coefficients plot: {e}")
    
    # 6. Model predictions vs actual
    try:
        y_pred = model.predict(X)
        
        # Create figure
        fig_pred, ax_pred = plt.subplots(figsize=figsize)
        
        # Plot actual and predicted
        ax_pred.plot(date_index, y, color='blue', label='Actual')
        ax_pred.plot(date_index, y_pred, color='red', label='Predicted')
        
        # Add labels and legend
        ax_pred.set_title('Model Fit: Actual vs Predicted')
        ax_pred.set_xlabel('Date')
        ax_pred.set_ylabel('Value')
        ax_pred.legend()
        ax_pred.grid(True, alpha=0.3)
        
        # Format x-axis
        plt.xticks(rotation=45)
        
        figures['predictions'] = fig_pred
    except Exception as e:
        logger.error(f"Error in predictions plot: {e}")
    
    # 7. Residuals by season
    try:
        if 'predictions' in locals():
            residuals = y - y_pred
            
            # Create DataFrame with residuals and date
            residuals_df = pd.DataFrame({
                'date': date_index,
                'residual': residuals
            })
            
            # Add month
            residuals_df['month'] = residuals_df['date'].dt.month
            residuals_df['month_name'] = residuals_df['date'].dt.strftime('%b')
            
            # Create boxplot of residuals by month
            fig_res, ax_res = plt.subplots(figsize=figsize)
            
            sns.boxplot(data=residuals_df, x='month_name', y='residual', ax=ax_res)
            ax_res.set_title('Residuals by Month')
            ax_res.set_xlabel('Month')
            ax_res.set_ylabel('Residual')
            ax_res.axhline(y=0, color='red', linestyle='--')
            
            # Set x-tick labels to month names
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            ax_res.set_xticklabels(month_names)
            
            figures['residuals'] = fig_res
    except Exception as e:
        logger.error(f"Error in residuals plot: {e}")
    
    return figures