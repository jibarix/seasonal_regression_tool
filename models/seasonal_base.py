"""
Base classes and utilities for seasonal time series models.
Provides foundation for different seasonal representation approaches.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Union, Optional, Tuple, Any
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
import statsmodels.api as sm

# Import from project modules
from models.models import BaseModel

# Setup logging
logger = logging.getLogger(__name__)


class BaseSeasonalModel(BaseModel, ABC):
    """
    Abstract base class for seasonal time series models.
    Extends the BaseModel with seasonality-specific functionality.
    """
    
    def __init__(self, name: str = None, date_col: str = 'date', 
                 seasonality_type: str = 'monthly', **kwargs):
        """
        Initialize the seasonal model.
        
        Args:
            name: Model name
            date_col: Name of the date column
            seasonality_type: Type of seasonality ('monthly', 'quarterly', 'weekly')
            **kwargs: Additional parameters passed to BaseModel
        """
        super().__init__(name=name, date_col=date_col, **kwargs)
        
        self.seasonality_type = seasonality_type
        self.seasonal_period = self._get_seasonal_period(seasonality_type)
        self.seasonal_components = None
        
        # Model-specific params
        self.model_params.update({
            'seasonality_type': seasonality_type,
            'seasonal_period': self.seasonal_period
        })
        
        # Track feature groups
        self.feature_groups = {
            'seasonal_features': [],
            'trend_features': [],
            'exogenous_features': []
        }
    
    def _get_seasonal_period(self, seasonality_type: str) -> int:
        """
        Get the seasonal period based on seasonality type.
        
        Args:
            seasonality_type: Type of seasonality
            
        Returns:
            Seasonal period (number of time units in one seasonal cycle)
        """
        if seasonality_type == 'monthly':
            return 12
        elif seasonality_type == 'quarterly':
            return 4
        elif seasonality_type == 'weekly':
            return 7
        elif seasonality_type == 'daily':
            return 24
        else:
            logger.warning(f"Unknown seasonality type: {seasonality_type}, defaulting to monthly (12)")
            return 12
    
    @abstractmethod
    def add_seasonal_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add seasonal features to the feature matrix.
        Must be implemented by concrete subclasses.
        
        Args:
            X: Feature matrix
            
        Returns:
            Feature matrix with added seasonal features
        """
        pass
    
    @abstractmethod
    def extract_seasonal_components(self, X: pd.DataFrame = None) -> Dict[str, np.ndarray]:
        """
        Extract seasonal components from the fitted model.
        Must be implemented by concrete subclasses.
        
        Args:
            X: Optional feature matrix for prediction (if None, use training data)
            
        Returns:
            Dictionary mapping time periods to seasonal components
        """
        pass
    
    def preprocess(self, X: pd.DataFrame, y: Optional[pd.Series] = None, 
                  fit: bool = False) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Preprocess data before model fitting or prediction.
        Adds seasonal features and handles missing values.
        
        Args:
            X: Feature matrix
            y: Target variable (optional)
            fit: Whether this is for fitting (True) or prediction (False)
            
        Returns:
            Tuple of (processed_X, processed_y)
        """
        # First apply standard preprocessing from BaseModel
        processed_X, processed_y = super().preprocess(X, y, fit)
        
        # Add seasonal features
        processed_X = self.add_seasonal_features(processed_X)
        
        # If fitting, update feature groups tracking
        if fit and hasattr(self, 'feature_groups'):
            # Log feature groups for interpretability
            feature_groups_info = {
                group: len(features) for group, features in self.feature_groups.items()
            }
            logger.info(f"Feature groups: {feature_groups_info}")
        
        return processed_X, processed_y
    
    def get_seasonal_forecast(self, X: pd.DataFrame, forecast_horizon: int = 12) -> pd.DataFrame:
        """
        Generate a forecast with seasonal components.
        
        Args:
            X: Feature matrix for conditioning the forecast
            forecast_horizon: Number of periods to forecast
            
        Returns:
            DataFrame with date, forecast, and components (trend, seasonal)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        # Get the last date in the data
        if self.date_col in X.columns:
            last_date = pd.to_datetime(X[self.date_col].max())
            
            # Create future dates
            if self.seasonality_type == 'monthly':
                future_dates = pd.date_range(
                    start=last_date + pd.DateOffset(months=1),
                    periods=forecast_horizon,
                    freq='MS'
                )
            elif self.seasonality_type == 'quarterly':
                future_dates = pd.date_range(
                    start=last_date + pd.DateOffset(months=3),
                    periods=forecast_horizon,
                    freq='QS'
                )
            elif self.seasonality_type == 'weekly':
                future_dates = pd.date_range(
                    start=last_date + pd.DateOffset(weeks=1),
                    periods=forecast_horizon,
                    freq='W-MON'
                )
            else:
                # Default to monthly
                future_dates = pd.date_range(
                    start=last_date + pd.DateOffset(months=1),
                    periods=forecast_horizon,
                    freq='MS'
                )
        else:
            # If no date column, create dummy dates
            future_dates = pd.date_range(
                start=datetime.now(),
                periods=forecast_horizon,
                freq='MS'
            )
        
        # Create future feature matrix
        future_X = pd.DataFrame({self.date_col: future_dates})
        
        # Extract month, quarter, etc. for seasonal features
        future_X['month'] = future_X[self.date_col].dt.month
        future_X['quarter'] = future_X[self.date_col].dt.quarter
        future_X['year'] = future_X[self.date_col].dt.year
        
        # Get predictions
        future_X = self.add_seasonal_features(future_X)
        predictions = self.predict(future_X)
        
        # Extract components if the model supports it
        components = self.extract_seasonal_components(future_X)
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'forecast': predictions
        })
        
        # Add components if available
        if components:
            for component_name, component_values in components.items():
                forecast_df[component_name] = component_values
        
        return forecast_df
    
    def plot_seasonal_components(self, X: Optional[pd.DataFrame] = None, 
                               y: Optional[pd.Series] = None,
                               ax: Optional[plt.Axes] = None) -> plt.Figure:
        """
        Plot seasonal components extracted from the model.
        
        Args:
            X: Optional feature matrix
            y: Optional target variable
            ax: Optional matplotlib axes
            
        Returns:
            Matplotlib figure
        """
        # Extract seasonal components
        components = self.extract_seasonal_components(X)
        
        if not components:
            logger.warning("No seasonal components available for plotting")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No seasonal components available", 
                    ha='center', va='center', fontsize=12)
            return fig
        
        # Create figure and axes if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure
        
        # Plot each seasonal period
        if 'seasonal' in components:
            seasonal = components['seasonal']
            periods = np.arange(1, len(seasonal) + 1)
            
            if self.seasonality_type == 'monthly':
                # Use month names for x-axis
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                x_labels = month_names[:len(seasonal)]
            else:
                x_labels = periods
            
            ax.bar(periods, seasonal, alpha=0.7, color='royalblue')
            ax.set_xticks(periods)
            ax.set_xticklabels(x_labels)
            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            
            ax.set_title(f"Seasonal Components ({self.seasonality_type.capitalize()})")
            ax.set_ylabel("Effect")
            
        return fig


def detect_seasonality(series: pd.Series, max_lag: int = 48) -> Dict[str, Any]:
    """
    Detect seasonality in a time series using autocorrelation.
    
    Args:
        series: Time series data
        max_lag: Maximum lag to consider
        
    Returns:
        Dictionary with seasonality information
    """
    # Calculate autocorrelation
    acf_values = acf(series.dropna(), nlags=max_lag, fft=True)
    
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
        'monthly': 30,
        'quarterly': 90,
        'annual': 365
    }
    
    detected_periods = {}
    
    for period_name, period in seasonal_periods.items():
        # Look for peaks close to the expected period
        for lag in range(max(1, period - 3), min(max_lag, period + 4)):
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
            if abs(top_peak - period) <= 3:
                primary_seasonality = period_name
                break
        
        if primary_seasonality is None:
            # Custom seasonality
            primary_seasonality = f"custom_{top_peak}"
    
    return {
        'has_seasonality': len(peak_indices) > 0,
        'primary_seasonality': primary_seasonality,
        'detected_periods': detected_periods,
        'peaks': sorted_peaks[:3],  # Top 3 peaks
        'peak_values': [acf_values[i] for i in sorted_peaks[:3]] if sorted_peaks else []
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


def prepare_seasonal_features(df: pd.DataFrame, date_col: str = 'date', 
                            seasonality_type: str = 'monthly', 
                            include_fourier: bool = True) -> pd.DataFrame:
    """
    Prepare standard seasonal features based on date information.
    
    Args:
        df: DataFrame with time series data
        date_col: Name of the date column
        seasonality_type: Type of seasonality ('monthly', 'quarterly', 'weekly')
        include_fourier: Whether to include Fourier terms
        
    Returns:
        DataFrame with added seasonal features
    """
    # Create a copy to avoid modifying the original
    result = df.copy()
    
    # Ensure date column is datetime
    if date_col in result.columns:
        result[date_col] = pd.to_datetime(result[date_col])
    else:
        logger.warning(f"Date column '{date_col}' not found in DataFrame")
        return result
    
    # Add basic date components
    result['year'] = result[date_col].dt.year
    result['month'] = result[date_col].dt.month
    result['quarter'] = result[date_col].dt.quarter
    
    # Add dummy variables based on seasonality type
    if seasonality_type == 'monthly':
        # Add month dummies (use 11 dummies, dropping the first)
        for month in range(2, 13):
            result[f'month_{month}'] = (result['month'] == month).astype(int)
            
    elif seasonality_type == 'quarterly':
        # Add quarter dummies (use 3 dummies, dropping the first)
        for quarter in range(2, 5):
            result[f'quarter_{quarter}'] = (result['quarter'] == quarter).astype(int)
            
    elif seasonality_type == 'weekly':
        # Add day of week dummies (use 6 dummies, dropping the first)
        result['day_of_week'] = result[date_col].dt.dayofweek
        for day in range(1, 7):
            result[f'day_{day}'] = (result['day_of_week'] == day).astype(int)
    
    # Add Fourier terms if requested
    if include_fourier:
        # Set seasonal period based on seasonality type
        if seasonality_type == 'monthly':
            period = 12
        elif seasonality_type == 'quarterly':
            period = 4
        elif seasonality_type == 'weekly':
            period = 7
        else:
            period = 12
        
        # Create time index
        time_idx = np.arange(len(result))
        
        # Add sine and cosine terms for first two harmonics
        for harm in range(1, 3):
            result[f'sin_{harm}'] = np.sin(2 * np.pi * harm * time_idx / period)
            result[f'cos_{harm}'] = np.cos(2 * np.pi * harm * time_idx / period)
    
    return result


def evaluate_seasonal_model(y_true: pd.Series, y_pred: np.ndarray, 
                          date_series: pd.Series = None,
                          seasonality_type: str = 'monthly') -> Dict[str, float]:
    """
    Evaluate a seasonal model with metrics that consider seasonality.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        date_series: Series of dates corresponding to the values
        seasonality_type: Type of seasonality
        
    Returns:
        Dictionary of seasonal evaluation metrics
    """
    # Standard metrics from base evaluation
    metrics = {
        'rmse': np.sqrt(np.mean((y_true - y_pred) ** 2)),
        'mae': np.mean(np.abs(y_true - y_pred)),
    }
    
    # If dates are provided, calculate season-specific metrics
    if date_series is not None:
        date_series = pd.to_datetime(date_series)
        
        if seasonality_type == 'monthly':
            # Calculate metrics by month
            months = date_series.dt.month
            month_metrics = {}
            
            for month in range(1, 13):
                mask = (months == month)
                if sum(mask) > 0:
                    month_metrics[f'rmse_month{month}'] = np.sqrt(
                        np.mean((y_true[mask] - y_pred[mask]) ** 2)
                    )
            
            metrics.update(month_metrics)
            
            # Calculate best and worst month
            month_rmse = [month_metrics.get(f'rmse_month{m}', float('inf')) for m in range(1, 13)]
            best_month = np.argmin(month_rmse) + 1
            worst_month = np.argmax(month_rmse) + 1
            
            metrics['best_month'] = float(best_month)
            metrics['worst_month'] = float(worst_month)
            
        elif seasonality_type == 'quarterly':
            # Calculate metrics by quarter
            quarters = date_series.dt.quarter
            quarter_metrics = {}
            
            for quarter in range(1, 5):
                mask = (quarters == quarter)
                if sum(mask) > 0:
                    quarter_metrics[f'rmse_q{quarter}'] = np.sqrt(
                        np.mean((y_true[mask] - y_pred[mask]) ** 2)
                    )
            
            metrics.update(quarter_metrics)
    
    # Calculate R² (coefficient of determination)
    if len(y_true) > 0:
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        
        if ss_total > 0:
            metrics['r_squared'] = 1 - (ss_residual / ss_total)
        else:
            metrics['r_squared'] = 0
    
    return metrics