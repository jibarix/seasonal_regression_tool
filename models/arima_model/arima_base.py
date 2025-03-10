"""
Base module for ARIMA time series models.
Provides abstract base class and utilities for all ARIMA implementations.
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Union, Optional, Tuple, Any
from abc import ABC, abstractmethod
from datetime import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm

# Import from project modules
from models.models import BaseModel

# Setup logging
logger = logging.getLogger(__name__)


class BaseARIMAModel(BaseModel, ABC):
    """
    Abstract base class for ARIMA time series models.
    Extends the BaseModel with ARIMA-specific functionality.
    """
    
    def __init__(self, name: str = None, date_col: str = 'date',
                 order: Tuple[int, int, int] = (1, 0, 0),
                 seasonal_order: Optional[Tuple[int, int, int, int]] = None,
                 **kwargs):
        """
        Initialize the ARIMA model.
        
        Args:
            name: Model name
            date_col: Name of the date column
            order: ARIMA order (p, d, q) - autoregressive, differencing, moving average
            seasonal_order: Seasonal ARIMA order (P, D, Q, s) - seasonal AR, differencing, MA, period
            **kwargs: Additional parameters passed to BaseModel
        """
        super().__init__(name=name or f"ARIMA_{order}", date_col=date_col, **kwargs)
        
        self.order = order
        self.seasonal_order = seasonal_order
        self.differencing_order = order[1]
        self.original_data = None
        
        # Model-specific params
        self.model_params.update({
            'order': order,
            'seasonal_order': seasonal_order
        })
        
        # Results
        self.results = None
        self.fitted_values = None
        self.residuals = None
        
    def preprocess(self, X: pd.DataFrame, y: Optional[pd.Series] = None, 
                  fit: bool = False) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Preprocess data for ARIMA modeling.
        
        Args:
            X: Feature matrix
            y: Target variable (optional)
            fit: Whether this is for fitting (True) or prediction (False)
            
        Returns:
            Tuple of (processed_X, processed_y)
        """
        # First apply standard preprocessing from BaseModel
        processed_X, processed_y = super().preprocess(X, y, fit)
        
        # Store original data if fitting
        if fit and y is not None:
            self.original_data = y.copy()
        
        # If y is None but we're predicting, use fitted values and extend
        if not fit and y is None and self.fitted_values is not None:
            processed_y = self.fitted_values.copy()
            
        return processed_X, processed_y
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'BaseARIMAModel':
        """
        Fit the ARIMA model to the training data.
        Must be implemented by concrete subclasses.
        
        Args:
            X: Feature matrix
            y: Target variable
            **kwargs: Additional model-specific parameters
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Generate predictions using the fitted model.
        Must be implemented by concrete subclasses.
        
        Args:
            X: Feature matrix
            **kwargs: Additional prediction parameters
            
        Returns:
            Array of predictions
        """
        pass
    
    def forecast(self, steps: int, X: Optional[pd.DataFrame] = None, 
                return_conf_int: bool = False, alpha: float = 0.05) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Generate forecasts for future periods.
        
        Args:
            steps: Number of steps ahead to forecast
            X: Future feature matrix for models with exogenous variables
            return_conf_int: Whether to return confidence intervals
            alpha: Significance level for confidence intervals
            
        Returns:
            Forecasts and optionally confidence intervals
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        # Should be implemented by subclasses
        raise NotImplementedError("Forecast method must be implemented by subclasses")
    
    def get_forecast_dates(self, last_date: datetime, steps: int, freq: str = 'MS') -> pd.DatetimeIndex:
        """
        Generate future dates for forecasting.
        
        Args:
            last_date: Last date in the training data
            steps: Number of steps ahead to forecast
            freq: Frequency of the dates ('MS' for month start, 'QS' for quarter start)
            
        Returns:
            DatetimeIndex of future dates
        """
        return pd.date_range(start=last_date, periods=steps + 1, freq=freq)[1:]
    
    def plot_diagnostics(self, figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
        """
        Plot diagnostic plots for ARIMA model.
        
        Args:
            figsize: Size of the figure
            
        Returns:
            Matplotlib figure
        """
        if not self.is_fitted or self.residuals is None:
            raise ValueError("Model must be fitted before plotting diagnostics")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot residuals
        axes[0, 0].plot(self.residuals)
        axes[0, 0].set_title('Residuals')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Residual')
        
        # Plot histogram of residuals
        axes[0, 1].hist(self.residuals, bins=20)
        axes[0, 1].set_title('Histogram of Residuals')
        axes[0, 1].set_xlabel('Residual')
        axes[0, 1].set_ylabel('Frequency')
        
        # Plot ACF of residuals
        sm.graphics.tsa.plot_acf(self.residuals, lags=20, ax=axes[1, 0])
        axes[1, 0].set_title('ACF of Residuals')
        
        # Plot PACF of residuals
        sm.graphics.tsa.plot_pacf(self.residuals, lags=20, ax=axes[1, 1])
        axes[1, 1].set_title('PACF of Residuals')
        
        plt.tight_layout()
        return fig
    
    def plot_components(self, figsize: Tuple[int, int] = (12, 10)) -> Optional[plt.Figure]:
        """
        Plot decomposition of the time series.
        
        Args:
            figsize: Size of the figure
            
        Returns:
            Matplotlib figure or None if decomposition is not possible
        """
        if not self.is_fitted or self.original_data is None:
            raise ValueError("Model must be fitted before plotting components")
        
        try:
            # Determine period for seasonal decomposition
            if self.seasonal_order is not None:
                period = self.seasonal_order[3]
            else:
                # Default period based on data frequency
                period = 12  # Monthly data (common default)
            
            # Need enough data for decomposition
            if len(self.original_data) < period * 2:
                logger.warning(f"Not enough data for decomposition. Need at least {period * 2} points.")
                return None
            
            # Create time series with datetime index if possible
            if hasattr(self, 'date_index_') and self.date_index_ is not None:
                ts = pd.Series(self.original_data.values, index=self.date_index_)
            else:
                ts = pd.Series(self.original_data.values)
            
            # Perform decomposition
            decomposition = seasonal_decompose(ts, model='additive', period=period)
            
            # Plot components
            fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
            
            # Original Series
            axes[0].plot(ts)
            axes[0].set_title('Original Series')
            
            # Trend Component
            axes[1].plot(decomposition.trend)
            axes[1].set_title('Trend Component')
            
            # Seasonal Component
            axes[2].plot(decomposition.seasonal)
            axes[2].set_title('Seasonal Component')
            
            # Residual Component
            axes[3].plot(decomposition.resid)
            axes[3].set_title('Residual Component')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error in decomposition: {e}")
            return None


def check_stationarity(series: pd.Series, alpha: float = 0.05) -> Dict[str, Any]:
    """
    Check if a time series is stationary using the Augmented Dickey-Fuller test.
    
    Args:
        series: Time series data
        alpha: Significance level
        
    Returns:
        Dictionary with test results
    """
    # Handle missing values
    series = series.dropna()
    
    if len(series) < 20:
        logger.warning("Series too short for reliable stationarity test")
        return {
            'stationary': None,
            'p_value': None,
            'critical_values': None,
            'test_statistic': None,
            'error': 'Series too short'
        }
    
    try:
        # Perform ADF test
        result = adfuller(series)
        
        # Extract results
        test_statistic = result[0]
        p_value = result[1]
        critical_values = result[4]
        
        # Determine if stationary
        stationary = p_value < alpha
        
        return {
            'stationary': stationary,
            'p_value': p_value,
            'critical_values': critical_values,
            'test_statistic': test_statistic
        }
        
    except Exception as e:
        logger.error(f"Error in stationarity check: {e}")
        return {
            'stationary': None,
            'p_value': None,
            'critical_values': None,
            'test_statistic': None,
            'error': str(e)
        }


def apply_differencing(series: pd.Series, d: int = 1, D: int = 0, s: int = 12) -> Tuple[pd.Series, List]:
    """
    Apply differencing to a time series to achieve stationarity.
    
    Args:
        series: Time series data
        d: Regular differencing order
        D: Seasonal differencing order
        s: Seasonal period
        
    Returns:
        Tuple of (differenced series, original values for inversion)
    """
    # Handle missing values
    series = series.interpolate().dropna()
    
    # Store original values for later inversion
    original_values = []
    
    # Seasonal differencing first (if specified)
    if D > 0 and s > 1:
        # Store values needed for inversion
        for i in range(D):
            seasonal_values = series.iloc[:s * (i + 1)].tolist()
            original_values.append(('seasonal', seasonal_values, s))
        
        # Apply seasonal differencing
        for i in range(D):
            series = series.diff(s).dropna()
    
    # Regular differencing
    if d > 0:
        # Store values needed for inversion
        for i in range(d):
            diff_values = series.iloc[:i + 1].tolist()
            original_values.append(('regular', diff_values, 1))
        
        # Apply regular differencing
        for i in range(d):
            series = series.diff().dropna()
    
    return series, original_values


def invert_differencing(differenced_series: pd.Series, original_values: List) -> pd.Series:
    """
    Invert differencing to recover the original time series.
    
    Args:
        differenced_series: Differenced time series
        original_values: Original values stored during differencing
        
    Returns:
        Original time series
    """
    result = differenced_series.copy()
    
    # Invert in reverse order
    for diff_type, values, period in reversed(original_values):
        if diff_type == 'regular':
            # Invert regular differencing
            result = result.cumsum()
            # Add initial value
            if len(values) > 0:
                result = result + values[0]
        elif diff_type == 'seasonal':
            # Invert seasonal differencing
            # Create a working copy with enough space for results
            temp = pd.Series(np.nan, index=range(len(result) + period))
            
            # Set initial values
            temp.iloc[:period] = values[:period]
            
            # Fill in the rest by inverting the differencing
            for i in range(period, len(temp)):
                if i - period < len(result):
                    temp.iloc[i] = temp.iloc[i - period] + result.iloc[i - period]
            
            # Trim to the right size
            result = temp.iloc[:len(differenced_series) + period]
    
    return result


def determine_arima_orders(series: pd.Series, max_p: int = 5, max_q: int = 5, 
                          max_d: int = 2, alpha: float = 0.05) -> Dict[str, Any]:
    """
    Determine appropriate ARIMA orders based on data characteristics.
    
    Args:
        series: Time series data
        max_p: Maximum autoregressive order to consider
        max_q: Maximum moving average order to consider
        max_d: Maximum differencing order to consider
        alpha: Significance level for tests
        
    Returns:
        Dictionary with recommended orders
    """
    results = {
        'recommended_order': None,
        'differencing_order': 0,
        'ar_order': 0,
        'ma_order': 0,
        'stationarity_result': None
    }
    
    # Check stationarity and determine differencing order (d)
    d = 0
    series_to_check = series.copy()
    
    # Check stationarity of original series
    stationarity_result = check_stationarity(series_to_check, alpha)
    results['stationarity_result'] = stationarity_result
    
    # Apply differencing until stationary or max_d reached
    while (not stationarity_result.get('stationary', False) and 
           d < max_d and stationarity_result.get('error') is None):
        d += 1
        series_to_check = series_to_check.diff().dropna()
        stationarity_result = check_stationarity(series_to_check, alpha)
        
        # Store the last stationarity result
        results[f'stationarity_result_d{d}'] = stationarity_result
    
    results['differencing_order'] = d
    
    # If we have a stationary series, determine p and q
    if stationarity_result.get('stationary', False) or d == max_d:
        # Calculate ACF and PACF
        try:
            # Use stationary series for ACF/PACF
            acf_values = acf(series_to_check.dropna(), nlags=max_p, fft=True)
            pacf_values = pacf(series_to_check.dropna(), nlags=max_q)
            
            results['acf_values'] = acf_values.tolist()
            results['pacf_values'] = pacf_values.tolist()
            
            # Determine p from PACF
            # Significant lags in PACF suggest AR terms
            p = 0
            for i in range(1, len(pacf_values)):
                # Check significance (above confidence interval)
                if abs(pacf_values[i]) > 1.96 / np.sqrt(len(series_to_check)):
                    p = i
                else:
                    # Once we find a non-significant lag, check a few more
                    # and if they're all non-significant, stop
                    if i > p + 2:
                        break
            
            results['ar_order'] = min(p, max_p)
            
            # Determine q from ACF
            # Significant lags in ACF suggest MA terms
            q = 0
            for i in range(1, len(acf_values)):
                # Check significance
                if abs(acf_values[i]) > 1.96 / np.sqrt(len(series_to_check)):
                    q = i
                else:
                    # Once we find a non-significant lag, check a few more
                    if i > q + 2:
                        break
            
            results['ma_order'] = min(q, max_q)
            
            # Recommended order
            results['recommended_order'] = (results['ar_order'], results['differencing_order'], results['ma_order'])
            
        except Exception as e:
            logger.error(f"Error determining ARIMA orders: {e}")
            results['error'] = str(e)
    
    return results


def analyze_acf_pacf(series: pd.Series, nlags: int = 40) -> Dict[str, Any]:
    """
    Analyze ACF and PACF of a time series to identify potential ARIMA orders.
    
    Args:
        series: Time series data
        nlags: Number of lags to consider
        
    Returns:
        Dictionary with ACF/PACF analysis results
    """
    series = series.dropna()
    
    if len(series) < nlags + 1:
        nlags = len(series) - 1
        logger.warning(f"Reduced nlags to {nlags} due to series length")
    
    try:
        # Calculate ACF and PACF
        acf_values = acf(series, nlags=nlags, fft=True)
        pacf_values = pacf(series, nlags=nlags)
        
        # Identify significant lags (exceeding confidence intervals)
        significance_threshold = 1.96 / np.sqrt(len(series))
        
        significant_acf = [i for i in range(1, len(acf_values)) 
                          if abs(acf_values[i]) > significance_threshold]
        
        significant_pacf = [i for i in range(1, len(pacf_values)) 
                           if abs(pacf_values[i]) > significance_threshold]
        
        # Check for seasonal patterns in ACF/PACF
        potential_seasonal_periods = []
        for period in [3, 4, 6, 12]:
            if period in significant_acf or period in significant_pacf:
                potential_seasonal_periods.append(period)
        
        # Suggest ARIMA orders based on patterns
        # Decaying ACF, significant PACF at early lags: AR model
        # Decaying PACF, significant ACF at early lags: MA model
        # Both show extended significant values: ARMA model
        
        result = {
            'acf_values': acf_values.tolist(),
            'pacf_values': pacf_values.tolist(),
            'significant_acf': significant_acf,
            'significant_pacf': significant_pacf,
            'potential_seasonal_periods': potential_seasonal_periods,
            'significance_threshold': significance_threshold
        }
        
        # Suggest model type
        if len(significant_acf) <= 2 and len(significant_pacf) > 2:
            result['suggested_model'] = 'AR'
            result['suggested_p'] = max(significant_pacf[:3]) if significant_pacf else 0
        elif len(significant_pacf) <= 2 and len(significant_acf) > 2:
            result['suggested_model'] = 'MA'
            result['suggested_q'] = max(significant_acf[:3]) if significant_acf else 0
        else:
            result['suggested_model'] = 'ARMA'
            result['suggested_p'] = max(significant_pacf[:3]) if significant_pacf else 0
            result['suggested_q'] = max(significant_acf[:3]) if significant_acf else 0
        
        return result
        
    except Exception as e:
        logger.error(f"Error in ACF/PACF analysis: {e}")
        return {'error': str(e)}