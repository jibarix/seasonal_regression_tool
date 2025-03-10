"""
SARIMA model implementation module.
Provides concrete SARIMA model class.
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Union, Optional, Tuple, Any
from datetime import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.eval_measures import rmse, aic

# Import from project modules
from arima_base import BaseARIMAModel, check_stationarity, apply_differencing, invert_differencing

# Setup logging
logger = logging.getLogger(__name__)


class SARIMAModel(BaseARIMAModel):
    """
    SARIMA (Seasonal ARIMA) model implementation.
    """
    
    def __init__(self, name: str = None, date_col: str = 'date',
                 order: Tuple[int, int, int] = (1, 1, 1),
                 seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 12),
                 trend: Optional[str] = None,
                 enforce_stationarity: bool = True,
                 enforce_invertibility: bool = True,
                 **kwargs):
        """
        Initialize the SARIMA model.
        
        Args:
            name: Model name
            date_col: Name of the date column
            order: ARIMA order (p, d, q) - autoregressive, differencing, moving average
            seasonal_order: Seasonal ARIMA order (P, D, Q, s) - seasonal AR, differencing, MA, period
            trend: Trend term (None, 'c', 't', 'ct')
            enforce_stationarity: Whether to enforce stationarity in AR parameters
            enforce_invertibility: Whether to enforce invertibility in MA parameters
            **kwargs: Additional parameters passed to BaseARIMAModel
        """
        super().__init__(
            name=name or f"SARIMA{order}x{seasonal_order}",
            date_col=date_col,
            order=order,
            seasonal_order=seasonal_order,
            **kwargs
        )
        
        self.trend = trend
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        
        # Update model parameters
        self.model_params.update({
            'trend': trend,
            'enforce_stationarity': enforce_stationarity,
            'enforce_invertibility': enforce_invertibility
        })
        
        # Initialize model-specific attributes
        self.original_index = None
        self.date_index_ = None
        self.differenced_data = None
        self.original_values_for_inversion = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'SARIMAModel':
        """
        Fit the SARIMA model to the training data.
        
        Args:
            X: Feature matrix
            y: Target variable
            **kwargs: Additional model-specific parameters
            
        Returns:
            Self for method chaining
        """
        # Preprocess data
        processed_X, processed_y = self.preprocess(X, y, fit=True)
        
        # Extract date information if available
        if self.date_col in processed_X.columns:
            self.date_index_ = pd.to_datetime(processed_X[self.date_col])
            self.original_index = processed_y.index
        
        # Get order parameters
        p, d, q = self.order
        P, D, Q, s = self.seasonal_order
        
        # Apply differencing manually if specified
        # This can help with stability in some cases
        manual_diff = kwargs.pop('manual_differencing', False)
        
        if manual_diff:
            # Apply regular and seasonal differencing
            differenced_y, original_values = apply_differencing(
                processed_y, d=d, D=D, s=s
            )
            self.differenced_data = differenced_y
            self.original_values_for_inversion = original_values
            
            # Reset orders since we've already differenced
            d_model, D_model = 0, 0
        else:
            differenced_y = processed_y
            self.differenced_data = differenced_y
            d_model, D_model = d, D
        
        # Fit SARIMA model
        try:
            # Create and fit the statsmodels SARIMAX model
            sarima_model = SARIMAX(
                differenced_y,
                order=(p, d_model, q),
                seasonal_order=(P, D_model, Q, s),
                trend=self.trend,
                enforce_stationarity=self.enforce_stationarity,
                enforce_invertibility=self.enforce_invertibility
            )
            
            self.model = sarima_model.fit(**kwargs)
            self.results = self.model
            
            # Store residuals
            self.residuals = self.model.resid
            
            # Store fitted values (need to invert differencing if manually applied)
            if manual_diff and self.original_values_for_inversion:
                self.fitted_values = invert_differencing(
                    self.model.fittedvalues,
                    self.original_values_for_inversion
                )
            else:
                self.fitted_values = self.model.fittedvalues
            
            # Set fitted flag
            self.is_fitted = True
            self.metadata['updated_at'] = datetime.now().isoformat()
            
            # Log model summary
            logger.info(f"Fitted SARIMA{self.order}x{self.seasonal_order} model")
            
            return self
            
        except Exception as e:
            logger.error(f"Error fitting SARIMA model: {e}")
            raise
    
    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Generate predictions using the fitted model.
        
        Args:
            X: Feature matrix
            **kwargs: Additional prediction parameters
            
        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Preprocess data
        processed_X, _ = self.preprocess(X, fit=False)
        
        # Extract prediction dates if available
        if self.date_col in processed_X.columns:
            pred_dates = pd.to_datetime(processed_X[self.date_col])
            
            # Get forecast horizon (number of periods after the last training date)
            if self.date_index_ is not None:
                last_train_date = self.date_index_.max()
                future_dates = pred_dates[pred_dates > last_train_date]
                
                if len(future_dates) > 0:
                    # Need to forecast
                    steps = len(future_dates)
                    forecasts = self.forecast(steps, X=processed_X)
                    
                    if len(forecasts) == len(processed_X):
                        return forecasts
                    
                    # Combine in-sample predictions with forecasts
                    in_sample_indices = pred_dates <= last_train_date
                    
                    if sum(in_sample_indices) > 0:
                        in_sample_preds = self.fitted_values.reindex(
                            self.original_index[processed_X.index[in_sample_indices]]
                        ).values
                        
                        all_preds = np.zeros(len(processed_X))
                        all_preds[in_sample_indices] = in_sample_preds
                        all_preds[~in_sample_indices] = forecasts
                        
                        return all_preds
                    
                    return forecasts
        
        # Default behavior: use fitted values for dates in training data
        # and forecast for others
        if self.fitted_values is not None:
            return self.fitted_values.values
        
        # If all else fails, just return zeros
        logger.warning("Could not generate predictions, returning zeros")
        return np.zeros(len(processed_X))
    
    def forecast(self, steps: int, X: Optional[pd.DataFrame] = None, 
                return_conf_int: bool = False, alpha: float = 0.05) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Generate forecasts for future periods.
        
        Args:
            steps: Number of steps ahead to forecast
            X: Future feature matrix (not used in pure SARIMA, but kept for interface consistency)
            return_conf_int: Whether to return confidence intervals
            alpha: Significance level for confidence intervals
            
        Returns:
            Forecasts and optionally confidence intervals
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        try:
            # Generate forecasts
            forecast_result = self.model.forecast(steps=steps)
            
            # Check if we need to invert manual differencing
            p, d, q = self.order
            P, D, Q, s = self.seasonal_order
            
            manual_diff = getattr(self, 'original_values_for_inversion', None) is not None
            
            if manual_diff:
                forecasts = invert_differencing(
                    forecast_result,
                    self.original_values_for_inversion
                )
            else:
                forecasts = forecast_result
            
            if return_conf_int:
                # Get confidence intervals
                pred_results = self.model.get_forecast(steps=steps)
                conf_int = pred_results.conf_int(alpha=alpha)
                
                lower_bounds = conf_int.iloc[:, 0].values
                upper_bounds = conf_int.iloc[:, 1].values
                
                # Invert differencing for confidence intervals if needed
                if manual_diff:
                    lower_bounds = invert_differencing(
                        pd.Series(lower_bounds),
                        self.original_values_for_inversion
                    ).values
                    
                    upper_bounds = invert_differencing(
                        pd.Series(upper_bounds),
                        self.original_values_for_inversion
                    ).values
                
                return forecasts.values, lower_bounds, upper_bounds
            
            return forecasts.values
            
        except Exception as e:
            logger.error(f"Error in SARIMA forecast: {e}")
            
            # Return zeros as fallback
            if return_conf_int:
                return np.zeros(steps), np.zeros(steps), np.zeros(steps)
            
            return np.zeros(steps)
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the fitted model.
        
        Returns:
            Dictionary with model information
        """
        if not self.is_fitted:
            return {'error': 'Model not fitted'}
        
        info = {
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'trend': self.trend,
            'aic': self.model.aic if hasattr(self.model, 'aic') else None,
            'bic': self.model.bic if hasattr(self.model, 'bic') else None,
            'params': self.model.params.to_dict() if hasattr(self.model.params, 'to_dict') else None
        }
        
        # Add summary if available
        if hasattr(self.model, 'summary'):
            try:
                summary_html = self.model.summary().as_html()
                info['summary_html'] = summary_html
            except:
                pass
        
        return info
    
    def plot_forecast(self, steps: int = 12, figsize: Tuple[int, int] = (10, 6),
                     alpha: float = 0.05, include_history: bool = True) -> plt.Figure:
        """
        Plot forecasts with confidence intervals.
        
        Args:
            steps: Number of steps to forecast
            figsize: Figure size
            alpha: Significance level for confidence intervals
            include_history: Whether to include historical data
            
        Returns:
            Matplotlib figure
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting forecasts")
        
        # Generate forecasts with confidence intervals
        forecasts, lower_bounds, upper_bounds = self.forecast(
            steps=steps, return_conf_int=True, alpha=alpha
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get the last date if available
        if self.date_index_ is not None:
            last_date = self.date_index_.max()
            
            # Get period from seasonal order
            _, _, _, s = self.seasonal_order
            
            if s == 12:  # Monthly data
                freq = 'MS'
            elif s == 4:  # Quarterly data
                freq = 'QS'
            else:
                freq = 'MS'  # Default to monthly
            
            forecast_dates = pd.date_range(
                start=last_date,
                periods=steps + 1,
                freq=freq
            )[1:]
            
            x_forecast = forecast_dates
            
            if include_history:
                # Plot historical data
                ax.plot(self.date_index_, self.original_data, 'b-', label='Historical')
                
                # Plot fitted values
                if self.fitted_values is not None:
                    ax.plot(self.date_index_, self.fitted_values, 'g-', label='Fitted')
        else:
            # Use indices if dates not available
            x_forecast = np.arange(len(self.original_data), len(self.original_data) + steps)
            
            if include_history:
                # Plot historical data
                ax.plot(np.arange(len(self.original_data)), self.original_data, 'b-', label='Historical')
                
                # Plot fitted values
                if self.fitted_values is not None:
                    ax.plot(np.arange(len(self.fitted_values)), self.fitted_values, 'g-', label='Fitted')
        
        # Plot forecasts
        ax.plot(x_forecast, forecasts, 'r-', label='Forecast')
        
        # Plot confidence intervals
        ax.fill_between(
            x_forecast,
            lower_bounds,
            upper_bounds,
            color='pink',
            alpha=0.3,
            label=f'{(1-alpha)*100}% Confidence Interval'
        )
        
        # Add labels and legend
        ax.set_title(f'SARIMA{self.order}x{self.seasonal_order} Forecast')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.legend()
        
        # Format x-axis if dates are available
        if self.date_index_ is not None:
            fig.autofmt_xdate()
        
        return fig


def create_sarima_model(y: pd.Series, X: Optional[pd.DataFrame] = None,
                       max_p: int = 2, max_d: int = 1, max_q: int = 2,
                       max_P: int = 1, max_D: int = 1, max_Q: int = 1,
                       m: int = 12,
                       information_criterion: str = 'aic') -> Tuple[SARIMAModel, Dict[str, Any]]:
    """
    Create and select the best SARIMA model based on information criterion.
    
    Args:
        y: Time series data
        X: Optional feature matrix with date column
        max_p, max_d, max_q: Maximum orders for ARIMA
        max_P, max_D, max_Q: Maximum seasonal orders
        m: Seasonal period
        information_criterion: Criterion for model selection ('aic', 'bic')
        
    Returns:
        Tuple of (best model, results dictionary)
    """
    # Check stationarity and apply differencing if needed
    stationarity_check = check_stationarity(y)
    
    if not stationarity_check.get('stationary', False):
        # Apply one level of differencing
        differenced_y, _ = apply_differencing(y, d=1)
        logger.info("Applied first-order differencing")
        
        # Check stationarity again
        stationarity_check_diff = check_stationarity(differenced_y)
        
        if not stationarity_check_diff.get('stationary', False) and max_d >= 2:
            # Apply second level of differencing
            differenced_y, _ = apply_differencing(differenced_y, d=1)
            logger.info("Applied second-order differencing")
            d_recommended = 2
        else:
            d_recommended = 1
    else:
        d_recommended = 0
        logger.info("Series is already stationary")
    
    # Results dictionary
    results = {
        'stationarity_check': stationarity_check,
        'recommended_d': d_recommended,
        'models_tried': [],
        'best_model': None
    }
    
    # Define date_col if X is provided
    date_col = None
    if X is not None:
        # Find first column with 'date' in the name
        for col in X.columns:
            if 'date' in col.lower():
                date_col = col
                break
    
    best_model = None
    best_criterion = float('inf')
    
    # Search grid for seasonal orders
    # We use a small grid to keep computational complexity reasonable
    seasonal_configurations = [
        (0, 0, 0, m),   # No seasonal component
        (1, 0, 0, m),   # Seasonal AR(1)
        (0, 1, 0, m),   # Seasonal I(1)
        (0, 0, 1, m),   # Seasonal MA(1)
        (1, 1, 0, m),   # Seasonal AR(1)I(1)
        (0, 1, 1, m),   # Seasonal I(1)MA(1)
        (1, 0, 1, m),   # Seasonal ARMA(1,1)
        (1, 1, 1, m)    # Full seasonal model
    ]
    
    # Try different non-seasonal parameters
    for p in range(max_p + 1):
        for d in range(min(d_recommended + 1, max_d + 1)):
            for q in range(max_q + 1):
                # Try selected seasonal configurations
                for P, D, Q, s in seasonal_configurations:
                    # Skip if all orders are 0
                    if p == 0 and d == 0 and q == 0 and P == 0 and D == 0 and Q == 0:
                        continue
                    
                    try:
                        # Create and fit SARIMA model
                        model = SARIMAModel(
                            date_col=date_col,
                            order=(p, d, q),
                            seasonal_order=(P, D, Q, s)
                        )
                        
                        if X is not None:
                            model.fit(X, y)
                        else:
                            # Create a dummy dataframe with date column if X not provided
                            if isinstance(y.index, pd.DatetimeIndex):
                                dummy_X = pd.DataFrame({
                                    'date': y.index
                                })
                                model.fit(dummy_X, y)
                            else:
                                # Create a range index if no dates available
                                dummy_X = pd.DataFrame({
                                    'dummy': range(len(y))
                                })
                                model.fit(dummy_X, y)
                        
                        # Get criterion value
                        if information_criterion == 'aic':
                            criterion_value = model.model.aic
                        else:  # 'bic'
                            criterion_value = model.model.bic
                        
                        # Track models tried
                        results['models_tried'].append({
                            'order': (p, d, q),
                            'seasonal_order': (P, D, Q, s),
                            information_criterion: criterion_value
                        })
                        
                        # Update best model if this one is better
                        if criterion_value < best_criterion:
                            best_criterion = criterion_value
                            best_model = model
                            results['best_model'] = {
                                'order': (p, d, q),
                                'seasonal_order': (P, D, Q, s),
                                information_criterion: criterion_value
                            }
                        
                    except Exception as e:
                        logger.warning(f"Error fitting SARIMA({p},{d},{q})({P},{D},{Q},{s}): {e}")
                        # Continue with next model
                        continue
    
    if best_model is None:
        logger.warning("Could not find suitable SARIMA model")
        # Create a simple seasonal model as fallback
        try:
            best_model = SARIMAModel(
                date_col=date_col,
                order=(1, d_recommended, 0),
                seasonal_order=(0, 1, 0, m)
            )
            
            if X is not None:
                best_model.fit(X, y)
            else:
                if isinstance(y.index, pd.DatetimeIndex):
                    dummy_X = pd.DataFrame({'date': y.index})
                    best_model.fit(dummy_X, y)
                else:
                    dummy_X = pd.DataFrame({'dummy': range(len(y))})
                    best_model.fit(dummy_X, y)
                    
            results['best_model'] = {
                'order': (1, d_recommended, 0),
                'seasonal_order': (0, 1, 0, m),
                'note': 'Fallback model due to fitting issues'
            }
        except Exception as e:
            logger.error(f"Error fitting fallback model: {e}")
            raise ValueError("Could not fit any SARIMA model to the data")
    
    return best_model, results


def seasonal_period_detection(y: pd.Series) -> int:
    """
    Detect the seasonal period from a time series.
    
    Args:
        y: Time series data
        
    Returns:
        Detected seasonal period (commonly 4, 12, or 52)
    """
    # Handle missing values
    y = y.dropna()
    
    if len(y) < 24:
        logger.warning("Series too short for reliable seasonal detection")
        return 12  # Default to monthly
    
    try:
        # Check if index is DatetimeIndex
        if isinstance(y.index, pd.DatetimeIndex):
            # Check frequency
            if y.index.freq is not None:
                freq = y.index.freq.freqstr
                
                if 'D' in freq:  # Daily
                    return 7  # Weekly seasonality
                elif 'W' in freq:  # Weekly
                    return 52  # Yearly seasonality
                elif 'M' in freq:  # Monthly
                    return 12  # Yearly seasonality
                elif 'Q' in freq:  # Quarterly
                    return 4  # Yearly seasonality
            else:
                # Try to infer frequency from dates
                dates_diff = y.index.to_series().diff().dropna()
                median_days = dates_diff.dt.days.median()
                
                if median_days < 2:  # Daily or higher frequency
                    return 7  # Weekly
                elif 25 <= median_days <= 32:  # Monthly
                    return 12  # Yearly
                elif 85 <= median_days <= 95:  # Quarterly
                    return 4  # Yearly
        
        # Use ACF to detect seasonality
        from statsmodels.tsa.stattools import acf
        
        n = len(y)
        nlags = min(n - 1, 366)  # Maximum lag to consider
        acf_values = acf(y, nlags=nlags, fft=True)
        
        # Find peaks in ACF
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(acf_values, height=0.1)
        
        if len(peaks) > 1:
            # Look for common seasonal periods
            for period in [12, 4, 52, 7]:
                if period in peaks or period + 1 in peaks or period - 1 in peaks:
                    return period
            
            # If no common period found, return the first significant peak
            if len(peaks) > 0:
                return peaks[0]
    
    except Exception as e:
        logger.error(f"Error in seasonal period detection: {e}")
    
    # Default to monthly seasonality
    return 12