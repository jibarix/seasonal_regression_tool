"""
ARIMAX model implementation module.
Provides ARIMA models with exogenous variables.
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Union, Optional, Tuple, Any
from datetime import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.eval_measures import rmse, aic

# Import from project modules
from arima_base import BaseARIMAModel, check_stationarity, apply_differencing, invert_differencing

# Setup logging
logger = logging.getLogger(__name__)


class ARIMAXModel(BaseARIMAModel):
    """
    ARIMAX (ARIMA with exogenous variables) model implementation.
    """
    
    def __init__(self, name: str = None, date_col: str = 'date',
                 order: Tuple[int, int, int] = (1, 1, 1),
                 exog_columns: Optional[List[str]] = None,
                 trend: Optional[str] = None,
                 enforce_stationarity: bool = True,
                 enforce_invertibility: bool = True,
                 **kwargs):
        """
        Initialize the ARIMAX model.
        
        Args:
            name: Model name
            date_col: Name of the date column
            order: ARIMA order (p, d, q) - autoregressive, differencing, moving average
            exog_columns: List of column names to use as exogenous variables
            trend: Trend term (None, 'c', 't', 'ct')
            enforce_stationarity: Whether to enforce stationarity in AR parameters
            enforce_invertibility: Whether to enforce invertibility in MA parameters
            **kwargs: Additional parameters passed to BaseARIMAModel
        """
        super().__init__(
            name=name or f"ARIMAX{order}",
            date_col=date_col,
            order=order,
            **kwargs
        )
        
        self.exog_columns = exog_columns or []
        self.trend = trend
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        
        # Update model parameters
        self.model_params.update({
            'exog_columns': exog_columns,
            'trend': trend,
            'enforce_stationarity': enforce_stationarity,
            'enforce_invertibility': enforce_invertibility
        })
        
        # Initialize model-specific attributes
        self.original_index = None
        self.date_index_ = None
        self.differenced_data = None
        self.original_values_for_inversion = None
        self.exog_scaler = None
        self.exog_train = None
    
    def preprocess(self, X: pd.DataFrame, y: Optional[pd.Series] = None, 
                  fit: bool = False) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Preprocess data before model fitting or prediction.
        Extract exogenous variables.
        
        Args:
            X: Feature matrix
            y: Target variable (optional)
            fit: Whether this is for fitting (True) or prediction (False)
            
        Returns:
            Tuple of (processed_X, processed_y)
        """
        # First apply standard preprocessing from BaseARIMAModel
        processed_X, processed_y = super().preprocess(X, y, fit)
        
        # Extract and standardize exogenous variables
        if self.exog_columns:
            # Check if all exogenous columns are in X
            missing_cols = [col for col in self.exog_columns if col not in processed_X.columns]
            if missing_cols:
                logger.warning(f"Exogenous columns not found in data: {missing_cols}")
                # Remove missing columns from exog_columns
                self.exog_columns = [col for col in self.exog_columns if col in processed_X.columns]
        
        return processed_X, processed_y
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'ARIMAXModel':
        """
        Fit the ARIMAX model to the training data.
        
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
        
        # Extract exogenous variables
        exog = None
        if self.exog_columns:
            exog = processed_X[self.exog_columns].values
            self.exog_train = exog
        
        # Get order parameters
        p, d, q = self.order
        
        # Apply differencing if needed
        if d > 0:
            differenced_y, original_values = apply_differencing(processed_y, d=d)
            self.differenced_data = differenced_y
            self.original_values_for_inversion = original_values
        else:
            differenced_y = processed_y
            self.differenced_data = differenced_y
        
        # Fit ARIMAX model
        try:
            # Create and fit the statsmodels SARIMAX model (used for both ARIMAX)
            arimax_model = SARIMAX(
                differenced_y,
                exog=exog,
                order=(p, 0, q),  # We manually handled differencing
                trend=self.trend,
                enforce_stationarity=self.enforce_stationarity,
                enforce_invertibility=self.enforce_invertibility
            )
            
            self.model = arimax_model.fit(**kwargs)
            self.results = self.model
            
            # Store residuals
            self.residuals = self.model.resid
            
            # Store fitted values (need to invert differencing)
            if d > 0 and self.original_values_for_inversion:
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
            logger.info(f"Fitted ARIMAX{self.order} model with {len(self.exog_columns)} exog variables")
            
            return self
            
        except Exception as e:
            logger.error(f"Error fitting ARIMAX model: {e}")
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
        
        # Extract exogenous variables
        exog = None
        if self.exog_columns:
            exog = processed_X[self.exog_columns].values
        
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
                    future_exog = exog[-steps:] if exog is not None else None
                    forecasts = self.forecast(steps, exog=future_exog)
                    
                    if len(forecasts) == len(processed_X):
                        return forecasts
                    
                    # Combine in-sample predictions with forecasts
                    in_sample_indices = pred_dates <= last_train_date
                    
                    if sum(in_sample_indices) > 0:
                        # For in-sample predictions, use the model's predict method
                        in_sample_exog = exog[in_sample_indices] if exog is not None else None
                        in_sample_preds = self.model.predict(exog=in_sample_exog)
                        
                        # Invert differencing if needed
                        p, d, q = self.order
                        if d > 0 and self.original_values_for_inversion:
                            in_sample_preds = invert_differencing(
                                in_sample_preds,
                                self.original_values_for_inversion
                            )
                        
                        # Combine in-sample and forecasts
                        all_preds = np.zeros(len(processed_X))
                        all_preds[in_sample_indices] = in_sample_preds
                        all_preds[~in_sample_indices] = forecasts
                        
                        return all_preds
                    
                    return forecasts
        
        # If no date column or all dates are in the training set,
        # use the model's predict method with all exogenous variables
        try:
            predictions = self.model.predict(exog=exog)
            
            # Invert differencing if needed
            p, d, q = self.order
            if d > 0 and self.original_values_for_inversion:
                predictions = invert_differencing(
                    predictions,
                    self.original_values_for_inversion
                )
                
            return predictions
            
        except Exception as e:
            logger.error(f"Error in ARIMAX prediction: {e}")
            
            # If all else fails, return fitted values or zeros
            if self.fitted_values is not None and len(self.fitted_values) == len(processed_X):
                return self.fitted_values.values
            
            logger.warning("Could not generate predictions, returning zeros")
            return np.zeros(len(processed_X))
    
    def forecast(self, steps: int, exog: Optional[np.ndarray] = None, 
                return_conf_int: bool = False, alpha: float = 0.05) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Generate forecasts for future periods.
        
        Args:
            steps: Number of steps ahead to forecast
            exog: Future exogenous variables (required if model was fitted with exogenous variables)
            return_conf_int: Whether to return confidence intervals
            alpha: Significance level for confidence intervals
            
        Returns:
            Forecasts and optionally confidence intervals
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        # Check if exogenous variables are needed but not provided
        if self.exog_columns and exog is None:
            logger.warning("Model was fitted with exogenous variables, but none provided for forecast")
            # Try to use zeros as fallback
            exog = np.zeros((steps, len(self.exog_columns)))
        
        try:
            # Generate forecasts
            forecast_result = self.model.forecast(steps=steps, exog=exog)
            
            # If we applied manual differencing, we need to invert it
            p, d, q = self.order
            if d > 0 and self.original_values_for_inversion:
                forecasts = invert_differencing(
                    forecast_result,
                    self.original_values_for_inversion
                )
            else:
                forecasts = forecast_result
            
            if return_conf_int:
                # Get confidence intervals
                pred_results = self.model.get_forecast(steps=steps, exog=exog)
                conf_int = pred_results.conf_int(alpha=alpha)
                
                lower_bounds = conf_int.iloc[:, 0].values
                upper_bounds = conf_int.iloc[:, 1].values
                
                # If we applied manual differencing, invert it for confidence intervals too
                if d > 0 and self.original_values_for_inversion:
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
            logger.error(f"Error in ARIMAX forecast: {e}")
            
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
            'exog_columns': self.exog_columns,
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
    
    def plot_forecast(self, steps: int = 12, exog: Optional[np.ndarray] = None,
                     figsize: Tuple[int, int] = (10, 6),
                     alpha: float = 0.05, include_history: bool = True) -> plt.Figure:
        """
        Plot forecasts with confidence intervals.
        
        Args:
            steps: Number of steps to forecast
            exog: Future exogenous variables (required if model was fitted with exogenous variables)
            figsize: Figure size
            alpha: Significance level for confidence intervals
            include_history: Whether to include historical data
            
        Returns:
            Matplotlib figure
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting forecasts")
        
        # Check if exogenous variables are needed but not provided
        if self.exog_columns and exog is None:
            logger.warning("Model was fitted with exogenous variables, but none provided for forecast")
            # Try to use zeros as fallback
            exog = np.zeros((steps, len(self.exog_columns)))
        
        # Generate forecasts with confidence intervals
        forecasts, lower_bounds, upper_bounds = self.forecast(
            steps=steps, exog=exog, return_conf_int=True, alpha=alpha
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get the last date if available
        if self.date_index_ is not None:
            last_date = self.date_index_.max()
            forecast_dates = pd.date_range(
                start=last_date,
                periods=steps + 1,
                freq='MS'  # Assuming monthly data, should be parameterized
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
        ax.set_title(f'ARIMAX{self.order} Forecast')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.legend()
        
        # Format x-axis if dates are available
        if self.date_index_ is not None:
            fig.autofmt_xdate()
        
        return fig


def select_exogenous_variables(X: pd.DataFrame, y: pd.Series, 
                              max_variables: int = 5,
                              date_col: str = 'date',
                              method: str = 'correlation') -> List[str]:
    """
    Select the most relevant exogenous variables for an ARIMAX model.
    
    Args:
        X: Feature matrix
        y: Target variable
        max_variables: Maximum number of variables to select
        date_col: Name of the date column
        method: Selection method ('correlation', 'granger', 'lasso')
        
    Returns:
        List of selected variable names
    """
    # Create a copy of X to avoid modifying the original
    X_copy = X.copy()
    
    # Remove date column if present
    if date_col in X_copy.columns:
        X_copy = X_copy.drop(columns=[date_col])
    
    # Handle missing values
    X_copy = X_copy.select_dtypes(include=np.number).dropna(axis=1)
    
    # Empty result if no numeric columns
    if X_copy.empty:
        logger.warning("No numeric columns found after removing missing values")
        return []
    
    # Method 1: Correlation-based selection
    if method == 'correlation' or method == 'corr':
        # Calculate correlation with target
        correlations = X_copy.corrwith(y).abs()
        
        # Sort by correlation and select top variables
        selected_vars = correlations.sort_values(ascending=False).head(max_variables).index.tolist()
        
        return selected_vars
    
    # Method 2: Granger causality test
    elif method == 'granger':
        from statsmodels.tsa.stattools import grangercausalitytests
        
        # Minimum p-value for each feature
        min_p_values = {}
        
        for col in X_copy.columns:
            # Prepare data
            data = pd.concat([y, X_copy[col]], axis=1).dropna()
            
            if len(data) < 10:
                logger.warning(f"Not enough data for Granger causality test for {col}")
                continue
            
            try:
                # Test for Granger causality at different lags
                max_lag = min(12, len(data) // 5)  # Avoid too many lags
                test_result = grangercausalitytests(data, maxlag=max_lag, verbose=False)
                
                # Get minimum p-value across all lags
                p_values = [result[0]['ssr_ftest'][1] for lag, result in test_result.items()]
                min_p_values[col] = min(p_values)
                
            except Exception as e:
                logger.warning(f"Error in Granger causality test for {col}: {e}")
                continue
        
        # Sort by p-value (ascending) and select top variables
        selected_vars = sorted(min_p_values.items(), key=lambda x: x[1])
        selected_vars = [var for var, p_val in selected_vars[:max_variables] if p_val < 0.05]
        
        return selected_vars
    
    # Method 3: LASSO-based selection
    elif method == 'lasso':
        from sklearn.linear_model import LassoCV
        from sklearn.preprocessing import StandardScaler
        
        try:
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_copy)
            
            # Fit LASSO with cross-validation
            lasso = LassoCV(cv=5, random_state=42)
            lasso.fit(X_scaled, y)
            
            # Get feature importances (coefficients)
            coef = pd.Series(lasso.coef_, index=X_copy.columns)
            
            # Select non-zero coefficients
            selected_vars = coef[coef != 0].abs().sort_values(ascending=False).head(max_variables).index.tolist()
            
            return selected_vars
            
        except Exception as e:
            logger.warning(f"Error in LASSO selection: {e}")
            # Fall back to correlation method
            return select_exogenous_variables(X, y, max_variables, date_col, 'correlation')
    
    else:
        logger.warning(f"Unknown selection method: {method}, using correlation instead")
        return select_exogenous_variables(X, y, max_variables, date_col, 'correlation')


def create_arimax_model(y: pd.Series, X: pd.DataFrame,
                       exog_columns: Optional[List[str]] = None,
                       max_p: int = 5, max_d: int = 2, max_q: int = 5,
                       information_criterion: str = 'aic') -> Tuple[ARIMAXModel, Dict[str, Any]]:
    """
    Create and select the best ARIMAX model based on information criterion.
    
    Args:
        y: Time series data
        X: Feature matrix
        exog_columns: Columns to use as exogenous variables (if None, will try to select)
        max_p, max_d, max_q: Maximum orders for ARIMA
        information_criterion: Criterion for model selection ('aic', 'bic')
        
    Returns:
        Tuple of (best model, results dictionary)
    """
    # Define date_col
    date_col = None
    for col in X.columns:
        if 'date' in col.lower():
            date_col = col
            break
    
    # If no exogenous columns provided, try to select them
    if exog_columns is None:
        exog_columns = select_exogenous_variables(X, y, max_variables=5, date_col=date_col)
        logger.info(f"Selected exogenous variables: {exog_columns}")
    
    # Check if X actually has the specified exogenous columns
    missing_cols = [col for col in exog_columns if col not in X.columns]
    if missing_cols:
        logger.warning(f"Exogenous columns not found in data: {missing_cols}")
        # Remove missing columns
        exog_columns = [col for col in exog_columns if col in X.columns]
    
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
        'exog_columns': exog_columns,
        'models_tried': [],
        'best_model': None
    }
    
    best_model = None
    best_criterion = float('inf')
    
    # Try different model orders
    # Using a reduced grid for efficiency
    p_values = [0, 1, 2, max(3, max_p // 2), max_p]
    d_values = range(min(d_recommended + 1, max_d + 1))
    q_values = [0, 1, 2, max(3, max_q // 2), max_q]
    
    for p in p_values:
        for d in d_values:
            for q in q_values:
                # Skip if all orders are 0
                if p == 0 and d == 0 and q == 0:
                    continue
                
                try:
                    # Create and fit ARIMAX model
                    model = ARIMAXModel(
                        date_col=date_col,
                        order=(p, d, q),
                        exog_columns=exog_columns
                    )
                    
                    model.fit(X, y)
                    
                    # Get criterion value
                    if information_criterion == 'aic':
                        criterion_value = model.model.aic
                    else:  # 'bic'
                        criterion_value = model.model.bic
                    
                    # Track models tried
                    results['models_tried'].append({
                        'order': (p, d, q),
                        information_criterion: criterion_value
                    })
                    
                    # Update best model if this one is better
                    if criterion_value < best_criterion:
                        best_criterion = criterion_value
                        best_model = model
                        results['best_model'] = {
                            'order': (p, d, q),
                            information_criterion: criterion_value
                        }
                    
                except Exception as e:
                    logger.warning(f"Error fitting ARIMAX({p},{d},{q}): {e}")
                    # Continue with next model
                    continue
    
    if best_model is None:
        logger.warning("Could not find suitable ARIMAX model")
        # Create a simple AR(1) model as fallback
        try:
            best_model = ARIMAXModel(
                date_col=date_col,
                order=(1, d_recommended, 0),
                exog_columns=exog_columns[:1] if exog_columns else []
            )
            
            best_model.fit(X, y)
                    
            results['best_model'] = {
                'order': (1, d_recommended, 0),
                'note': 'Fallback model due to fitting issues'
            }
        except Exception as e:
            logger.error(f"Error fitting fallback model: {e}")
            raise ValueError("Could not fit any ARIMAX model to the data")
    
    return best_model, results