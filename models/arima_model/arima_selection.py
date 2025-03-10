"""
ARIMA model selection module.
Provides automated order selection and model comparison functionality.
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Union, Optional, Tuple, Any
import matplotlib.pyplot as plt
from itertools import product
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.eval_measures import aic, bic, rmse

# Import from project modules
from arima_base import BaseARIMAModel, check_stationarity, apply_differencing
from arima_models import ARIMAModel, create_arima_model
from sarima_models import SARIMAModel, create_sarima_model, seasonal_period_detection
from arimax_models import ARIMAXModel, create_arimax_model, select_exogenous_variables
from sarimax_models import SARIMAXModel, create_sarimax_model

# Setup logging
logger = logging.getLogger(__name__)


def auto_arima(y: pd.Series, X: Optional[pd.DataFrame] = None,
              exog_columns: Optional[List[str]] = None,
              seasonal: bool = True,
              stepwise: bool = True,
              max_p: int = 5, max_d: int = 2, max_q: int = 5,
              max_P: int = 2, max_D: int = 1, max_Q: int = 2,
              m: Optional[int] = None,
              information_criterion: str = 'aic',
              with_exogenous: bool = False,
              return_all_models: bool = False) -> Union[BaseARIMAModel, Tuple[BaseARIMAModel, Dict[str, Any]]]:
    """
    Automatically select best ARIMA model based on information criterion.
    
    Args:
        y: Time series data
        X: Feature matrix (optional, required if with_exogenous=True)
        exog_columns: Columns to use as exogenous variables (if None, will try to select)
        seasonal: Whether to include seasonal components
        stepwise: Whether to use stepwise selection (more efficient but might miss global optimum)
        max_p, max_d, max_q: Maximum orders for ARIMA
        max_P, max_D, max_Q: Maximum orders for seasonal components
        m: Seasonal period (if None, will try to detect automatically)
        information_criterion: Criterion for model selection ('aic', 'bic')
        with_exogenous: Whether to include exogenous variables
        return_all_models: Whether to return all tried models info
        
    Returns:
        Best model or tuple of (best model, results dictionary)
    """
    logger.info(f"Starting automatic ARIMA model selection with parameters: "
                f"seasonal={seasonal}, with_exogenous={with_exogenous}")
    
    # If using exogenous variables, check that X is provided
    if with_exogenous and X is None:
        raise ValueError("X must be provided when with_exogenous=True")
    
    # Detect seasonal period if needed
    if seasonal and m is None:
        m = seasonal_period_detection(y)
        logger.info(f"Detected seasonal period m={m}")
    
    # Create different model types based on parameters
    if seasonal and with_exogenous:
        # SARIMAX model
        model, results = create_sarimax_model(
            y=y, X=X, exog_columns=exog_columns,
            max_p=max_p, max_d=max_d, max_q=max_q,
            max_P=max_P, max_D=max_D, max_Q=max_Q,
            m=m, information_criterion=information_criterion
        )
    elif seasonal and not with_exogenous:
        # SARIMA model
        model, results = create_sarima_model(
            y=y, X=X,  # X might be used for date column
            max_p=max_p, max_d=max_d, max_q=max_q,
            max_P=max_P, max_D=max_D, max_Q=max_Q,
            m=m, information_criterion=information_criterion
        )
    elif not seasonal and with_exogenous:
        # ARIMAX model
        model, results = create_arimax_model(
            y=y, X=X, exog_columns=exog_columns,
            max_p=max_p, max_d=max_d, max_q=max_q,
            information_criterion=information_criterion
        )
    else:
        # ARIMA model
        model, results = create_arima_model(
            y=y, X=X,  # X might be used for date column
            max_p=max_p, max_d=max_d, max_q=max_q,
            information_criterion=information_criterion
        )
    
    # Return results based on return_all_models flag
    if return_all_models:
        return model, results
    else:
        return model


def compute_information_criteria(model: BaseARIMAModel) -> Dict[str, float]:
    """
    Compute various information criteria for model comparison.
    
    Args:
        model: Fitted ARIMA model
        
    Returns:
        Dictionary with information criteria
    """
    if not model.is_fitted:
        raise ValueError("Model must be fitted before computing information criteria")
    
    criteria = {}
    
    # Get the statsmodels results object
    results = model.results
    
    # AIC and BIC (available directly)
    criteria['aic'] = results.aic
    criteria['bic'] = results.bic
    
    # HQIC (Hannan-Quinn Information Criterion)
    if hasattr(results, 'hqic'):
        criteria['hqic'] = results.hqic
    
    # AICc (AIC with correction for small sample sizes)
    n = len(model.original_data)
    k = len(results.params)
    if n > k + 2:
        aicc = results.aic + (2 * k * (k + 1)) / (n - k - 1)
        criteria['aicc'] = aicc
    
    return criteria


def compare_models(models: List[BaseARIMAModel], 
                  test_data: Optional[pd.DataFrame] = None, 
                  test_target: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    Compare multiple ARIMA models using information criteria and optionally test error.
    
    Args:
        models: List of fitted ARIMA models
        test_data: Test feature matrix (optional)
        test_target: Test target series (optional)
        
    Returns:
        DataFrame with model comparison metrics
    """
    # Initialize results dictionary
    results = []
    
    for i, model in enumerate(models):
        if not model.is_fitted:
            logger.warning(f"Model {i} is not fitted, skipping")
            continue
        
        # Get model name and type
        model_name = model.name
        model_type = model.__class__.__name__
        
        # Get model orders
        order = model.order
        seasonal_order = getattr(model, 'seasonal_order', None)
        
        # Get information criteria
        criteria = compute_information_criteria(model)
        
        # Create result dictionary
        result = {
            'model_name': model_name,
            'model_type': model_type,
            'order': str(order),
            'seasonal_order': str(seasonal_order) if seasonal_order else None,
            'aic': criteria.get('aic'),
            'bic': criteria.get('bic'),
            'aicc': criteria.get('aicc'),
            'hqic': criteria.get('hqic')
        }
        
        # If test data is provided, calculate test error metrics
        if test_data is not None and test_target is not None:
            try:
                # Generate predictions
                predictions = model.predict(test_data)
                
                # Calculate error metrics
                result['rmse'] = np.sqrt(np.mean((test_target - predictions) ** 2))
                result['mae'] = np.mean(np.abs(test_target - predictions))
                
                # Calculate MAPE if no zeros in test_target
                if np.all(test_target != 0):
                    result['mape'] = np.mean(np.abs((test_target - predictions) / test_target)) * 100
                else:
                    # Alternative: sMAPE
                    denominator = np.abs(test_target) + np.abs(predictions)
                    result['smape'] = np.mean(2 * np.abs(test_target - predictions) / denominator) * 100
                
            except Exception as e:
                logger.error(f"Error calculating test metrics for model {model_name}: {e}")
        
        results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by AIC by default
    if 'aic' in results_df.columns:
        results_df = results_df.sort_values('aic')
    
    return results_df


def evaluate_residuals(model: BaseARIMAModel, 
                      plot: bool = True, 
                      figsize: Tuple[int, int] = (12, 10)) -> Dict[str, Any]:
    """
    Evaluate model residuals using statistical tests and plots.
    
    Args:
        model: Fitted ARIMA model
        plot: Whether to create diagnostic plots
        figsize: Figure size for plots
        
    Returns:
        Dictionary with test results and optionally a figure
    """
    from scipy import stats
    from statsmodels.stats.diagnostic import acorr_ljungbox
    
    if not model.is_fitted or model.residuals is None:
        raise ValueError("Model must be fitted with residuals before evaluation")
    
    # Get residuals
    residuals = model.residuals
    
    # Initialize results dictionary
    results = {}
    
    # Test for normality (Shapiro-Wilk)
    try:
        shapiro_test = stats.shapiro(residuals)
        results['shapiro_test'] = {
            'statistic': shapiro_test[0],
            'p_value': shapiro_test[1],
            'normal_residuals': shapiro_test[1] > 0.05
        }
    except Exception as e:
        logger.warning(f"Error in Shapiro-Wilk test: {e}")
    
    # Test for autocorrelation (Ljung-Box)
    try:
        ljung_box = acorr_ljungbox(residuals, lags=[10, 20, 30])
        results['ljung_box_test'] = {
            'lags': [10, 20, 30],
            'statistics': ljung_box.iloc[:, 0].tolist(),
            'p_values': ljung_box.iloc[:, 1].tolist(),
            'no_autocorrelation': all(p > 0.05 for p in ljung_box.iloc[:, 1])
        }
    except Exception as e:
        logger.warning(f"Error in Ljung-Box test: {e}")
    
    # Test for heteroscedasticity (Breusch-Pagan)
    try:
        from statsmodels.stats.diagnostic import het_breuschpagan
        # Create a series of integers as the independent variable
        x = np.arange(len(residuals)).reshape(-1, 1)
        bp_test = het_breuschpagan(residuals, x)
        results['breusch_pagan_test'] = {
            'statistic': bp_test[0],
            'p_value': bp_test[1],
            'homoscedastic': bp_test[1] > 0.05
        }
    except Exception as e:
        logger.warning(f"Error in Breusch-Pagan test: {e}")
    
    # Generate diagnostic plots if requested
    if plot:
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot residuals
        axes[0, 0].plot(residuals)
        axes[0, 0].set_title('Residuals')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Residual')
        
        # Plot histogram of residuals
        axes[0, 1].hist(residuals, bins=20, density=True)
        # Add a normal distribution curve
        x = np.linspace(min(residuals), max(residuals), 100)
        axes[0, 1].plot(x, stats.norm.pdf(x, np.mean(residuals), np.std(residuals)))
        axes[0, 1].set_title('Histogram of Residuals with Normal Curve')
        axes[0, 1].set_xlabel('Residual')
        axes[0, 1].set_ylabel('Density')
        
        # Plot ACF of residuals
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        plot_acf(residuals, lags=20, ax=axes[1, 0])
        axes[1, 0].set_title('ACF of Residuals')
        
        # Plot PACF of residuals
        plot_pacf(residuals, lags=20, ax=axes[1, 1])
        axes[1, 1].set_title('PACF of Residuals')
        
        plt.tight_layout()
        results['diagnostics_plot'] = fig
    
    # Return the results dictionary
    return results


def cross_validate_arima(y: pd.Series, X: Optional[pd.DataFrame] = None,
                        model_type: str = 'auto',
                        order: Optional[Tuple[int, int, int]] = None,
                        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
                        exog_columns: Optional[List[str]] = None,
                        n_splits: int = 5,
                        train_pct: float = 0.8,
                        rolling_window: bool = True,
                        date_col: str = 'date') -> Dict[str, Any]:
    """
    Perform time series cross-validation for ARIMA models.
    
    Args:
        y: Time series data
        X: Feature matrix (optional)
        model_type: Type of model ('auto', 'arima', 'sarima', 'arimax', 'sarimax')
        order: ARIMA order (p, d, q) - if None, will be selected automatically
        seasonal_order: Seasonal order (P, D, Q, m) - if None, will be selected automatically
        exog_columns: Exogenous variable columns
        n_splits: Number of cross-validation splits
        train_pct: Initial training percentage
        rolling_window: Whether to use rolling window (True) or expanding window (False)
        date_col: Date column name
        
    Returns:
        Dictionary with cross-validation results
    """
    if X is None and exog_columns is not None:
        raise ValueError("X must be provided when exog_columns is specified")
    
    # Ensure y is a pandas Series with an index
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    
    # Get total length
    n = len(y)
    
    # Calculate initial training size
    init_train_size = int(n * train_pct)
    
    # Create time series splits
    test_starts = np.linspace(init_train_size, n-1, n_splits+1).astype(int)[:-1]
    split_indices = []
    
    for test_start in test_starts:
        if rolling_window:
            # Rolling window approach
            train_indices = list(range(test_start - init_train_size, test_start))
        else:
            # Expanding window approach
            train_indices = list(range(test_start))
        
        test_indices = list(range(test_start, min(test_start + (n - init_train_size) // n_splits, n)))
        split_indices.append((train_indices, test_indices))
    
    # Initialize results
    cv_results = {
        'model_type': model_type,
        'order': order,
        'seasonal_order': seasonal_order,
        'splits': []
    }
    
    # Cross-validation
    for i, (train_idx, test_idx) in enumerate(split_indices):
        logger.info(f"Cross-validation split {i+1}/{len(split_indices)}")
        
        try:
            # Split data
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            
            X_train = X.iloc[train_idx] if X is not None else None
            X_test = X.iloc[test_idx] if X is not None else None
            
            # Create model
            if model_type == 'auto':
                # Determine if we need seasonal and exogenous
                seasonal = seasonal_order is not None
                with_exog = exog_columns is not None
                
                # Use auto_arima to select model
                model = auto_arima(
                    y=y_train, 
                    X=X_train, 
                    exog_columns=exog_columns,
                    seasonal=seasonal,
                    with_exogenous=with_exog
                )
            elif model_type == 'arima':
                if order is None:
                    raise ValueError("order must be provided for 'arima' model_type")
                
                model = ARIMAModel(date_col=date_col, order=order)
                model.fit(X_train if X_train is not None else pd.DataFrame({date_col: y_train.index}), y_train)
            elif model_type == 'sarima':
                if order is None or seasonal_order is None:
                    raise ValueError("order and seasonal_order must be provided for 'sarima' model_type")
                
                model = SARIMAModel(date_col=date_col, order=order, seasonal_order=seasonal_order)
                model.fit(X_train if X_train is not None else pd.DataFrame({date_col: y_train.index}), y_train)
            elif model_type == 'arimax':
                if order is None:
                    raise ValueError("order must be provided for 'arimax' model_type")
                if exog_columns is None:
                    raise ValueError("exog_columns must be provided for 'arimax' model_type")
                
                model = ARIMAXModel(date_col=date_col, order=order, exog_columns=exog_columns)
                model.fit(X_train, y_train)
            elif model_type == 'sarimax':
                if order is None or seasonal_order is None:
                    raise ValueError("order and seasonal_order must be provided for 'sarimax' model_type")
                if exog_columns is None:
                    raise ValueError("exog_columns must be provided for 'sarimax' model_type")
                
                model = SARIMAXModel(date_col=date_col, order=order, seasonal_order=seasonal_order, exog_columns=exog_columns)
                model.fit(X_train, y_train)
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
            
            # Make predictions
            predictions = model.predict(X_test if X_test is not None else pd.DataFrame({date_col: y_test.index}))
            
            # Calculate metrics
            rmse_val = np.sqrt(np.mean((y_test - predictions) ** 2))
            mae_val = np.mean(np.abs(y_test - predictions))
            
            # Store split results
            split_result = {
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'rmse': rmse_val,
                'mae': mae_val
            }
            
            # Calculate MAPE if no zeros in y_test
            if not np.any(y_test == 0):
                mape_val = np.mean(np.abs((y_test - predictions) / y_test)) * 100
                split_result['mape'] = mape_val
            
            cv_results['splits'].append(split_result)
            
        except Exception as e:
            logger.error(f"Error in cross-validation split {i+1}: {e}")
            cv_results['splits'].append({'error': str(e)})
    
    # Calculate aggregate metrics
    if cv_results['splits']:
        valid_splits = [s for s in cv_results['splits'] if 'error' not in s]
        if valid_splits:
            cv_results['mean_rmse'] = np.mean([s['rmse'] for s in valid_splits])
            cv_results['mean_mae'] = np.mean([s['mae'] for s in valid_splits])
            
            if all('mape' in s for s in valid_splits):
                cv_results['mean_mape'] = np.mean([s['mape'] for s in valid_splits])
    
    return cv_results


def plot_cross_validation_results(cv_results: Dict[str, Any], 
                                 metric: str = 'rmse', 
                                 figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot time series cross-validation results.
    
    Args:
        cv_results: Results from cross_validate_arima
        metric: Which metric to plot ('rmse', 'mae', 'mape')
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Extract valid splits
    valid_splits = [s for s in cv_results['splits'] if 'error' not in s and metric in s]
    
    if not valid_splits:
        raise ValueError(f"No valid splits with metric '{metric}' found in cv_results")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot metric for each split
    split_numbers = list(range(1, len(valid_splits) + 1))
    metric_values = [s[metric] for s in valid_splits]
    
    ax.plot(split_numbers, metric_values, 'o-', label=f'CV {metric.upper()}')
    
    # Add mean line
    mean_value = np.mean(metric_values)
    ax.axhline(mean_value, linestyle='--', color='r', label=f'Mean {metric.upper()}: {mean_value:.4f}')
    
    # Add labels and title
    ax.set_xlabel('Cross-Validation Split')
    ax.set_ylabel(metric.upper())
    
    model_str = cv_results['model_type']
    if cv_results['order']:
        model_str += f" {cv_results['order']}"
    if cv_results['seasonal_order']:
        model_str += f" x {cv_results['seasonal_order']}"
        
    ax.set_title(f'{metric.upper()} for {model_str} Time Series Cross-Validation')
    ax.legend()
    ax.grid(True)
    
    return fig


def grid_search_arima(y: pd.Series, X: Optional[pd.DataFrame] = None,
                     p_values: List[int] = [0, 1, 2],
                     d_values: List[int] = [0, 1],
                     q_values: List[int] = [0, 1, 2],
                     P_values: Optional[List[int]] = None, 
                     D_values: Optional[List[int]] = None,
                     Q_values: Optional[List[int]] = None,
                     m: Optional[int] = None,
                     exog_columns: Optional[List[str]] = None,
                     information_criterion: str = 'aic',
                     return_best: bool = True,
                     date_col: str = 'date') -> Union[pd.DataFrame, BaseARIMAModel]:
    """
    Perform grid search over ARIMA model parameters.
    
    Args:
        y: Time series data
        X: Feature matrix (optional)
        p_values, d_values, q_values: Lists of values for p, d, q parameters
        P_values, D_values, Q_values: Lists of values for seasonal P, D, Q parameters
        m: Seasonal period (optional)
        exog_columns: Exogenous variable columns (optional)
        information_criterion: Criterion for model selection ('aic', 'bic')
        return_best: Whether to return only the best model or all results
        date_col: Date column name
        
    Returns:
        DataFrame with all models or the best model
    """
    logger.info("Starting ARIMA grid search")
    
    # Determine if we need a seasonal model
    seasonal = all(x is not None for x in [P_values, D_values, Q_values, m])
    
    # Determine if we need a model with exogenous variables
    with_exogenous = exog_columns is not None and X is not None
    
    # Generate parameter combinations
    param_grid = list(product(p_values, d_values, q_values))
    logger.info(f"Generated {len(param_grid)} parameter combinations for non-seasonal component")
    
    # Add seasonal parameters if needed
    if seasonal:
        seasonal_grid = list(product(P_values, D_values, Q_values))
        logger.info(f"Generated {len(seasonal_grid)} parameter combinations for seasonal component")
        full_grid = list(product(param_grid, seasonal_grid))
        logger.info(f"Total grid size: {len(full_grid)} combinations")
    else:
        full_grid = [(params, None) for params in param_grid]
        logger.info(f"Total grid size: {len(full_grid)} combinations (non-seasonal)")
    
    # Results list
    results = []
    
    # Best model tracking
    best_model = None
    best_criterion_value = float('inf')
    
    # Iterate over grid
    for grid_idx, ((p, d, q), seasonal_params) in enumerate(full_grid):
        logger.info(f"Fitting model {grid_idx + 1}/{len(full_grid)}: "
                    f"ARIMA({p},{d},{q})"
                    f"{f'x{seasonal_params}' if seasonal_params else ''}"
                    f"{' with exogenous variables' if with_exogenous else ''}")
        
        try:
            # Create model
            if seasonal and with_exogenous:
                P, D, Q = seasonal_params
                model = SARIMAXModel(
                    date_col=date_col,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, m),
                    exog_columns=exog_columns
                )
                model.fit(X, y)
            elif seasonal:
                P, D, Q = seasonal_params
                model = SARIMAModel(
                    date_col=date_col,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, m)
                )
                model.fit(X if X is not None else pd.DataFrame({date_col: y.index}), y)
            elif with_exogenous:
                model = ARIMAXModel(
                    date_col=date_col,
                    order=(p, d, q),
                    exog_columns=exog_columns
                )
                model.fit(X, y)
            else:
                model = ARIMAModel(
                    date_col=date_col,
                    order=(p, d, q)
                )
                model.fit(X if X is not None else pd.DataFrame({date_col: y.index}), y)
            
            # Get information criterion value
            criterion_value = getattr(model.model, information_criterion)
            
            # Create result entry
            result = {
                'order': (p, d, q),
                'seasonal_order': seasonal_params,
                information_criterion: criterion_value,
                'model_type': model.__class__.__name__
            }
            
            # Update best model if this one is better
            if criterion_value < best_criterion_value:
                best_criterion_value = criterion_value
                best_model = model
                result['best'] = True
            else:
                result['best'] = False
                
            results.append(result)
            
        except Exception as e:
            logger.warning(f"Error fitting model with order=({p},{d},{q}), "
                          f"seasonal_order={seasonal_params}: {e}")
            # Add error entry
            results.append({
                'order': (p, d, q),
                'seasonal_order': seasonal_params,
                'error': str(e),
                'model_type': 'Failed',
                'best': False
            })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    if results_df.empty:
        logger.error("No successful models found in grid search")
        return results_df
    
    # Sort by information criterion
    if information_criterion in results_df.columns:
        results_df = results_df.sort_values(information_criterion)
    
    # Return best model or results DataFrame
    if return_best:
        if best_model is not None:
            return best_model
        else:
            logger.error("No successful models found in grid search")
            return None
    else:
        return results_df