"""
Modeling module for time series forecasting.
Provides functionality for regression model training, evaluation,
time series cross-validation, and residual analysis.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional, Tuple, Any
import logging
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
import warnings

# Setup logging
logger = logging.getLogger(__name__)

def train_model(X_train: pd.DataFrame, 
               y_train: pd.Series, 
               model_type: str = 'linear',
               params: Optional[Dict] = None) -> Any:
    """
    Train a regression model for time series forecasting.
    
    Args:
        X_train: Training features
        y_train: Training target
        model_type: Type of model to train ('linear', 'ridge', 'lasso', 'elasticnet',
                  'randomforest', 'gbm')
        params: Optional parameters for the model
        
    Returns:
        Trained model object
    """
    if params is None:
        params = {}
    
    # Initialize model based on type
    if model_type == 'linear':
        model = LinearRegression(**params)
    elif model_type == 'ridge':
        alpha = params.get('alpha', 1.0)
        model = Ridge(alpha=alpha, random_state=42, **{k: v for k, v in params.items() if k != 'alpha'})
    elif model_type == 'lasso':
        alpha = params.get('alpha', 1.0)
        model = Lasso(alpha=alpha, random_state=42, **{k: v for k, v in params.items() if k != 'alpha'})
    elif model_type == 'elasticnet':
        alpha = params.get('alpha', 1.0)
        l1_ratio = params.get('l1_ratio', 0.5)
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42, 
                         **{k: v for k, v in params.items() if k not in ['alpha', 'l1_ratio']})
    elif model_type == 'randomforest':
        n_estimators = params.get('n_estimators', 100)
        max_depth = params.get('max_depth', None)
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, 
                                     random_state=42, 
                                     **{k: v for k, v in params.items() if k not in ['n_estimators', 'max_depth']})
    elif model_type == 'gbm':
        n_estimators = params.get('n_estimators', 100)
        learning_rate = params.get('learning_rate', 0.1)
        max_depth = params.get('max_depth', 3)
        model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, 
                                        max_depth=max_depth, random_state=42,
                                        **{k: v for k, v in params.items() 
                                           if k not in ['n_estimators', 'learning_rate', 'max_depth']})
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Fit the model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)
    
    return model

def evaluate_model(model: Any, 
                  X: pd.DataFrame, 
                  y: pd.Series,
                  return_predictions: bool = False) -> Dict:
    """
    Evaluate a trained model on given data.
    
    Args:
        model: Trained model object
        X: Feature data
        y: Target data
        return_predictions: Whether to include predictions in the result
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    
    # Calculate MAPE, handling zero values
    mask = y != 0
    mape = np.mean(np.abs((y[mask] - y_pred[mask]) / y[mask])) * 100 if any(mask) else np.nan
    
    # Calculate R²
    r2 = r2_score(y, y_pred)
    
    # Create result dictionary
    result = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2
    }
    
    # Add predictions if requested
    if return_predictions:
        result['predictions'] = y_pred
    
    return result

def time_series_cv(X: pd.DataFrame, 
                  y: pd.Series, 
                  model_type: str = 'linear',
                  params: Optional[Dict] = None,
                  n_splits: int = 5,
                  initial_train_size: Optional[int] = None) -> Dict:
    """
    Perform time series cross-validation.
    
    Args:
        X: Feature data
        y: Target data
        model_type: Type of model to train
        params: Optional parameters for the model
        n_splits: Number of splits for cross-validation
        initial_train_size: Size of initial training set (None for automatic)
        
    Returns:
        Dictionary with evaluation results
    """
    if initial_train_size is None:
        # Default to training on at least 70% of the data
        initial_train_size = int(len(X) * 0.7)
    
    # Initialize time series split
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=None, gap=0)
    
    # Initialize results
    cv_results = {
        'rmse': [],
        'mae': [],
        'mape': [],
        'r2': [],
        'train_indices': [],
        'test_indices': []
    }
    
    # Perform cross-validation
    for train_index, test_index in tscv.split(X):
        # Ensure minimum training size
        if len(train_index) < initial_train_size:
            continue
        
        # Split data
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Train model
        model = train_model(X_train, y_train, model_type, params)
        
        # Evaluate model
        eval_results = evaluate_model(model, X_test, y_test)
        
        # Store results
        cv_results['rmse'].append(eval_results['rmse'])
        cv_results['mae'].append(eval_results['mae'])
        cv_results['mape'].append(eval_results['mape'])
        cv_results['r2'].append(eval_results['r2'])
        cv_results['train_indices'].append(train_index)
        cv_results['test_indices'].append(test_index)
    
    # Calculate mean and std of metrics
    for metric in ['rmse', 'mae', 'mape', 'r2']:
        cv_results[f'mean_{metric}'] = np.mean(cv_results[metric])
        cv_results[f'std_{metric}'] = np.std(cv_results[metric])
    
    return cv_results

def analyze_residuals(y_true: pd.Series, 
                     y_pred: np.ndarray, 
                     date_index: Optional[pd.DatetimeIndex] = None) -> Dict:
    """
    Analyze residuals from a model.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        date_index: Optional DatetimeIndex for time-based analysis
        
    Returns:
        Dictionary with residual analysis results
    """
    # Calculate residuals
    residuals = y_true - y_pred
    
    # Basic statistics
    stats = {
        'mean': np.mean(residuals),
        'std': np.std(residuals),
        'min': np.min(residuals),
        'max': np.max(residuals)
    }
    
    # Create result with residuals and stats
    result = {
        'residuals': residuals,
        'stats': stats
    }
    
    # Test for autocorrelation
    try:
        # Ljung-Box test for autocorrelation
        lb_test = acorr_ljungbox(residuals, lags=[1], return_df=True)
        result['ljung_box'] = {
            'statistic': lb_test['lb_stat'].values[0],
            'p_value': lb_test['lb_pvalue'].values[0]
        }
        
        # Durbin-Watson test for autocorrelation
        dw_stat = durbin_watson(residuals)
        result['durbin_watson'] = dw_stat
        
        # Interpret Durbin-Watson statistic
        if dw_stat < 1.5:
            dw_interpretation = "Positive autocorrelation"
        elif dw_stat > 2.5:
            dw_interpretation = "Negative autocorrelation"
        else:
            dw_interpretation = "No significant autocorrelation"
            
        result['durbin_watson_interpretation'] = dw_interpretation
        
    except Exception as e:
        logger.warning(f"Error in autocorrelation tests: {e}")
    
    return result

def plot_residuals(residuals: pd.Series, 
                  date_index: Optional[pd.DatetimeIndex] = None,
                  title: str = "Residual Analysis") -> None:
    """
    Plot residuals for analysis.
    
    Args:
        residuals: Series of residuals
        date_index: Optional DatetimeIndex for time-based plotting
        title: Plot title
    """
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot residuals over time/index
    if date_index is not None:
        axes[0, 0].plot(date_index, residuals, 'o-', markersize=4)
        axes[0, 0].set_xlabel('Date')
    else:
        axes[0, 0].plot(residuals.index, residuals, 'o-', markersize=4)
        axes[0, 0].set_xlabel('Index')
    
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals Over Time')
    axes[0, 0].axhline(y=0, color='r', linestyle='-')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Histogram of residuals
    axes[0, 1].hist(residuals, bins=20, alpha=0.7)
    axes[0, 1].set_xlabel('Residual Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Histogram of Residuals')
    axes[0, 1].grid(True, alpha=0.3)
    
    # QQ plot of residuals
    from scipy import stats
    stats.probplot(residuals, plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Autocorrelation plot
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(residuals, lags=20, ax=axes[1, 1])
    axes[1, 1].set_title('Autocorrelation Plot')
    
    # Adjust layout and show plot
    plt.tight_layout()
    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.92)
    plt.show()

def plot_actual_vs_predicted(y_true: pd.Series, 
                            y_pred: np.ndarray,
                            date_index: Optional[pd.DatetimeIndex] = None,
                            title: str = "Actual vs Predicted") -> None:
    """
    Plot actual vs predicted values.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        date_index: Optional DatetimeIndex for time-based plotting
        title: Plot title
    """
    plt.figure(figsize=(12, 6))
    
    if date_index is not None:
        plt.plot(date_index, y_true, 'o-', label='Actual', markersize=4)
        plt.plot(date_index, y_pred, 's-', label='Predicted', markersize=4)
        plt.xlabel('Date')
    else:
        plt.plot(y_true.index, y_true, 'o-', label='Actual', markersize=4)
        plt.plot(y_pred.index, y_pred, 's-', label='Predicted', markersize=4)
        plt.xlabel('Index')
    
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def train_and_evaluate(df: pd.DataFrame,
                      target_col: str,
                      feature_cols: List[str],
                      date_col: Optional[str] = None,
                      model_type: str = 'linear',
                      params: Optional[Dict] = None,
                      test_size: float = 0.2,
                      use_cv: bool = True,
                      n_cv_splits: int = 5,
                      plot_results: bool = True) -> Dict:
    """
    Train and evaluate a model on the given data.
    
    Args:
        df: DataFrame with features and target
        target_col: Name of target column
        feature_cols: List of feature columns
        date_col: Name of date column (optional)
        model_type: Type of model to train
        params: Optional parameters for the model
        test_size: Proportion of data to use for testing
        use_cv: Whether to use time series cross-validation
        n_cv_splits: Number of splits for cross-validation
        plot_results: Whether to plot results
        
    Returns:
        Dictionary with trained model and evaluation results
    """
    if df.empty or target_col not in df.columns:
        logger.warning("Empty DataFrame or target column not found")
        return {}
    
    # Filter to only include columns that exist in the DataFrame
    valid_cols = [col for col in feature_cols if col in df.columns]
    if not valid_cols:
        logger.warning("None of the provided feature columns exist in the DataFrame")
        return {}
    
    # Create result dictionary
    result = {
        'model_type': model_type,
        'target_col': target_col,
        'feature_cols': valid_cols,
        'params': params or {}
    }
    
    # Extract features and target
    X = df[valid_cols].copy()
    y = df[target_col].copy()
    
    # Get date index if available
    date_index = None
    if date_col is not None and date_col in df.columns:
        date_index = pd.to_datetime(df[date_col])
        result['date_col'] = date_col
    
    # Handle missing values
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())
    
    # Split data into train and test sets
    test_size_records = int(len(df) * test_size)
    train_size_records = len(df) - test_size_records
    
    X_train, X_test = X.iloc[:train_size_records], X.iloc[train_size_records:]
    y_train, y_test = y.iloc[:train_size_records], y.iloc[train_size_records:]
    
    # Store train/test split info
    result['train_size'] = len(X_train)
    result['test_size'] = len(X_test)
    
    # Train model on training data
    model = train_model(X_train, y_train, model_type, params)
    result['model'] = model
    
    # Evaluate model on test data
    test_eval = evaluate_model(model, X_test, y_test, return_predictions=True)
    result['test_evaluation'] = test_eval
    
    # Store test set predictions
    y_test_pred = test_eval['predictions']
    result['y_test'] = y_test
    result['y_test_pred'] = y_test_pred
    
    # Perform time series cross-validation if requested
    if use_cv:
        cv_results = time_series_cv(X, y, model_type, params, n_splits=n_cv_splits)
        result['cv_results'] = cv_results
    
    # Analyze residuals
    residual_analysis = analyze_residuals(
        y_test, y_test_pred, date_index[train_size_records:] if date_index is not None else None
    )
    result['residual_analysis'] = residual_analysis
    
    # Plot results if requested
    if plot_results:
        # Plot actual vs predicted
        plot_actual_vs_predicted(
            y_test, y_test_pred, 
            date_index[train_size_records:] if date_index is not None else None,
            title=f"{model_type.capitalize()} Model: Actual vs Predicted"
        )
        
        # Plot residuals
        plot_residuals(
            pd.Series(residual_analysis['residuals'], index=y_test.index),
            date_index[train_size_records:] if date_index is not None else None,
            title=f"{model_type.capitalize()} Model: Residual Analysis"
        )
    
    # Extract feature importances if available
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.Series(
            model.feature_importances_, index=valid_cols
        ).sort_values(ascending=False)
        result['feature_importance'] = feature_importance
        
        if plot_results:
            # Plot feature importances
            plt.figure(figsize=(10, 6))
            feature_importance.head(20).plot(kind='bar')
            plt.title('Feature Importances')
            plt.ylabel('Importance')
            plt.tight_layout()
            plt.show()
    elif hasattr(model, 'coef_'):
        # For linear models
        coefficients = pd.Series(
            model.coef_, index=valid_cols
        ).sort_values(key=abs, ascending=False)
        result['coefficients'] = coefficients
        
        if plot_results:
            # Plot coefficients
            plt.figure(figsize=(10, 6))
            coefficients.head(20).plot(kind='bar')
            plt.title('Model Coefficients')
            plt.ylabel('Coefficient Value')
            plt.tight_layout()
            plt.show()
    
    return result

def setup_logging(log_file):
    """Setup logging to file and console."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger