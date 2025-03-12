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
                      plot_results: bool = True,
                      data_loader: Optional[Any] = None,
                      frequency: Optional[str] = None) -> Dict:
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
        data_loader: Optional DataLoader instance for missing value handling
        frequency: Optional frequency of data for group-based imputation
        
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
    
    # Handle missing values consistently using data_loader if provided
    if data_loader is not None and frequency is not None:
        # Create a temporary df with features and target
        temp_df = pd.concat([X, y], axis=1)
        
        # Use the data_loader's missing value handling
        if frequency in ['monthly', 'quarterly']:
            fill_groups = ['year', 'month' if frequency == 'monthly' else 'quarter']
            if all(group in df.columns for group in fill_groups):
                # Add the grouping columns if they exist in the original df
                for group in fill_groups:
                    if group not in temp_df.columns and group in df.columns:
                        temp_df[group] = df[group]
                temp_df = data_loader.handle_missing_values(temp_df, method='mean', fill_groups=fill_groups)
            else:
                temp_df = data_loader.handle_missing_values(temp_df, method='mean')
        else:
            temp_df = data_loader.handle_missing_values(temp_df, method='mean')
        
        # Extract back the features and target
        X = temp_df[valid_cols]
        y = temp_df[target_col]
    else:
        # Fallback to simple imputation if data_loader not provided
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


def train_prophet_model(df, target_col, date_col, params=None, test_size=0.2):
    """
    Train and evaluate a Prophet model for time series forecasting.
    
    Args:
        df: DataFrame with features and target
        target_col: Name of target column
        date_col: Name of date column
        params: Optional parameters for Prophet model
        test_size: Proportion of data to use for testing
        
    Returns:
        Dictionary with evaluation results and forecasts
    """
    from prophet import Prophet
    import pandas as pd
    import numpy as np
    import logging
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    logger = logging.getLogger(__name__)
    
    # Check if the required columns exist
    if date_col not in df.columns:
        logger.error(f"Date column '{date_col}' not found in dataframe")
        raise ValueError(f"Date column '{date_col}' not found in dataframe")
    
    if target_col not in df.columns:
        logger.error(f"Target column '{target_col}' not found in dataframe")
        raise ValueError(f"Target column '{target_col}' not found in dataframe")
    
    # Create a copy to avoid modifying the original
    data = df.copy()
    
    if not params:
        params = {}
    
    # Ensure date column is datetime
    data[date_col] = pd.to_datetime(data[date_col])
    
    # Check for missing values in target
    if data[target_col].isna().any():
        logger.warning(f"Target column '{target_col}' has {data[target_col].isna().sum()} missing values")
        # Fill missing values with forward fill, then backward fill
        data[target_col] = data[target_col].ffill().bfill()
    
    # Log all columns in the dataframe
    logger.info(f"Columns in input dataframe: {data.columns.tolist()}")
    
    # Prepare data for Prophet (requires 'ds' for dates and 'y' for target)
    prophet_data = pd.DataFrame()
    prophet_data['ds'] = data[date_col]
    prophet_data['y'] = data[target_col]
    
    # Initialize Prophet model
    model = Prophet(
        yearly_seasonality=params.get('yearly_seasonality', True),
        weekly_seasonality=params.get('weekly_seasonality', False),
        daily_seasonality=params.get('daily_seasonality', False),
        changepoint_prior_scale=params.get('changepoint_prior_scale', 0.05),
        seasonality_prior_scale=params.get('seasonality_prior_scale', 10.0)
    )
    
    # Add seasonal periods if specified
    seasonal_periods = params.get('seasonal_periods', None)
    if seasonal_periods:
        for period in seasonal_periods:
            if isinstance(period, dict):
                model.add_seasonality(
                    name=period.get('name', f"custom_season_{period.get('period')}"),
                    period=period.get('period'),
                    fourier_order=period.get('fourier_order', 5)
                )
    
    # Add country holidays if specified
    country_holidays = params.get('country_holidays', None)
    if country_holidays:
        model.add_country_holidays(country_name=country_holidays)
    
    # Add regressors if specified
    regressors = params.get('regressors', [])
    if regressors:
        logger.info(f"Adding regressors: {regressors}")
        for reg in regressors:
            if reg in data.columns:
                # Copy the regressor data
                reg_data = data[reg].copy()
                
                # Check for NaN values
                if reg_data.isna().any():
                    nan_count = reg_data.isna().sum()
                    logger.warning(f"Found {nan_count} NaN values in regressor '{reg}', filling with mean")
                    reg_mean = reg_data.mean()
                    reg_data = reg_data.fillna(reg_mean)
                
                # Add to Prophet dataframe
                prophet_data[reg] = reg_data
                model.add_regressor(reg)
                logger.info(f"Regressor '{reg}' added to Prophet model")
            else:
                logger.error(f"Requested regressor '{reg}' not found in dataframe")
                raise ValueError(f"Regressor '{reg}' not found in dataframe")
    
    # Verify prophet_data contains all expected columns
    logger.info(f"Prophet dataframe columns: {prophet_data.columns.tolist()}")
    logger.info(f"Prophet dataframe shape: {prophet_data.shape}")
    
    # Split into train and test sets
    train_size = int(len(prophet_data) * (1 - test_size))
    train_data = prophet_data.iloc[:train_size]
    test_data = prophet_data.iloc[train_size:]
    
    logger.info(f"Training on {len(train_data)} samples, testing on {len(test_data)} samples")
    
    # Fit the model
    logger.info("Fitting Prophet model...")
    model.fit(train_data)
    logger.info("Prophet model fitting complete")
    
    # Create future dataframe and make predictions
    logger.info("Making predictions...")
    future = model.make_future_dataframe(
        periods=len(test_data),
        freq=params.get('freq', 'MS')  # MS for month start
    )
    
    # Add regressor values to future dataframe
    if regressors:
        for reg in regressors:
            if reg in data.columns:
                # Calculate the mean of the regressor to use for future values
                reg_mean = data[reg].mean()
                
                # For the existing data points, use the actual values
                known_values = data[reg].values
                
                # For future points beyond our data, use the mean
                unknown_values = np.full(len(future) - len(data), reg_mean)
                
                # Combine and assign to the future dataframe
                future[reg] = pd.Series(
                    np.concatenate([known_values, unknown_values]),
                    index=future.index
                )
                
                # Ensure no NaN values in regressor
                if future[reg].isna().any():
                    logger.warning(f"Found {future[reg].isna().sum()} NaN values in regressor '{reg}', filling with mean")
                    future[reg] = future[reg].fillna(reg_mean)
    
    # Make predictions
    forecast = model.predict(future)
    logger.info("Predictions complete")
    
    # Evaluate on test set
    y_true = test_data['y'].values
    y_pred = forecast.iloc[train_size:]['yhat'].values
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Calculate MAPE, handling zero values
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if any(mask) else np.nan
    
    # Calculate R²
    r2 = r2_score(y_true, y_pred)
    
    logger.info(f"Evaluation metrics - RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}%, R²: {r2:.4f}")
    
    # Prepare results
    result = {
        'model': model,
        'forecast': forecast,
        'train_data': train_data,
        'test_data': test_data,
        'test_evaluation': {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'predictions': y_pred
        },
        'y_test': pd.Series(y_true, index=test_data.index),
        'y_test_pred': pd.Series(y_pred, index=test_data.index)
    }
    
    return result

def plot_prophet_forecast(prophet_result, title="Prophet Forecast"):
    """
    Plot the forecast from a Prophet model.
    
    Args:
        prophet_result: Result dictionary from train_prophet_model
        title: Plot title
    """
    from prophet.plot import plot_plotly, plot_components_plotly
    import matplotlib.pyplot as plt
    
    model = prophet_result['model']
    forecast = prophet_result['forecast']
    train_data = prophet_result['train_data']
    test_data = prophet_result['test_data']
    
    # Create the figure
    plt.figure(figsize=(12, 6))
    
    # Plot the forecast
    plt.plot(forecast['ds'], forecast['yhat'], 'b-', label='Forecast')
    plt.fill_between(forecast['ds'], 
                    forecast['yhat_lower'], 
                    forecast['yhat_upper'], 
                    color='blue', alpha=0.2, label='Uncertainty Interval')
    
    # Plot the actual values
    plt.plot(train_data['ds'], train_data['y'], 'k.', label='Training Data')
    plt.plot(test_data['ds'], test_data['y'], 'r.', label='Test Data')
    
    # Add labels and title
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return

def plot_prophet_components(prophet_result, title="Prophet Components"):
    """
    Plot the components from a Prophet model.
    
    Args:
        prophet_result: Result dictionary from train_prophet_model
        title: Plot title
    """
    from prophet.plot import plot_components
    import matplotlib.pyplot as plt
    
    model = prophet_result['model']
    forecast = prophet_result['forecast']
    
    plt.figure(figsize=(12, 10))
    plot_components(model, forecast)
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()
    
    return