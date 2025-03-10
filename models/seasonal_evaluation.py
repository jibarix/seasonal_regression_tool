"""
Evaluation module for seasonal time series models.
Provides metrics and comparison functions focused on seasonality.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Union, Optional, Tuple, Any
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf

# Setup logging
logger = logging.getLogger(__name__)


def calculate_seasonal_error_metrics(y_true: pd.Series, y_pred: pd.Series,
                                    date_series: pd.Series,
                                    seasonality_type: str = 'monthly') -> Dict[str, Dict[str, float]]:
    """
    Calculate error metrics broken down by seasonal periods.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        date_series: Dates corresponding to the values
        seasonality_type: Type of seasonality ('monthly', 'quarterly')
        
    Returns:
        Dictionary with error metrics by season
    """
    # Ensure Series have the same index
    if len(y_true) != len(y_pred) or len(y_true) != len(date_series):
        raise ValueError("All input series must have the same length")
    
    # Convert to pandas Series if needed
    y_true = pd.Series(y_true) if not isinstance(y_true, pd.Series) else y_true
    y_pred = pd.Series(y_pred) if not isinstance(y_pred, pd.Series) else y_pred
    date_series = pd.Series(date_series) if not isinstance(date_series, pd.Series) else date_series
    
    # Ensure date_series is datetime
    date_series = pd.to_datetime(date_series)
    
    # Create DataFrame with all data
    df = pd.DataFrame({
        'date': date_series,
        'true': y_true,
        'pred': y_pred,
        'error': y_true - y_pred,
        'abs_error': np.abs(y_true - y_pred),
        'squared_error': (y_true - y_pred) ** 2
    })
    
    # Add seasonal period
    if seasonality_type == 'monthly':
        df['period'] = df['date'].dt.month
        period_names = {i: pd.Timestamp(2000, i, 1).strftime('%b') for i in range(1, 13)}
    elif seasonality_type == 'quarterly':
        df['period'] = df['date'].dt.quarter
        period_names = {1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'}
    else:
        raise ValueError(f"Unsupported seasonality type: {seasonality_type}")
    
    # Calculate metrics by period
    seasonal_metrics = {}
    
    # Overall metrics
    overall_metrics = {
        'rmse': np.sqrt(np.mean(df['squared_error'])),
        'mae': np.mean(df['abs_error']),
        'median_ae': np.median(df['abs_error']),
        'mean_error': np.mean(df['error']),
        'count': len(df)
    }
    
    seasonal_metrics['overall'] = overall_metrics
    
    # Calculate metrics for each period
    for period in sorted(df['period'].unique()):
        period_df = df[df['period'] == period]
        
        if len(period_df) > 0:
            period_metrics = {
                'rmse': np.sqrt(np.mean(period_df['squared_error'])),
                'mae': np.mean(period_df['abs_error']),
                'median_ae': np.median(period_df['abs_error']),
                'mean_error': np.mean(period_df['error']),
                'count': len(period_df),
                'period_name': period_names.get(period, str(period))
            }
            
            seasonal_metrics[f'period_{period}'] = period_metrics
    
    # Find best and worst periods
    period_rmse = {k: v['rmse'] for k, v in seasonal_metrics.items() 
                  if k.startswith('period_')}
    
    if period_rmse:
        best_period = min(period_rmse.items(), key=lambda x: x[1])[0]
        worst_period = max(period_rmse.items(), key=lambda x: x[1])[0]
        
        seasonal_metrics['best_period'] = seasonal_metrics[best_period]
        seasonal_metrics['worst_period'] = seasonal_metrics[worst_period]
    
    logger.info(f"Calculated seasonal error metrics for {seasonality_type} data")
    return seasonal_metrics


def compare_seasonal_models(models: Dict[str, Any], 
                           X_test: pd.DataFrame, 
                           y_test: pd.Series,
                           date_col: str = 'date',
                           seasonality_type: str = 'monthly') -> pd.DataFrame:
    """
    Compare multiple seasonal models with different approaches.
    
    Args:
        models: Dictionary mapping model names to model objects
        X_test: Test feature matrix
        y_test: Test target variable
        date_col: Name of date column
        seasonality_type: Type of seasonality
        
    Returns:
        DataFrame with comparison metrics
    """
    comparison_results = []
    
    # Get dates from test data
    date_series = X_test[date_col] if date_col in X_test.columns else None
    
    for model_name, model in models.items():
        # Generate predictions
        try:
            y_pred = model.predict(X_test)
            
            # Calculate overall metrics
            rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
            mae = np.mean(np.abs(y_test - y_pred))
            
            # Calculate R²
            ss_total = np.sum((y_test - np.mean(y_test)) ** 2)
            ss_residual = np.sum((y_test - y_pred) ** 2)
            r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
            
            # Calculate AIC if the model has a 'model' attribute with a method 'aic'
            aic = getattr(getattr(model, 'model', None), 'aic', float('nan'))
            if callable(aic):
                aic = aic()
            
            # Get number of parameters if available
            n_params = len(getattr(getattr(model, 'model', None), 'params', [])) - 1  # Subtract 1 for intercept
            
            # Calculate seasonal metrics if date_series is available
            if date_series is not None:
                seasonal_metrics = calculate_seasonal_error_metrics(
                    y_test, y_pred, date_series, seasonality_type
                )
                
                # Get worst period metrics
                worst_period = seasonal_metrics.get('worst_period', {})
                worst_period_name = worst_period.get('period_name', 'N/A')
                worst_period_rmse = worst_period.get('rmse', float('nan'))
                
                # Get best period metrics
                best_period = seasonal_metrics.get('best_period', {})
                best_period_name = best_period.get('period_name', 'N/A')
                best_period_rmse = best_period.get('rmse', float('nan'))
                
                # Calculate seasonal variance ratio
                # (ratio of error variance across seasons to overall error variance)
                period_rmses = [v['rmse'] for k, v in seasonal_metrics.items() 
                              if k.startswith('period_')]
                
                if period_rmses:
                    seasonal_variance = np.var(period_rmses)
                    seasonal_variance_ratio = seasonal_variance / rmse**2
                else:
                    seasonal_variance_ratio = float('nan')
            else:
                worst_period_name = 'N/A'
                worst_period_rmse = float('nan')
                best_period_name = 'N/A'
                best_period_rmse = float('nan')
                seasonal_variance_ratio = float('nan')
            
            # Append results
            comparison_results.append({
                'model_name': model_name,
                'rmse': rmse,
                'mae': mae,
                'r_squared': r_squared,
                'aic': aic,
                'n_params': n_params,
                'worst_period': worst_period_name,
                'worst_period_rmse': worst_period_rmse,
                'best_period': best_period_name,
                'best_period_rmse': best_period_rmse,
                'seasonal_variance_ratio': seasonal_variance_ratio
            })
            
        except Exception as e:
            logger.error(f"Error evaluating model '{model_name}': {e}")
            comparison_results.append({
                'model_name': model_name,
                'rmse': float('nan'),
                'mae': float('nan'),
                'r_squared': float('nan'),
                'aic': float('nan'),
                'n_params': float('nan'),
                'error': str(e)
            })
    
    # Create DataFrame and sort by RMSE
    comparison_df = pd.DataFrame(comparison_results)
    if not comparison_df.empty and 'rmse' in comparison_df.columns:
        comparison_df = comparison_df.sort_values('rmse')
    
    logger.info(f"Compared {len(models)} seasonal models")
    return comparison_df


def calculate_directional_accuracy(y_true: pd.Series, y_pred: pd.Series,
                                  date_series: pd.Series,
                                  seasonality_type: str = 'monthly') -> Dict[str, float]:
    """
    Calculate directional accuracy metrics for seasonal forecasts.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        date_series: Dates corresponding to the values
        seasonality_type: Type of seasonality
        
    Returns:
        Dictionary with directional accuracy metrics
    """
    # Ensure Series have the same index
    if len(y_true) != len(y_pred) or len(y_true) != len(date_series):
        raise ValueError("All input series must have the same length")
    
    # Convert to pandas Series if needed
    y_true = pd.Series(y_true) if not isinstance(y_true, pd.Series) else y_true
    y_pred = pd.Series(y_pred) if not isinstance(y_pred, pd.Series) else y_pred
    date_series = pd.to_datetime(date_series) if not isinstance(date_series, pd.Series) else pd.to_datetime(date_series)
    
    # Calculate direction of changes
    true_diff = y_true.diff()
    pred_diff = y_pred.diff()
    
    # Create DataFrame with all data
    df = pd.DataFrame({
        'date': date_series,
        'true_diff': true_diff,
        'pred_diff': pred_diff,
    })
    
    # Remove first row (NaN due to diff)
    df = df.dropna()
    
    # Calculate overall directional accuracy
    correct_direction = (df['true_diff'] * df['pred_diff']) > 0
    overall_accuracy = correct_direction.mean()
    
    # Add seasonal period
    if seasonality_type == 'monthly':
        df['period'] = df['date'].dt.month
        period_names = {i: pd.Timestamp(2000, i, 1).strftime('%b') for i in range(1, 13)}
    elif seasonality_type == 'quarterly':
        df['period'] = df['date'].dt.quarter
        period_names = {1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'}
    else:
        raise ValueError(f"Unsupported seasonality type: {seasonality_type}")
    
    # Calculate directional accuracy by period
    seasonal_accuracy = {}
    seasonal_accuracy['overall'] = overall_accuracy
    
    for period in sorted(df['period'].unique()):
        period_df = df[df['period'] == period]
        
        if len(period_df) > 0:
            period_correct = (period_df['true_diff'] * period_df['pred_diff']) > 0
            period_accuracy = period_correct.mean()
            
            seasonal_accuracy[f'period_{period}'] = {
                'accuracy': period_accuracy,
                'count': len(period_df),
                'period_name': period_names.get(period, str(period))
            }
    
    # Find best and worst periods
    period_accuracy = {k: v['accuracy'] for k, v in seasonal_accuracy.items() 
                       if isinstance(v, dict)}
    
    if period_accuracy:
        best_period = max(period_accuracy.items(), key=lambda x: x[1])[0]
        worst_period = min(period_accuracy.items(), key=lambda x: x[1])[0]
        
        seasonal_accuracy['best_period'] = seasonal_accuracy[best_period]
        seasonal_accuracy['worst_period'] = seasonal_accuracy[worst_period]
    
    logger.info(f"Calculated directional accuracy metrics for {seasonality_type} data")
    return seasonal_accuracy


def seasonal_forecast_evaluation(model: Any, X_train: pd.DataFrame, y_train: pd.Series,
                                X_test: pd.DataFrame, y_test: pd.Series,
                                date_col: str = 'date',
                                seasonality_type: str = 'monthly') -> Dict[str, Any]:
    """
    Comprehensive evaluation of a seasonal forecast model.
    
    Args:
        model: Model object with predict method
        X_train: Training feature matrix
        y_train: Training target variable
        X_test: Test feature matrix
        y_test: Test target variable
        date_col: Name of date column
        seasonality_type: Type of seasonality
        
    Returns:
        Dictionary with evaluation metrics and results
    """
    # Create evaluation result dictionary
    evaluation = {
        'model_type': type(model).__name__,
        'train_size': len(X_train),
        'test_size': len(X_test)
    }
    
    # Get dates from test data
    test_dates = X_test[date_col] if date_col in X_test.columns else None
    
    # Generate predictions
    try:
        y_pred = model.predict(X_test)
        
        # Calculate overall error metrics
        errors = y_test - y_pred
        abs_errors = np.abs(errors)
        squared_errors = errors ** 2
        
        evaluation['metrics'] = {
            'rmse': np.sqrt(np.mean(squared_errors)),
            'mae': np.mean(abs_errors),
            'median_ae': np.median(abs_errors),
            'mean_error': np.mean(errors),
            'max_error': np.max(abs_errors),
            'min_error': np.min(abs_errors)
        }
        
        # Calculate R²
        ss_total = np.sum((y_test - np.mean(y_test)) ** 2)
        ss_residual = np.sum(squared_errors)
        evaluation['metrics']['r_squared'] = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
        
        # Calculate AIC if the model has a 'model' attribute with a method 'aic'
        aic = getattr(getattr(model, 'model', None), 'aic', None)
        if callable(aic):
            evaluation['metrics']['aic'] = aic()
        
        # Calculate seasonal metrics if date_series is available
        if test_dates is not None:
            # Get seasonal error metrics
            seasonal_metrics = calculate_seasonal_error_metrics(
                y_test, y_pred, test_dates, seasonality_type
            )
            evaluation['seasonal_metrics'] = seasonal_metrics
            
            # Get directional accuracy
            directional_accuracy = calculate_directional_accuracy(
                y_test, y_pred, test_dates, seasonality_type
            )
            evaluation['directional_accuracy'] = directional_accuracy
            
            # Check for remaining seasonality in residuals
            try:
                # Create Series with datetime index
                residuals = pd.Series(errors, index=pd.to_datetime(test_dates))
                
                # Define seasonal period in observations
                if seasonality_type == 'monthly':
                    period = 12
                elif seasonality_type == 'quarterly':
                    period = 4
                else:
                    period = 12
                
                # Only run decomposition if enough data
                if len(residuals) >= period * 2:
                    # Perform seasonal decomposition
                    if np.isnan(residuals).any():
                        residuals = residuals.interpolate()
                    
                    decomposition = seasonal_decompose(
                        residuals, model='additive', period=period, extrapolate_trend='freq'
                    )
                    
                    seasonal_component = decomposition.seasonal
                    
                    # Calculate variance of seasonal component
                    var_seasonal = np.var(seasonal_component)
                    var_residual = np.var(residuals)
                    
                    # Calculate strength of remaining seasonality
                    seasonality_strength = var_seasonal / var_residual if var_residual > 0 else 0
                    
                    # Store results
                    evaluation['residual_seasonality'] = {
                        'seasonality_strength': seasonality_strength,
                        'seasonal_variance': var_seasonal,
                        'residual_variance': var_residual
                    }
                else:
                    evaluation['residual_seasonality'] = {
                        'note': f"Not enough data for decomposition (need at least {period * 2} points)"
                    }
            except Exception as e:
                logger.warning(f"Error in residual seasonality analysis: {e}")
                evaluation['residual_seasonality'] = {
                    'error': str(e)
                }
        
        # Add forecast quality assessment
        evaluation['quality_assessment'] = assess_forecast_quality(evaluation)
        
    except Exception as e:
        logger.error(f"Error in seasonal forecast evaluation: {e}")
        evaluation['error'] = str(e)
    
    logger.info(f"Completed seasonal forecast evaluation for {type(model).__name__}")
    return evaluation


def assess_forecast_quality(evaluation: Dict[str, Any]) -> Dict[str, str]:
    """
    Assess the quality of a seasonal forecast based on evaluation metrics.
    
    Args:
        evaluation: Dictionary with evaluation metrics
        
    Returns:
        Dictionary with quality assessments
    """
    assessment = {}
    
    # Get metrics
    metrics = evaluation.get('metrics', {})
    r_squared = metrics.get('r_squared', 0)
    
    # Assess overall fit
    if r_squared >= 0.9:
        assessment['overall_fit'] = 'Excellent'
    elif r_squared >= 0.8:
        assessment['overall_fit'] = 'Good'
    elif r_squared >= 0.6:
        assessment['overall_fit'] = 'Acceptable'
    elif r_squared >= 0.4:
        assessment['overall_fit'] = 'Poor'
    else:
        assessment['overall_fit'] = 'Very Poor'
    
    # Check if seasonal metrics are available
    seasonal_metrics = evaluation.get('seasonal_metrics', {})
    if seasonal_metrics:
        # Get worst period
        worst_period = seasonal_metrics.get('worst_period', {})
        worst_period_name = worst_period.get('period_name', 'Unknown')
        
        # Get best period
        best_period = seasonal_metrics.get('best_period', {})
        best_period_name = best_period.get('period_name', 'Unknown')
        
        # Assess seasonal performance
        assessment['seasonal_balance'] = 'Well-balanced' if worst_period.get('rmse', 0) / metrics.get('rmse', 1) < 1.5 else 'Unbalanced'
        assessment['best_period'] = best_period_name
        assessment['worst_period'] = worst_period_name
    
    # Check for residual seasonality
    residual_seasonality = evaluation.get('residual_seasonality', {})
    if 'seasonality_strength' in residual_seasonality:
        seasonality_strength = residual_seasonality['seasonality_strength']
        
        if seasonality_strength < 0.1:
            assessment['residual_seasonality'] = 'Minimal'
        elif seasonality_strength < 0.3:
            assessment['residual_seasonality'] = 'Moderate'
        else:
            assessment['residual_seasonality'] = 'Strong'
    
    # Check directional accuracy
    directional_accuracy = evaluation.get('directional_accuracy', {})
    overall_accuracy = directional_accuracy.get('overall', 0)
    
    if overall_accuracy >= 0.8:
        assessment['directional_accuracy'] = 'Excellent'
    elif overall_accuracy >= 0.7:
        assessment['directional_accuracy'] = 'Good'
    elif overall_accuracy >= 0.6:
        assessment['directional_accuracy'] = 'Acceptable'
    else:
        assessment['directional_accuracy'] = 'Poor'
    
    # Overall assessment
    if assessment.get('overall_fit') in ['Excellent', 'Good'] and assessment.get('directional_accuracy') in ['Excellent', 'Good'] and assessment.get('residual_seasonality') in ['Minimal', 'Moderate']:
        assessment['overall'] = 'High quality forecast'
    elif assessment.get('overall_fit') in ['Very Poor'] or assessment.get('directional_accuracy') in ['Poor'] or assessment.get('residual_seasonality') in ['Strong']:
        assessment['overall'] = 'Low quality forecast'
    else:
        assessment['overall'] = 'Medium quality forecast'
    
    return assessment