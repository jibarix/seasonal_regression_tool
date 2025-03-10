"""
Advanced forecast analysis module for seasonal time series.
Provides model comparison, horizon analysis, and forecast quality assessment.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Union, Optional, Tuple, Any

# Local imports
from seasonal_error_metrics import calculate_seasonal_error_metrics, calculate_directional_accuracy

# Setup logging
logger = logging.getLogger(__name__)


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


def seasonal_forecast_errors_by_horizon(
    y_true: pd.Series,
    forecasts: Dict[str, pd.DataFrame],
    date_col: str = 'date',
    target_col: str = 'forecast',
    max_horizon: int = 12
) -> Dict[str, Any]:
    """
    Analyze how forecast errors change with forecast horizon, with focus on seasonal patterns.
    
    Args:
        y_true: Series of actual values with datetime index
        forecasts: Dictionary mapping model names to forecast DataFrames
        date_col: Name of date column in forecast DataFrames
        target_col: Name of forecast column in forecast DataFrames
        max_horizon: Maximum forecast horizon to analyze
        
    Returns:
        Dictionary with error metrics by horizon and model
    """
    # Ensure y_true has datetime index
    if not isinstance(y_true.index, pd.DatetimeIndex):
        raise ValueError("y_true must have a datetime index")
    
    # Initialize results
    results = {
        'by_horizon': {},
        'by_model': {},
        'by_season': {}
    }
    
    # Create dataframe for true values
    true_df = pd.DataFrame({'actual': y_true})
    true_df.index.name = date_col
    true_df = true_df.reset_index()
    
    # Extract month for seasonal analysis
    true_df['month'] = true_df[date_col].dt.month
    true_df['quarter'] = true_df[date_col].dt.quarter
    
    # Analyze each model's forecasts
    for model_name, forecast_df in forecasts.items():
        # Ensure date column is datetime
        forecast_df[date_col] = pd.to_datetime(forecast_df[date_col])
        
        # Get unique forecast start dates
        start_dates = forecast_df['forecast_start'].unique() if 'forecast_start' in forecast_df.columns else [None]
        
        # Initialize model metrics
        model_metrics = {
            'horizon_rmse': {},
            'horizon_mae': {},
            'season_rmse': {},
            'season_mae': {}
        }
        
        for start_date in start_dates:
            # Filter forecasts for this start date if needed
            if start_date is not None:
                model_forecasts = forecast_df[forecast_df['forecast_start'] == start_date].copy()
            else:
                model_forecasts = forecast_df.copy()
            
            # Skip if no forecasts with this start date
            if len(model_forecasts) == 0:
                continue
            
            # Calculate horizon for each forecast point
            if 'horizon' not in model_forecasts.columns:
                if 'forecast_start' in model_forecasts.columns:
                    # Calculate horizon based on difference from forecast start
                    for i, row in model_forecasts.iterrows():
                        forecast_date = row[date_col]
                        model_forecasts.loc[i, 'horizon'] = (forecast_date - start_date).days // 30 + 1
                else:
                    # Assume forecasts are in order from closest to furthest horizon
                    model_forecasts['horizon'] = range(1, len(model_forecasts) + 1)
            
            # Merge with true values on date
            merged = pd.merge(model_forecasts, true_df, on=date_col, how='inner')
            
            # Skip if no matching dates
            if len(merged) == 0:
                continue
            
            # Calculate errors
            merged['error'] = merged['actual'] - merged[target_col]
            merged['abs_error'] = np.abs(merged['error'])
            merged['squared_error'] = merged['error'] ** 2
            
            # Calculate errors by horizon
            for h in range(1, min(max_horizon + 1, merged['horizon'].max() + 1)):
                horizon_data = merged[merged['horizon'] == h]
                
                if len(horizon_data) > 0:
                    # Calculate RMSE and MAE for this horizon
                    horizon_rmse = np.sqrt(np.mean(horizon_data['squared_error']))
                    horizon_mae = np.mean(horizon_data['abs_error'])
                    
                    # Store in model metrics
                    if h not in model_metrics['horizon_rmse']:
                        model_metrics['horizon_rmse'][h] = []
                        model_metrics['horizon_mae'][h] = []
                    
                    model_metrics['horizon_rmse'][h].append(horizon_rmse)
                    model_metrics['horizon_mae'][h].append(horizon_mae)
            
            # Calculate errors by season (month or quarter)
            # For each month
            for month in range(1, 13):
                month_data = merged[merged['month'] == month]
                
                if len(month_data) > 0:
                    # Calculate RMSE and MAE for this month
                    month_rmse = np.sqrt(np.mean(month_data['squared_error']))
                    month_mae = np.mean(month_data['abs_error'])
                    
                    month_name = pd.Timestamp(2000, month, 1).strftime('%b')
                    
                    # Store in model metrics
                    if month_name not in model_metrics['season_rmse']:
                        model_metrics['season_rmse'][month_name] = []
                        model_metrics['season_mae'][month_name] = []
                    
                    model_metrics['season_rmse'][month_name].append(month_rmse)
                    model_metrics['season_mae'][month_name].append(month_mae)
            
            # For each quarter
            for quarter in range(1, 5):
                quarter_data = merged[merged['quarter'] == quarter]
                
                if len(quarter_data) > 0:
                    # Calculate RMSE and MAE for this quarter
                    quarter_rmse = np.sqrt(np.mean(quarter_data['squared_error']))
                    quarter_mae = np.mean(quarter_data['abs_error'])
                    
                    quarter_name = f"Q{quarter}"
                    
                    # Store in model metrics
                    if quarter_name not in model_metrics['season_rmse']:
                        model_metrics['season_rmse'][quarter_name] = []
                        model_metrics['season_mae'][quarter_name] = []
                    
                    model_metrics['season_rmse'][quarter_name].append(quarter_rmse)
                    model_metrics['season_mae'][quarter_name].append(quarter_mae)
        
        # Average metrics across all forecast start dates
        avg_horizon_rmse = {h: np.mean(values) for h, values in model_metrics['horizon_rmse'].items()}
        avg_horizon_mae = {h: np.mean(values) for h, values in model_metrics['horizon_mae'].items()}
        
        avg_season_rmse = {s: np.mean(values) for s, values in model_metrics['season_rmse'].items()}
        avg_season_mae = {s: np.mean(values) for s, values in model_metrics['season_mae'].items()}
        
        # Store in results
        results['by_model'][model_name] = {
            'horizon_rmse': avg_horizon_rmse,
            'horizon_mae': avg_horizon_mae,
            'season_rmse': avg_season_rmse,
            'season_mae': avg_season_mae
        }
        
        # Aggregate across horizons for overall metrics
        for h in avg_horizon_rmse:
            if h not in results['by_horizon']:
                results['by_horizon'][h] = {
                    'models': {},
                    'avg_rmse': 0,
                    'avg_mae': 0,
                    'count': 0
                }
            
            results['by_horizon'][h]['models'][model_name] = {
                'rmse': avg_horizon_rmse[h],
                'mae': avg_horizon_mae[h]
            }
            
            # Update averages
            results['by_horizon'][h]['avg_rmse'] += avg_horizon_rmse[h]
            results['by_horizon'][h]['avg_mae'] += avg_horizon_mae[h]
            results['by_horizon'][h]['count'] += 1
        
        # Aggregate across seasons
        for season in avg_season_rmse:
            if season not in results['by_season']:
                results['by_season'][season] = {
                    'models': {},
                    'avg_rmse': 0,
                    'avg_mae': 0,
                    'count': 0
                }
            
            results['by_season'][season]['models'][model_name] = {
                'rmse': avg_season_rmse[season],
                'mae': avg_season_mae[season]
            }
            
            # Update averages
            results['by_season'][season]['avg_rmse'] += avg_season_rmse[season]
            results['by_season'][season]['avg_mae'] += avg_season_mae[season]
            results['by_season'][season]['count'] += 1
    
    # Calculate final averages
    for h in results['by_horizon']:
        if results['by_horizon'][h]['count'] > 0:
            results['by_horizon'][h]['avg_rmse'] /= results['by_horizon'][h]['count']
            results['by_horizon'][h]['avg_mae'] /= results['by_horizon'][h]['count']
    
    for season in results['by_season']:
        if results['by_season'][season]['count'] > 0:
            results['by_season'][season]['avg_rmse'] /= results['by_season'][season]['count']
            results['by_season'][season]['avg_mae'] /= results['by_season'][season]['count']
    
    # Find best/worst horizons and seasons
    if results['by_horizon']:
        best_horizon = min(results['by_horizon'].items(), key=lambda x: x[1]['avg_rmse'])
        worst_horizon = max(results['by_horizon'].items(), key=lambda x: x[1]['avg_rmse'])
        
        results['summary'] = {
            'best_horizon': {
                'horizon': best_horizon[0],
                'avg_rmse': best_horizon[1]['avg_rmse'],
                'avg_mae': best_horizon[1]['avg_mae']
            },
            'worst_horizon': {
                'horizon': worst_horizon[0],
                'avg_rmse': worst_horizon[1]['avg_rmse'],
                'avg_mae': worst_horizon[1]['avg_mae']
            }
        }
    
    if results['by_season']:
        best_season = min(results['by_season'].items(), key=lambda x: x[1]['avg_rmse'])
        worst_season = max(results['by_season'].items(), key=lambda x: x[1]['avg_rmse'])
        
        if 'summary' not in results:
            results['summary'] = {}
        
        results['summary']['best_season'] = {
            'season': best_season[0],
            'avg_rmse': best_season[1]['avg_rmse'],
            'avg_mae': best_season[1]['avg_mae']
        }
        
        results['summary']['worst_season'] = {
            'season': worst_season[0],
            'avg_rmse': worst_season[1]['avg_rmse'],
            'avg_mae': worst_season[1]['avg_mae']
        }
    
    return results


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
    if assessment.get('overall_fit') in ['Excellent', 'Good'] and assessment.get('directional_accuracy') in ['Excellent', 'Good']:
        assessment['overall'] = 'High quality forecast'
    elif assessment.get('overall_fit') in ['Very Poor'] or assessment.get('directional_accuracy') in ['Poor']:
        assessment['overall'] = 'Low quality forecast'
    else:
        assessment['overall'] = 'Medium quality forecast'
    
    return assessment


def evaluate_forecast_horizons(model: Any, 
                              X: pd.DataFrame, 
                              y: pd.Series,
                              date_col: str = 'date',
                              horizons: List[int] = [1, 3, 6, 12],
                              seasonality_type: str = 'monthly') -> Dict[str, Any]:
    """
    Evaluate model performance across different forecast horizons.
    
    Args:
        model: Fitted model
        X: Feature matrix
        y: Target variable
        date_col: Name of date column
        horizons: List of horizons to evaluate (in periods)
        seasonality_type: Type of seasonality
        
    Returns:
        Dictionary with metrics by horizon
    """
    results = {
        'model_type': type(model).__name__,
        'by_horizon': {}
    }
    
    # Get date series
    if date_col in X.columns:
        dates = pd.to_datetime(X[date_col])
    else:
        logger.warning(f"Date column '{date_col}' not found, using index as dates")
        dates = pd.Series(X.index)
    
    # Sort everything by date
    sorted_indices = dates.argsort()
    X_sorted = X.iloc[sorted_indices].reset_index(drop=True)
    y_sorted = y.iloc[sorted_indices].reset_index(drop=True)
    dates_sorted = dates.iloc[sorted_indices].reset_index(drop=True)
    
    # For each horizon
    for horizon in horizons:
        # Split data to create a rolling forecast
        horizon_metrics = []
        
        # Create rolling windows
        n = len(X_sorted)
        for i in range(n - horizon):
            # Training data: up to current point
            X_train = X_sorted.iloc[:i+1]
            y_train = y_sorted.iloc[:i+1]
            
            # Test data: horizon steps ahead
            X_test = X_sorted.iloc[i+1:i+1+horizon]
            y_test = y_sorted.iloc[i+1:i+1+horizon]
            dates_test = dates_sorted.iloc[i+1:i+1+horizon]
            
            # Skip if not enough data
            if len(X_train) < 10 or len(X_test) < horizon:
                continue
            
            try:
                # Create a clone of the model
                from copy import deepcopy
                model_clone = deepcopy(model)
                
                # Fit on training data
                model_clone.fit(X_train, y_train)
                
                # Predict on test data
                y_pred = model_clone.predict(X_test)
                
                # Calculate metrics
                metrics = {
                    'rmse': np.sqrt(np.mean((y_test - y_pred) ** 2)),
                    'mae': np.mean(np.abs(y_test - y_pred)),
                    'start_date': dates_sorted.iloc[i].strftime('%Y-%m-%d'),
                    'end_date': dates_test.iloc[-1].strftime('%Y-%m-%d')
                }
                
                # Add seasonal metrics if possible
                if date_col in X_test.columns:
                    try:
                        seasonal_metrics = calculate_seasonal_error_metrics(
                            y_test, y_pred, X_test[date_col], seasonality_type
                        )
                        metrics['seasonal'] = seasonal_metrics
                    except Exception as e:
                        logger.warning(f"Error calculating seasonal metrics: {e}")
                
                horizon_metrics.append(metrics)
                
            except Exception as e:
                logger.warning(f"Error in forecast for horizon {horizon}, window {i}: {e}")
        
        # Calculate average metrics for this horizon
        if horizon_metrics:
            avg_rmse = np.mean([m['rmse'] for m in horizon_metrics])
            avg_mae = np.mean([m['mae'] for m in horizon_metrics])
            
            results['by_horizon'][horizon] = {
                'avg_rmse': avg_rmse,
                'avg_mae': avg_mae,
                'n_windows': len(horizon_metrics),
                'window_metrics': horizon_metrics[:5]  # Only include first 5 for brevity
            }
    
    # Assess how error grows with horizon
    if len(results['by_horizon']) >= 2:
        horizons_list = sorted(results['by_horizon'].keys())
        rmse_values = [results['by_horizon'][h]['avg_rmse'] for h in horizons_list]
        
        # Calculate error growth rate (simple linear approximation)
        if len(horizons_list) > 1:
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(horizons_list, rmse_values)
            
            results['error_growth'] = {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value**2,
                'relative_growth_rate': slope / rmse_values[0] if rmse_values[0] > 0 else float('nan')
            }
            
            # Assess growth rate
            relative_rate = results['error_growth']['relative_growth_rate']
            if relative_rate < 0.1:
                results['horizon_stability'] = 'Excellent'
            elif relative_rate < 0.2:
                results['horizon_stability'] = 'Good'
            elif relative_rate < 0.5:
                results['horizon_stability'] = 'Moderate'
            else:
                results['horizon_stability'] = 'Poor'
    
    return results


def generate_comprehensive_evaluation(model: Any,
                                     X_train: pd.DataFrame, y_train: pd.Series,
                                     X_test: pd.DataFrame, y_test: pd.Series,
                                     date_col: str = 'date',
                                     seasonality_type: str = 'monthly') -> Dict[str, Any]:
    """
    Generate a comprehensive evaluation report for a seasonal model.
    
    Args:
        model: Fitted model
        X_train: Training feature matrix
        y_train: Training target
        X_test: Test feature matrix
        y_test: Test target
        date_col: Name of date column
        seasonality_type: Type of seasonality
        
    Returns:
        Dictionary with comprehensive evaluation results
    """
    from seasonal_model_diagnostics import test_seasonality_significance
    
    # Create comprehensive report
    report = {
        'model_info': {
            'type': type(model).__name__,
            'seasonality_type': seasonality_type,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'date_range': {
                'train_start': X_train[date_col].min() if date_col in X_train else None,
                'train_end': X_train[date_col].max() if date_col in X_train else None,
                'test_start': X_test[date_col].min() if date_col in X_test else None,
                'test_end': X_test[date_col].max() if date_col in X_test else None
            }
        },
        'sections': {}
    }
    
    # Basic forecast evaluation
    from seasonal_error_metrics import seasonal_forecast_evaluation
    basic_eval = seasonal_forecast_evaluation(
        model, X_train, y_train, X_test, y_test, date_col, seasonality_type
    )
    
    # Update report with basic metrics
    report['sections']['basic_performance'] = {
        'metrics': basic_eval.get('metrics', {}),
        'seasonal_metrics': basic_eval.get('seasonal_metrics', {}),
        'directional_accuracy': basic_eval.get('directional_accuracy', {})
    }
    
    # Quality assessment
    quality = assess_forecast_quality(basic_eval)
    report['sections']['quality_assessment'] = quality
    
    # Test seasonality significance
    try:
        seasonality_significance = test_seasonality_significance(model)
        report['sections']['seasonality_significance'] = seasonality_significance
    except Exception as e:
        logger.warning(f"Error in seasonality significance test: {e}")
    
    # Add recommendations based on findings
    recommendations = []
    
    # Check overall quality
    if quality.get('overall') == 'Low quality forecast':
        recommendations.append(
            "Model performs poorly overall. Consider using a different modeling approach, adding more features, or focusing on shorter forecast horizons."
        )
    
    # Check for seasonal imbalance
    if quality.get('seasonal_balance') == 'Unbalanced':
        best_period = quality.get('best_period')
        worst_period = quality.get('worst_period')
        recommendations.append(
            f"Model performance varies significantly across seasons (best: {best_period}, worst: {worst_period}). "
            "Consider building separate models for problematic seasons or adding more season-specific features."
        )
    
    # Check seasonality significance
    seasonality = report['sections'].get('seasonality_significance', {})
    if seasonality.get('overall_significance') == 'Low':
        recommendations.append(
            "Seasonal components have low statistical significance. Consider simplifying the seasonal representation or using non-seasonal models."
        )
    
    # Add recommendations to report
    report['recommendations'] = recommendations
    
    return report