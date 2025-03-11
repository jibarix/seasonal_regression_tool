"""
Feature engineering module for time series data.
Provides functionality for creating Fourier terms, lagged variables,
and other transformations for time series analysis.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional, Tuple
import logging

# Setup logging
logger = logging.getLogger(__name__)

def create_fourier_terms(df: pd.DataFrame, 
                         date_col: str = 'date',
                         periods: List[int] = [12],  # Default to annual seasonality for monthly data
                         harmonics: int = 3) -> pd.DataFrame:
    """
    Create Fourier terms to model seasonality in time series data.
    
    Args:
        df: DataFrame containing date column
        date_col: Name of the date column
        periods: List of seasonal periods to model (e.g. [12] for annual in monthly data)
        harmonics: Number of harmonics to include for each period
        
    Returns:
        DataFrame with added Fourier terms
    """
    if df.empty:
        return df
    
    result = df.copy()
    
    if date_col not in result.columns:
        logger.warning(f"Date column '{date_col}' not found in DataFrame")
        return result
    
    # Ensure date column is datetime
    result[date_col] = pd.to_datetime(result[date_col])
    
    # Get continuous time index (starting from 0)
    time_index = np.arange(len(result))
    
    # Generate Fourier terms for each period
    for period in periods:
        for harm in range(1, harmonics + 1):
            # Create sine term
            result[f'sin_{period}_{harm}'] = np.sin(2 * np.pi * harm * time_index / period)
            
            # Create cosine term
            result[f'cos_{period}_{harm}'] = np.cos(2 * np.pi * harm * time_index / period)
    
    return result

def create_lagged_features(df: pd.DataFrame, 
                          columns: List[str], 
                          lags: List[int]) -> pd.DataFrame:
    """
    Create lagged versions of specified columns.
    
    Args:
        df: DataFrame with time series data
        columns: List of columns to create lags for
        lags: List of lag values (e.g. [1, 2, 3] for t-1, t-2, t-3)
        
    Returns:
        DataFrame with added lagged features
    """
    result = df.copy()
    
    for col in columns:
        if col not in result.columns:
            logger.warning(f"Column '{col}' not found in DataFrame")
            continue
        
        for lag in lags:
            # Create lagged column
            result[f'{col}_lag_{lag}'] = result[col].shift(lag)
    
    return result

def create_rolling_features(df: pd.DataFrame,
                           columns: List[str],
                           windows: List[int] = [3, 6, 12],
                           functions: List[str] = ['mean', 'std', 'min', 'max']) -> pd.DataFrame:
    """
    Create rolling window features for specified columns.
    
    Args:
        df: DataFrame with time series data
        columns: List of columns to create rolling features for
        windows: List of window sizes
        functions: List of aggregation functions to apply
        
    Returns:
        DataFrame with added rolling features
    """
    result = df.copy()
    
    for col in columns:
        if col not in result.columns:
            logger.warning(f"Column '{col}' not found in DataFrame")
            continue
        
        for window in windows:
            rolling = result[col].rolling(window=window, min_periods=1)
            
            for func in functions:
                # Get the rolling aggregate
                if func == 'mean':
                    result[f'{col}_roll_{window}_{func}'] = rolling.mean()
                elif func == 'std':
                    result[f'{col}_roll_{window}_{func}'] = rolling.std()
                elif func == 'min':
                    result[f'{col}_roll_{window}_{func}'] = rolling.min()
                elif func == 'max':
                    result[f'{col}_roll_{window}_{func}'] = rolling.max()
                else:
                    logger.warning(f"Unsupported rolling function: {func}")
    
    return result

def safe_log_transform(df: pd.DataFrame, columns: List[str], offset: float = 1.0) -> pd.DataFrame:
    """
    Apply log transformation safely handling zeros and negative values.
    
    Args:
        df: DataFrame with columns to transform
        columns: List of columns to log transform
        offset: Value to add before taking log (for handling zeros/negatives)
        
    Returns:
        DataFrame with log-transformed columns
    """
    result = df.copy()
    
    for col in columns:
        if col not in result.columns:
            logger.warning(f"Column '{col}' not found in DataFrame")
            continue
        
        # Check for negative or zero values
        min_val = result[col].min()
        col_offset = offset
        
        if min_val <= 0:
            # Adjust offset for negative values
            col_offset = abs(min_val) + offset
            logger.info(f"Adjusted log transform offset to {col_offset} for column '{col}'")
        
        # Apply log transformation
        result[f'{col}_log'] = np.log(result[col] + col_offset)
    
    return result

def create_date_features(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """
    Extract additional date features from the date column.
    
    Args:
        df: DataFrame containing date column
        date_col: Name of the date column
        
    Returns:
        DataFrame with added date features
    """
    if df.empty:
        return df
    
    result = df.copy()
    
    if date_col not in result.columns:
        logger.warning(f"Date column '{date_col}' not found in DataFrame")
        return result
    
    # Ensure date column is datetime
    result[date_col] = pd.to_datetime(result[date_col])
    
    # Extract date components
    result['year'] = result[date_col].dt.year
    result['month'] = result[date_col].dt.month
    result['quarter'] = result[date_col].dt.quarter
    result['day_of_year'] = result[date_col].dt.dayofyear
    result['day_of_month'] = result[date_col].dt.day
    result['week_of_year'] = result[date_col].dt.isocalendar().week
    
    # Create cyclical features for month and quarter
    result['month_sin'] = np.sin(2 * np.pi * result['month'] / 12)
    result['month_cos'] = np.cos(2 * np.pi * result['month'] / 12)
    result['quarter_sin'] = np.sin(2 * np.pi * result['quarter'] / 4)
    result['quarter_cos'] = np.cos(2 * np.pi * result['quarter'] / 4)
    
    return result

def create_interaction_features(df: pd.DataFrame, 
                               feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
    """
    Create interaction features between pairs of columns.
    
    Args:
        df: DataFrame with columns to interact
        feature_pairs: List of tuples with column name pairs to interact
        
    Returns:
        DataFrame with added interaction features
    """
    result = df.copy()
    
    for col1, col2 in feature_pairs:
        if col1 not in result.columns:
            logger.warning(f"Column '{col1}' not found in DataFrame")
            continue
            
        if col2 not in result.columns:
            logger.warning(f"Column '{col2}' not found in DataFrame")
            continue
        
        # Create interaction feature
        result[f'{col1}_{col2}_interaction'] = result[col1] * result[col2]
    
    return result

def engineer_features(df: pd.DataFrame, 
                     target_col: str,
                     date_col: str = 'date',
                     numeric_cols: Optional[List[str]] = None,
                     create_lags: bool = True,
                     create_fourier: bool = True,
                     create_rolling: bool = True,
                     log_transform: bool = True,
                     max_lag: int = 6) -> pd.DataFrame:
    """
    Main function to engineer all features for model training.
    
    Args:
        df: DataFrame with time series data
        target_col: Name of the target column
        date_col: Name of the date column
        numeric_cols: List of numeric columns to use as features (None for auto-detect)
        create_lags: Whether to create lagged features
        create_fourier: Whether to create Fourier terms
        create_rolling: Whether to create rolling features
        log_transform: Whether to apply log transformation to numeric features
        max_lag: Maximum lag to create for features
        
    Returns:
        DataFrame with engineered features
    """
    if df.empty:
        return df
    
    result = df.copy()
    
    # Auto-detect numeric columns if not provided
    if numeric_cols is None:
        numeric_cols = result.select_dtypes(include=['number']).columns.tolist()
        
        # Remove date column and target from numeric columns for feature engineering
        if date_col in numeric_cols:
            numeric_cols.remove(date_col)
        
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
    
    # Add date features
    result = create_date_features(result, date_col)
    
    # Create Fourier terms for seasonality
    if create_fourier:
        result = create_fourier_terms(result, date_col, periods=[4, 12], harmonics=2)
    
    # Create lagged features for target and other numeric columns
    if create_lags:
        lag_cols = numeric_cols.copy()
        if target_col not in lag_cols and target_col in result.columns:
            lag_cols.append(target_col)
        
        # Create lags from 1 to max_lag
        result = create_lagged_features(result, lag_cols, list(range(1, max_lag + 1)))
    
    # Create rolling window features
    if create_rolling:
        roll_cols = numeric_cols.copy()
        if target_col not in roll_cols and target_col in result.columns:
            roll_cols.append(target_col)
        
        result = create_rolling_features(result, roll_cols, windows=[3, 6, 12], 
                                       functions=['mean', 'std'])
    
    # Apply log transformation to numeric features
    if log_transform:
        log_cols = numeric_cols.copy()
        if target_col not in log_cols and target_col in result.columns:
            log_cols.append(target_col)
        
        result = safe_log_transform(result, log_cols)
    
    return result