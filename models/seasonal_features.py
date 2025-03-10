"""
Feature generation module for seasonal time series analysis.
Provides functions to create and transform seasonal features.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Union, Optional, Tuple, Any
from datetime import datetime
import warnings

# Setup logging
logger = logging.getLogger(__name__)


def add_time_variables(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """
    Add time-related variables (quarters, months, etc.) to a DataFrame.
    
    Args:
        df: DataFrame containing date column
        date_col: Name of the date column
        
    Returns:
        DataFrame with additional time variables
    """
    if df.empty:
        return df
    
    # Create a copy to avoid modifying the original
    result = df.copy()
    
    # Ensure date column is datetime
    result[date_col] = pd.to_datetime(result[date_col])
    
    # Add year
    result['year'] = result[date_col].dt.year
    
    # Add month (1-12)
    result['month'] = result[date_col].dt.month
    
    # Add month name
    result['month_name'] = result[date_col].dt.strftime('%b')
    
    # Add month end date (the actual period the data represents)
    result['month_end'] = result[date_col] + pd.offsets.MonthEnd(0)
    
    # Add quarter number (1-4)
    result['quarter'] = result[date_col].dt.quarter
    
    # Add quarter end date
    result['quarter_end'] = result[date_col] + pd.offsets.QuarterEnd(0)
    
    # Add dummy variables for quarters
    for q in range(1, 5):
        result[f'Q{q}'] = (result['quarter'] == q).astype(int)
    
    # Add period label
    result['period'] = result[date_col].dt.strftime('%b %Y')
    
    logger.debug(f"Added time variables to DataFrame")
    return result


def add_month_dummies(df: pd.DataFrame, date_col: str = 'date', 
                     drop_first: bool = True) -> pd.DataFrame:
    """
    Add monthly dummy variables to the base features.
    
    Args:
        df: DataFrame with base features
        date_col: Name of date column
        drop_first: Whether to drop the first dummy to avoid collinearity
        
    Returns:
        DataFrame with month dummies added
    """
    # Create a copy to avoid modifying the input
    result = df.copy()
    
    # Ensure date column is datetime
    result[date_col] = pd.to_datetime(result[date_col])
    
    # Extract month and create dummies - ensure numeric month values
    months = result[date_col].dt.month.astype(int)
    
    # Create dummy variables
    if drop_first:
        # Create 11 dummies (dropping January)
        for month in range(2, 13):
            result[f'month_{month}'] = (months == month).astype(int)
            logger.debug(f"Created dummy for month {month}")
    else:
        # Create all 12 dummies
        for month in range(1, 13):
            result[f'month_{month}'] = (months == month).astype(int)
            logger.debug(f"Created dummy for month {month}")
    
    logger.info(f"Added {'11' if drop_first else '12'} month dummy variables")
    return result


def add_quarter_dummies(df: pd.DataFrame, date_col: str = 'date',
                       drop_first: bool = True) -> pd.DataFrame:
    """
    Add quarterly dummy variables to the base features.
    
    Args:
        df: DataFrame with base features
        date_col: Name of date column
        drop_first: Whether to drop the first dummy to avoid collinearity
        
    Returns:
        DataFrame with quarter dummies added
    """
    # Create a copy to avoid modifying the input
    result = df.copy()
    
    # Ensure date column is datetime
    result[date_col] = pd.to_datetime(result[date_col])
    
    # Extract quarter and create dummies - ensure numeric quarter values
    quarters = result[date_col].dt.quarter.astype(int)
    
    # Create dummy variables
    if drop_first:
        # Create 3 dummies (dropping Q1)
        for quarter in range(2, 5):
            result[f'quarter_{quarter}'] = (quarters == quarter).astype(int)
            logger.debug(f"Created dummy for quarter {quarter}")
    else:
        # Create all 4 dummies
        for quarter in range(1, 5):
            result[f'quarter_{quarter}'] = (quarters == quarter).astype(int)
            logger.debug(f"Created dummy for quarter {quarter}")
    
    logger.info(f"Added {'3' if drop_first else '4'} quarter dummy variables")
    return result


def add_fourier_terms(df: pd.DataFrame, date_col: str = 'date', 
                     period: int = 12, harmonics: int = 2) -> pd.DataFrame:
    """
    Add Fourier terms (sine and cosine) for representing seasonality.
    
    Args:
        df: DataFrame with base features
        date_col: Name of date column
        period: Seasonal period (e.g., 12 for monthly data with yearly cycle)
        harmonics: Number of harmonics to include
        
    Returns:
        DataFrame with Fourier terms added
    """
    # Create a copy to avoid modifying the input
    result = df.copy()
    
    # Ensure date column is datetime
    result[date_col] = pd.to_datetime(result[date_col])
    
    # Get fractional time through the period
    # For monthly data, this is the month (1-12) divided by 12
    if period == 12:  # Monthly data
        time_in_period = result[date_col].dt.month
    elif period == 4:  # Quarterly data
        time_in_period = result[date_col].dt.quarter
    else:
        # For custom periods, create a continuous time index
        # This assumes evenly spaced observations
        time_in_period = np.arange(len(result)) % period + 1
    
    # Create Fourier terms for each harmonic
    for harm in range(1, harmonics + 1):
        # Create sine term
        sin_name = f'sin_h{harm}'
        result[sin_name] = np.sin(2 * np.pi * harm * time_in_period / period)
        
        # Create cosine term
        cos_name = f'cos_h{harm}'
        result[cos_name] = np.cos(2 * np.pi * harm * time_in_period / period)
        
        logger.debug(f"Created Fourier terms for harmonic {harm}")
    
    logger.info(f"Added {harmonics * 2} Fourier terms (sine and cosine) for period {period}")
    return result


def add_seasonal_components(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """
    Add trigonometric seasonal components and trend variables.
    Uses annual and semi-annual cycles for monthly data.
    
    Args:
        df: DataFrame with base features
        date_col: Name of date column
        
    Returns:
        DataFrame with trigonometric seasonal components added
    """
    # Create a copy to avoid modifying the input
    result = df.copy()
    
    # Ensure date column is datetime
    result[date_col] = pd.to_datetime(result[date_col])
    
    # Add trend variables - compute months since start of data
    dates = result[date_col]
    min_date = dates.min()
    result['trend'] = (dates - min_date).dt.days / 30  # Trend in months
    result['trend_squared'] = result['trend'] ** 2
    
    # Extract month for seasonal components
    month_num = dates.dt.month
    
    # Create sine and cosine components for annual seasonality
    result['sin_annual'] = np.sin(2 * np.pi * month_num / 12)
    result['cos_annual'] = np.cos(2 * np.pi * month_num / 12)
    
    # Create sine and cosine components for semi-annual seasonality
    result['sin_semiannual'] = np.sin(4 * np.pi * month_num / 12)
    result['cos_semiannual'] = np.cos(4 * np.pi * month_num / 12)
    
    logger.info("Added trigonometric seasonal components and trend variables")
    return result


def create_seasonal_lags(df: pd.DataFrame, variables: List[str], 
                        lag_years: int = 1, date_col: str = 'date') -> pd.DataFrame:
    """
    Create seasonal lags (e.g., values from same month in previous years).
    
    Args:
        df: DataFrame with time series data
        variables: List of variables to create seasonal lags for
        lag_years: Number of years to lag
        date_col: Name of date column
        
    Returns:
        DataFrame with added seasonal lag features
    """
    # Create a copy to avoid modifying the original
    result = df.copy()
    
    # Sort by date
    result = result.sort_values(date_col).reset_index(drop=True)
    
    # Ensure date column is datetime
    result[date_col] = pd.to_datetime(result[date_col])
    
    # Get frequency information
    if 'month' not in result.columns:
        result['month'] = result[date_col].dt.month
    
    if 'quarter' not in result.columns:
        result['quarter'] = result[date_col].dt.quarter
    
    if 'year' not in result.columns:
        result['year'] = result[date_col].dt.year
    
    # Create seasonal lags
    for var in variables:
        if var not in result.columns:
            logger.warning(f"Variable '{var}' not found in DataFrame")
            continue
        
        # Create a temporary DataFrame with year, month/quarter, and value
        temp_df = result[['year', 'month', 'quarter', var]].copy()
        
        # Create lags for each year
        for lag in range(1, lag_years + 1):
            lag_name = f"{var}_seasonal_lag{lag}"
            
            # Create lag by matching month and year-lag
            result[lag_name] = np.nan
            
            # For each unique month
            for month in range(1, 13):
                # For each unique year
                for year in result['year'].unique():
                    # Current month-year
                    current_mask = (result['month'] == month) & (result['year'] == year)
                    
                    # Previous year (same month)
                    prev_year = year - lag
                    prev_mask = (temp_df['month'] == month) & (temp_df['year'] == prev_year)
                    
                    # If we have data for the previous year
                    if prev_mask.any():
                        # Get the lagged value
                        lag_value = temp_df.loc[prev_mask, var].iloc[0]
                        
                        # Assign to current month-year
                        result.loc[current_mask, lag_name] = lag_value
            
            logger.debug(f"Created seasonal lag feature: {lag_name}")
    
    logger.info(f"Created {lag_years} seasonal lag features for {len(variables)} variables")
    return result


def create_growth_rate_features(df: pd.DataFrame, variables: List[str], 
                              date_col: str = 'date') -> pd.DataFrame:
    """
    Create year-over-year and month-over-month growth rate features.
    
    Args:
        df: DataFrame with time series data
        variables: List of variables to create growth rates for
        date_col: Name of date column
        
    Returns:
        DataFrame with added growth rate features
    """
    # Create a copy to avoid modifying the original
    result = df.copy()
    
    # Sort by date
    result = result.sort_values(date_col).reset_index(drop=True)
    
    # Ensure date column is datetime
    result[date_col] = pd.to_datetime(result[date_col])
    
    # Add month-over-month growth rates
    for var in variables:
        if var not in result.columns:
            logger.warning(f"Variable '{var}' not found in DataFrame")
            continue
        
        # Month-over-month growth rate
        mom_name = f"{var}_mom_growth"
        
        # Handle zeros and prevent division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            mom_growth = result[var].pct_change(1)
            
        # Replace infinite values with NaN
        result[mom_name] = mom_growth.replace([np.inf, -np.inf], np.nan)
        
        logger.debug(f"Created month-over-month growth rate: {mom_name}")
    
    # Create seasonal lags for year-over-year growth
    seasonal_lags_df = create_seasonal_lags(result, variables, lag_years=1, date_col=date_col)
    
    # Use seasonal lags to calculate year-over-year growth rates
    for var in variables:
        if var not in result.columns:
            continue
        
        # Year-over-year growth rate
        yoy_name = f"{var}_yoy_growth"
        lag_name = f"{var}_seasonal_lag1"
        
        if lag_name in seasonal_lags_df.columns:
            # Get the seasonal lag
            seasonal_lag = seasonal_lags_df[lag_name]
            
            # Calculate year-over-year growth
            with np.errstate(divide='ignore', invalid='ignore'):
                yoy_growth = (result[var] - seasonal_lag) / seasonal_lag
                
            # Replace infinite values with NaN
            result[yoy_name] = yoy_growth.replace([np.inf, -np.inf], np.nan)
            
            logger.debug(f"Created year-over-year growth rate: {yoy_name}")
    
    logger.info(f"Created growth rate features for {len(variables)} variables")
    return result


def create_seasonal_indicators(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """
    Create various seasonal indicator features, like holiday periods,
    start/end of quarter, etc.
    
    Args:
        df: DataFrame with time series data
        date_col: Name of date column
        
    Returns:
        DataFrame with added seasonal indicator features
    """
    # Create a copy to avoid modifying the original
    result = df.copy()
    
    # Ensure date column is datetime
    result[date_col] = pd.to_datetime(result[date_col])
    
    # Extract date components
    if 'month' not in result.columns:
        result['month'] = result[date_col].dt.month
    
    if 'quarter' not in result.columns:
        result['quarter'] = result[date_col].dt.quarter
    
    # End of quarter indicator (last month in quarter)
    result['end_of_quarter'] = ((result['month'] % 3) == 0).astype(int)
    
    # Start of quarter indicator (first month in quarter)
    result['start_of_quarter'] = ((result['month'] % 3) == 1).astype(int)
    
    # Mid-quarter indicator (middle month in quarter)
    result['mid_quarter'] = ((result['month'] % 3) == 2).astype(int)
    
    # Holiday season indicator (November-December)
    result['holiday_season'] = result['month'].isin([11, 12]).astype(int)
    
    # Summer months indicator (June-August)
    result['summer_months'] = result['month'].isin([6, 7, 8]).astype(int)
    
    # Winter months indicator (December-February)
    result['winter_months'] = result['month'].isin([12, 1, 2]).astype(int)
    
    # Spring months indicator (March-May)
    result['spring_months'] = result['month'].isin([3, 4, 5]).astype(int)
    
    # Fall months indicator (September-November)
    result['fall_months'] = result['month'].isin([9, 10, 11]).astype(int)
    
    logger.info("Added seasonal indicator features")
    return result


def prepare_seasonal_features(df: pd.DataFrame, feature_type: str = 'all',
                             date_col: str = 'date') -> pd.DataFrame:
    """
    Prepare seasonal features of specified type for modeling.
    
    Args:
        df: DataFrame with time series data
        feature_type: Type of seasonal features to create 
                     ('dummy', 'fourier', 'trig', 'all')
        date_col: Name of date column
        
    Returns:
        DataFrame with seasonal features added
    """
    # Create a copy to avoid modifying the original
    result = df.copy()
    
    # Add basic time variables
    result = add_time_variables(result, date_col)
    
    # Add features based on type
    if feature_type in ['dummy', 'all']:
        # Add monthly dummy variables
        result = add_month_dummies(result, date_col)
        
    if feature_type in ['fourier', 'all']:
        # Add Fourier terms for yearly cycle
        result = add_fourier_terms(result, date_col, period=12, harmonics=2)
        
    if feature_type in ['trig', 'all']:
        # Add trigonometric components
        result = add_seasonal_components(result, date_col)
        
    if feature_type in ['indicators', 'all']:
        # Add seasonal indicators
        result = create_seasonal_indicators(result, date_col)
    
    logger.info(f"Prepared seasonal features of type '{feature_type}'")
    return result


def detect_data_frequency(df: pd.DataFrame, date_col: str = 'date') -> str:
    """
    Detect the frequency of the time series data.
    
    Args:
        df: DataFrame with time series data
        date_col: Name of date column
        
    Returns:
        Detected frequency ('monthly', 'quarterly', 'annual', 'unknown')
    """
    if df.empty or len(df) < 2:
        return 'unknown'
    
    # Ensure date column is datetime
    if date_col in df.columns:
        dates = pd.to_datetime(df[date_col]).sort_values()
    else:
        logger.warning(f"Date column '{date_col}' not found in DataFrame")
        return 'unknown'
    
    # Calculate the difference between consecutive dates
    date_diffs = dates.diff().dropna()
    
    if date_diffs.empty:
        return 'unknown'
    
    # Use median to avoid outliers
    median_days = date_diffs.dt.days.median()
    
    # Classification based on median days difference
    if 25 <= median_days <= 32:
        return 'monthly'
    elif 85 <= median_days <= 95:
        return 'quarterly'
    elif 350 <= median_days <= 380:
        return 'annual'
    else:
        # Try to infer from pattern of months
        if len(dates) >= 4:
            months = dates.dt.month.tolist()
            month_diffs = np.diff(months)
            
            # Check if the pattern is consistent with quarterly data
            # (differences of 3 months)
            if all(diff in [3, -9] for diff in month_diffs):
                return 'quarterly'
            # Check if all dates are the same month (annual data)
            elif len(set(months)) == 1:
                return 'annual'
                
        logger.warning(f"Could not clearly determine frequency. Median days: {median_days}")
        return 'unknown'


def get_period_boundaries(df: pd.DataFrame, date_col: str = 'date', 
                        seasonal_type: str = 'monthly') -> pd.DataFrame:
    """
    Get the start and end dates for each seasonal period.
    
    Args:
        df: DataFrame with time series data
        date_col: Name of date column
        seasonal_type: Type of seasonality ('monthly', 'quarterly', 'annual')
        
    Returns:
        DataFrame with period boundaries
    """
    # Ensure date column is datetime
    dates = pd.to_datetime(df[date_col])
    
    # Get unique years
    years = dates.dt.year.unique()
    
    periods = []
    
    if seasonal_type == 'monthly':
        # Get boundaries for each month in each year
        for year in years:
            for month in range(1, 13):
                # Find all dates in this month and year
                mask = (dates.dt.year == year) & (dates.dt.month == month)
                
                if mask.any():
                    start_date = dates[mask].min()
                    end_date = dates[mask].max()
                    
                    periods.append({
                        'year': year,
                        'period': month,
                        'period_name': f"{start_date.strftime('%b')} {year}",
                        'start_date': start_date,
                        'end_date': end_date
                    })
    
    elif seasonal_type == 'quarterly':
        # Get boundaries for each quarter in each year
        for year in years:
            for quarter in range(1, 5):
                # Find all dates in this quarter and year
                mask = (dates.dt.year == year) & (dates.dt.quarter == quarter)
                
                if mask.any():
                    start_date = dates[mask].min()
                    end_date = dates[mask].max()
                    
                    periods.append({
                        'year': year,
                        'period': quarter,
                        'period_name': f"Q{quarter} {year}",
                        'start_date': start_date,
                        'end_date': end_date
                    })
    
    elif seasonal_type == 'annual':
        # Get boundaries for each year
        for year in years:
            # Find all dates in this year
            mask = (dates.dt.year == year)
            
            if mask.any():
                start_date = dates[mask].min()
                end_date = dates[mask].max()
                
                periods.append({
                    'year': year,
                    'period': year,
                    'period_name': str(year),
                    'start_date': start_date,
                    'end_date': end_date
                })
    
    # Create DataFrame from periods
    period_df = pd.DataFrame(periods)
    
    return period_df


def create_seasonal_pattern_features(df: pd.DataFrame, date_col: str = 'date',
                                    known_patterns: Dict[str, List[int]] = None) -> pd.DataFrame:
    """
    Create features for known seasonal patterns.
    
    Args:
        df: DataFrame with time series data
        date_col: Name of date column
        known_patterns: Dictionary mapping pattern names to lists of period indices
                      (e.g., {'summer_peak': [6, 7, 8]} for summer months)
        
    Returns:
        DataFrame with added seasonal pattern features
    """
    # Create a copy to avoid modifying the original
    result = df.copy()
    
    # Ensure date column is datetime
    result[date_col] = pd.to_datetime(result[date_col])
    
    # Extract month
    result['month'] = result[date_col].dt.month
    
    # Use default patterns if none provided
    if known_patterns is None:
        known_patterns = {
            'winter': [12, 1, 2],
            'spring': [3, 4, 5],
            'summer': [6, 7, 8],
            'fall': [9, 10, 11],
            'holiday_season': [11, 12],
            'quarter_start': [1, 4, 7, 10],
            'quarter_end': [3, 6, 9, 12],
        }
    
    # Create features for each pattern
    for pattern_name, months in known_patterns.items():
        # Create indicator for this pattern
        result[f'pattern_{pattern_name}'] = result['month'].isin(months).astype(int)
        logger.debug(f"Created seasonal pattern feature: pattern_{pattern_name}")
    
    logger.info(f"Created {len(known_patterns)} seasonal pattern features")
    return result