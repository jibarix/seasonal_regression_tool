"""
Data transformations module for time series analysis.
Provides various transformations with proper handling of edge cases.
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Union, Optional, Callable, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings

# Setup logging
logger = logging.getLogger(__name__)


class TimeSeriesTransformer:
    """
    Class for applying and reverting transformations to time series data.
    Handles various transformation types with proper tracking for inversion.
    """
    
    def __init__(self):
        """Initialize the transformer with empty transformation history."""
        self.transformation_history = {}
        self.scalers = {}
        
    def fit_transform(self, df: pd.DataFrame, 
                     columns: List[str],
                     transformations: List[str],
                     params: Optional[Dict] = None) -> pd.DataFrame:
        """
        Apply a sequence of transformations to specified columns.
        
        Args:
            df: DataFrame to transform
            columns: Columns to transform
            transformations: List of transformations to apply in sequence
            params: Optional parameters for transformations
            
        Returns:
            Transformed DataFrame
        """
        result = df.copy()
        
        if params is None:
            params = {}
            
        # Apply each transformation in sequence
        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found in DataFrame")
                continue
                
            # Initialize transformation history for this column
            if col not in self.transformation_history:
                self.transformation_history[col] = []
                
            # For tracking transformed values in each step
            current_series = result[col].copy()
            
            for transform in transformations:
                if transform == 'log':
                    # Get offset parameter or use default
                    offset = params.get('log_offset', 0)
                    
                    # Apply log transformation
                    result[col], log_offset = self._log_transform(current_series, offset)
                    
                    # Record transformation
                    self.transformation_history[col].append({
                        'type': 'log',
                        'offset': log_offset
                    })
                    
                elif transform == 'sqrt':
                    # Get offset parameter or use default
                    offset = params.get('sqrt_offset', 0)
                    
                    # Apply square root transformation
                    result[col], sqrt_offset = self._sqrt_transform(current_series, offset)
                    
                    # Record transformation
                    self.transformation_history[col].append({
                        'type': 'sqrt',
                        'offset': sqrt_offset
                    })
                    
                elif transform == 'diff':
                    # Get lag parameter or use default
                    lag = params.get('diff_lag', 1)
                    
                    # Apply differencing
                    result[col] = self._diff_transform(current_series, lag)
                    
                    # Record transformation
                    self.transformation_history[col].append({
                        'type': 'diff',
                        'lag': lag,
                        'first_values': current_series.iloc[:lag].tolist()
                    })
                    
                elif transform == 'seasonal_diff':
                    # Get seasonal period parameter
                    period = params.get('seasonal_period', 12)
                    
                    # Apply seasonal differencing
                    result[col] = self._seasonal_diff_transform(current_series, period)
                    
                    # Record transformation
                    self.transformation_history[col].append({
                        'type': 'seasonal_diff',
                        'period': period,
                        'first_values': current_series.iloc[:period].tolist()
                    })
                    
                elif transform == 'standardize':
                    # Get parameters
                    with_mean = params.get('with_mean', True)
                    with_std = params.get('with_std', True)
                    
                    # Apply standardization
                    result[col], scaler = self._standardize(current_series, with_mean, with_std)
                    
                    # Store scaler for inverse transformation
                    self.scalers[col] = scaler
                    
                    # Record transformation
                    self.transformation_history[col].append({
                        'type': 'standardize',
                        'with_mean': with_mean,
                        'with_std': with_std
                    })
                    
                elif transform == 'normalize':
                    # Apply normalization
                    result[col], scaler = self._normalize(current_series)
                    
                    # Store scaler for inverse transformation
                    self.scalers[col] = scaler
                    
                    # Record transformation
                    self.transformation_history[col].append({
                        'type': 'normalize'
                    })
                    
                elif transform == 'box_cox':
                    # Get lambda parameter
                    lmbda = params.get('box_cox_lambda', None)
                    
                    # Apply Box-Cox transformation
                    result[col], lmbda, offset = self._box_cox_transform(current_series, lmbda)
                    
                    # Record transformation
                    self.transformation_history[col].append({
                        'type': 'box_cox',
                        'lambda': lmbda,
                        'offset': offset
                    })
                    
                elif transform == 'yeo_johnson':
                    # Get lambda parameter
                    lmbda = params.get('yeo_johnson_lambda', None)
                    
                    # Apply Yeo-Johnson transformation
                    result[col], lmbda = self._yeo_johnson_transform(current_series, lmbda)
                    
                    # Record transformation
                    self.transformation_history[col].append({
                        'type': 'yeo_johnson',
                        'lambda': lmbda
                    })
                    
                else:
                    logger.warning(f"Unknown transformation: {transform}")
                    continue
                    
                # Update current series for next transformation in sequence
                current_series = result[col].copy()
                
        return result
        
    def inverse_transform(self, df: pd.DataFrame, 
                         columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Revert transformations applied to the specified columns.
        
        Args:
            df: DataFrame with transformed data
            columns: Columns to inverse transform (None for all transformed)
            
        Returns:
            DataFrame with original scale data
        """
        result = df.copy()
        
        if columns is None:
            columns = list(self.transformation_history.keys())
            
        for col in columns:
            if col not in self.transformation_history:
                logger.warning(f"No transformation history for column '{col}'")
                continue
                
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found in DataFrame")
                continue
                
            # Get transformation history for this column
            transformations = self.transformation_history[col]
            
            # Apply inverse transformations in reverse order
            current_series = result[col].copy()
            
            for transform in reversed(transformations):
                transform_type = transform['type']
                
                if transform_type == 'log':
                    # Apply inverse log transformation
                    result[col] = self._inverse_log_transform(current_series, transform['offset'])
                    
                elif transform_type == 'sqrt':
                    # Apply inverse sqrt transformation
                    result[col] = self._inverse_sqrt_transform(current_series, transform['offset'])
                    
                elif transform_type == 'diff':
                    # Apply inverse differencing
                    result[col] = self._inverse_diff_transform(
                        current_series, transform['lag'], transform['first_values']
                    )
                    
                elif transform_type == 'seasonal_diff':
                    # Apply inverse seasonal differencing
                    result[col] = self._inverse_seasonal_diff_transform(
                        current_series, transform['period'], transform['first_values']
                    )
                    
                elif transform_type == 'standardize':
                    # Apply inverse standardization
                    if col in self.scalers:
                        result[col] = self._inverse_standardize(current_series, self.scalers[col])
                    else:
                        logger.warning(f"Scaler for column '{col}' not found")
                        
                elif transform_type == 'normalize':
                    # Apply inverse normalization
                    if col in self.scalers:
                        result[col] = self._inverse_normalize(current_series, self.scalers[col])
                    else:
                        logger.warning(f"Scaler for column '{col}' not found")
                        
                elif transform_type == 'box_cox':
                    # Apply inverse Box-Cox transformation
                    result[col] = self._inverse_box_cox_transform(
                        current_series, transform['lambda'], transform['offset']
                    )
                    
                elif transform_type == 'yeo_johnson':
                    # Apply inverse Yeo-Johnson transformation
                    result[col] = self._inverse_yeo_johnson_transform(
                        current_series, transform['lambda']
                    )
                
                # Update current series for next inverse transformation
                current_series = result[col].copy()
                
        return result
    
    def _log_transform(self, series: pd.Series, offset: float = 0) -> Tuple[pd.Series, float]:
        """
        Apply log transformation with proper handling of zero/negative values.
        
        Args:
            series: Series to transform
            offset: Optional offset to add before taking log
            
        Returns:
            Tuple of (transformed series, offset used)
        """
        # Check if we need to adjust offset for negative values
        min_val = series.min()
        if min_val <= 0:
            # Set offset to ensure all values are positive
            offset = abs(min_val) + 1
            logger.info(f"Adjusting log transform offset to {offset} for negative values")
            
        # Apply log transformation
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            result = np.log(series + offset)
            
        return result, offset
        
    def _sqrt_transform(self, series: pd.Series, offset: float = 0) -> Tuple[pd.Series, float]:
        """
        Apply square root transformation with proper handling of negative values.
        
        Args:
            series: Series to transform
            offset: Optional offset to add before taking sqrt
            
        Returns:
            Tuple of (transformed series, offset used)
        """
        # Check if we need to adjust offset for negative values
        min_val = series.min()
        if min_val < 0:
            # Set offset to ensure all values are non-negative
            offset = abs(min_val) + 0.01
            logger.info(f"Adjusting sqrt transform offset to {offset} for negative values")
            
        # Apply sqrt transformation
        result = np.sqrt(series + offset)
        
        return result, offset
        
    def _diff_transform(self, series: pd.Series, lag: int = 1) -> pd.Series:
        """
        Apply differencing transformation.
        
        Args:
            series: Series to transform
            lag: Lag for differencing
            
        Returns:
            Differenced series
        """
        # Apply differencing
        result = series.diff(lag)
        
        # First 'lag' values will be NaN - keep them for inverse transformation
        return result
    
    def _seasonal_diff_transform(self, series: pd.Series, period: int = 12) -> pd.Series:
        """
        Apply seasonal differencing transformation.
        
        Args:
            series: Series to transform
            period: Seasonal period (e.g., 12 for monthly data with annual seasonality)
            
        Returns:
            Seasonally differenced series
        """
        # Apply seasonal differencing
        result = series.diff(period)
        
        # First 'period' values will be NaN - keep them for inverse transformation
        return result
    
    def _standardize(self, series: pd.Series, with_mean: bool = True, 
                    with_std: bool = True) -> Tuple[pd.Series, StandardScaler]:
        """
        Standardize series (z-score normalization).
        
        Args:
            series: Series to transform
            with_mean: Whether to center the data before scaling
            with_std: Whether to scale the data to unit variance
            
        Returns:
            Tuple of (standardized series, fitted scaler)
        """
        # Handle missing values
        valid_mask = ~series.isna()
        valid_indices = valid_mask[valid_mask].index
        
        # Create and fit scaler
        scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
        
        # Reshape for scikit-learn
        values = series[valid_indices].values.reshape(-1, 1)
        scaled_values = scaler.fit_transform(values).flatten()
        
        # Create result series (preserving NaN values)
        result = series.copy()
        result.loc[valid_indices] = scaled_values
        
        return result, scaler
    
    def _normalize(self, series: pd.Series) -> Tuple[pd.Series, MinMaxScaler]:
        """
        Normalize series to [0, 1] range.
        
        Args:
            series: Series to transform
            
        Returns:
            Tuple of (normalized series, fitted scaler)
        """
        # Handle missing values
        valid_mask = ~series.isna()
        valid_indices = valid_mask[valid_mask].index
        
        # Create and fit scaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Reshape for scikit-learn
        values = series[valid_indices].values.reshape(-1, 1)
        scaled_values = scaler.fit_transform(values).flatten()
        
        # Create result series (preserving NaN values)
        result = series.copy()
        result.loc[valid_indices] = scaled_values
        
        return result, scaler
    
    def _box_cox_transform(self, series: pd.Series, 
                          lmbda: Optional[float] = None) -> Tuple[pd.Series, float, float]:
        """
        Apply Box-Cox transformation.
        
        Args:
            series: Series to transform
            lmbda: Box-Cox transformation parameter (None for auto-selection)
            
        Returns:
            Tuple of (transformed series, lambda used, offset used)
        """
        from scipy import stats
        
        # Handle missing values
        valid_mask = ~series.isna()
        valid_indices = valid_mask[valid_mask].index
        
        # Box-Cox requires positive values
        min_val = series[valid_indices].min()
        offset = 0
        
        if min_val <= 0:
            offset = abs(min_val) + 1
            logger.info(f"Adding offset of {offset} for Box-Cox transformation")
        
        # Apply Box-Cox transformation
        shifted_values = series[valid_indices] + offset
        
        if lmbda is None:
            # Auto-select lambda
            transformed_values, lmbda = stats.boxcox(shifted_values)
            logger.info(f"Selected Box-Cox lambda: {lmbda}")
        else:
            # Use provided lambda
            transformed_values = stats.boxcox(shifted_values, lmbda=lmbda)
        
        # Create result series (preserving NaN values)
        result = series.copy()
        result.loc[valid_indices] = transformed_values
        
        return result, lmbda, offset
    
    def _yeo_johnson_transform(self, series: pd.Series, 
                              lmbda: Optional[float] = None) -> Tuple[pd.Series, float]:
        """
        Apply Yeo-Johnson transformation (works with negative values).
        
        Args:
            series: Series to transform
            lmbda: Yeo-Johnson transformation parameter (None for auto-selection)
            
        Returns:
            Tuple of (transformed series, lambda used)
        """
        from scipy import stats
        
        # Handle missing values
        valid_mask = ~series.isna()
        valid_indices = valid_mask[valid_mask].index
        
        # Apply Yeo-Johnson transformation
        values = series[valid_indices].values.reshape(-1, 1)
        
        if lmbda is None:
            # Auto-select lambda
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                try:
                    transformed_values, lmbda = stats.yeojohnson(values.flatten())
                    logger.info(f"Selected Yeo-Johnson lambda: {lmbda}")
                except:
                    logger.warning("Yeo-Johnson transformation failed, using identity transformation")
                    transformed_values = values.flatten()
                    lmbda = 1  # identity transformation
        else:
            # Use provided lambda
            try:
                transformed_values = stats.yeojohnson(values.flatten(), lmbda=lmbda)
            except:
                logger.warning("Yeo-Johnson transformation failed, using identity transformation")
                transformed_values = values.flatten()
        
        # Create result series (preserving NaN values)
        result = series.copy()
        result.loc[valid_indices] = transformed_values
        
        return result, lmbda
    
    def _inverse_log_transform(self, series: pd.Series, offset: float = 0) -> pd.Series:
        """
        Apply inverse log transformation.
        
        Args:
            series: Series to inverse transform
            offset: Offset used in original transformation
            
        Returns:
            Inverse transformed series
        """
        # Apply exponential and subtract offset
        result = np.exp(series) - offset
        return result
    
    def _inverse_sqrt_transform(self, series: pd.Series, offset: float = 0) -> pd.Series:
        """
        Apply inverse square root transformation.
        
        Args:
            series: Series to inverse transform
            offset: Offset used in original transformation
            
        Returns:
            Inverse transformed series
        """
        # Square the values and subtract offset
        result = series ** 2 - offset
        return result
    
    def _inverse_diff_transform(self, series: pd.Series, lag: int, 
                               first_values: List[float]) -> pd.Series:
        """
        Apply inverse differencing transformation.
        
        Args:
            series: Differenced series
            lag: Lag used in original differencing
            first_values: Original first 'lag' values before differencing
            
        Returns:
            Original undifferenced series
        """
        # Create result series (starting with NaN)
        result = pd.Series(index=series.index, dtype=float)
        
        # Set first 'lag' values from original series
        result.iloc[:lag] = first_values
        
        # Reconstruct the rest by cumulative sum
        for i in range(lag, len(series)):
            result.iloc[i] = result.iloc[i-lag] + series.iloc[i]
        
        return result
    
    def _inverse_seasonal_diff_transform(self, series: pd.Series, period: int, 
                                        first_values: List[float]) -> pd.Series:
        """
        Apply inverse seasonal differencing transformation.
        
        Args:
            series: Seasonally differenced series
            period: Seasonal period used in original differencing
            first_values: Original first 'period' values before differencing
            
        Returns:
            Original undifferenced series
        """
        # Create result series (starting with NaN)
        result = pd.Series(index=series.index, dtype=float)
        
        # Set first 'period' values from original series
        result.iloc[:period] = first_values
        
        # Reconstruct the rest by adding the seasonal difference to value from previous season
        for i in range(period, len(series)):
            result.iloc[i] = result.iloc[i-period] + series.iloc[i]
        
        return result
    
    def _inverse_standardize(self, series: pd.Series, scaler: StandardScaler) -> pd.Series:
        """
        Apply inverse standardization.
        
        Args:
            series: Standardized series
            scaler: Fitted StandardScaler used in original transformation
            
        Returns:
            Original unstandardized series
        """
        # Handle missing values
        valid_mask = ~series.isna()
        valid_indices = valid_mask[valid_mask].index
        
        # Reshape for scikit-learn
        values = series[valid_indices].values.reshape(-1, 1)
        original_values = scaler.inverse_transform(values).flatten()
        
        # Create result series (preserving NaN values)
        result = series.copy()
        result.loc[valid_indices] = original_values
        
        return result
    
    def _inverse_normalize(self, series: pd.Series, scaler: MinMaxScaler) -> pd.Series:
        """
        Apply inverse normalization.
        
        Args:
            series: Normalized series
            scaler: Fitted MinMaxScaler used in original transformation
            
        Returns:
            Original unnormalized series
        """
        # Handle missing values
        valid_mask = ~series.isna()
        valid_indices = valid_mask[valid_mask].index
        
        # Reshape for scikit-learn
        values = series[valid_indices].values.reshape(-1, 1)
        original_values = scaler.inverse_transform(values).flatten()
        
        # Create result series (preserving NaN values)
        result = series.copy()
        result.loc[valid_indices] = original_values
        
        return result
    
    def _inverse_box_cox_transform(self, series: pd.Series, lmbda: float, 
                                  offset: float = 0) -> pd.Series:
        """
        Apply inverse Box-Cox transformation.
        
        Args:
            series: Box-Cox transformed series
            lmbda: Lambda parameter used in original transformation
            offset: Offset used in original transformation
            
        Returns:
            Original untransformed series
        """
        from scipy import special
        
        # Handle missing values
        valid_mask = ~series.isna()
        
        # Apply inverse Box-Cox transformation
        if abs(lmbda) < 1e-10:  # close to zero
            result = np.exp(series)
        else:
            result = np.power(lmbda * series + 1, 1/lmbda)
        
        # Subtract offset
        result = result - offset
        
        # Handle NaN values
        result[~valid_mask] = np.nan
        
        return result
    
    def _inverse_yeo_johnson_transform(self, series: pd.Series, lmbda: float) -> pd.Series:
        """
        Apply inverse Yeo-Johnson transformation.
        
        Args:
            series: Yeo-Johnson transformed series
            lmbda: Lambda parameter used in original transformation
            
        Returns:
            Original untransformed series
        """
        # Handle missing values
        valid_mask = ~series.isna()
        valid_indices = valid_mask[valid_mask].index
        
        # Apply inverse Yeo-Johnson transformation
        values = series[valid_indices].values
        
        # Implementation based on the Yeo-Johnson formula
        transformed_values = np.zeros_like(values)
        
        # For values where x >= 0
        pos_mask = values >= 0
        if abs(lmbda - 2) > 1e-10:  # lambda != 2
            transformed_values[pos_mask] = np.power(values[pos_mask] * (lmbda - 1) + 1, 1/(lmbda - 1)) - 1
        else:  # lambda = 2
            transformed_values[pos_mask] = np.exp(values[pos_mask]) - 1
            
        # For values where x < 0
        neg_mask = ~pos_mask
        if abs(lmbda) > 1e-10:  # lambda != 0
            transformed_values[neg_mask] = 1 - np.power(-(2 - lmbda) * values[neg_mask] + 1, 1/(2 - lmbda))
        else:  # lambda = 0
            transformed_values[neg_mask] = 1 - np.exp(-values[neg_mask])
        
        # Create result series (preserving NaN values)
        result = series.copy()
        result.loc[valid_indices] = transformed_values
        
        return result


class TransformationPipeline:
    """
    Class for creating and executing a sequence of transformations.
    """
    
    def __init__(self):
        """Initialize the pipeline with empty transformer."""
        self.transformer = TimeSeriesTransformer()
        self.steps = []
        
    def add_step(self, columns: List[str], transformations: List[str], 
                params: Optional[Dict] = None) -> 'TransformationPipeline':
        """
        Add a transformation step to the pipeline.
        
        Args:
            columns: Columns to transform
            transformations: List of transformations to apply
            params: Optional parameters for transformations
            
        Returns:
            Self for method chaining
        """
        self.steps.append({
            'columns': columns,
            'transformations': transformations,
            'params': params or {}
        })
        
        return self
        
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all transformation steps in the pipeline.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        result = df.copy()
        
        for step in self.steps:
            result = self.transformer.fit_transform(
                result, 
                step['columns'], 
                step['transformations'],
                step['params']
            )
            
        return result
        
    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Revert all transformations in the pipeline.
        
        Args:
            df: DataFrame with transformed data
            
        Returns:
            DataFrame with original scale data
        """
        return self.transformer.inverse_transform(df)
    
    def clear(self) -> None:
        """Clear all steps in the pipeline."""
        self.steps = []
        self.transformer = TimeSeriesTransformer()


# Helper functions for common transformation combinations

def log_scale_transform(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Apply log transformation to specified columns.
    
    Args:
        df: DataFrame to transform
        columns: Columns to transform
        
    Returns:
        Transformed DataFrame
    """
    pipeline = TransformationPipeline()
    return pipeline.add_step(columns, ['log']).fit_transform(df)

def difference_transform(df: pd.DataFrame, columns: List[str], lag: int = 1) -> pd.DataFrame:
    """
    Apply differencing to specified columns.
    
    Args:
        df: DataFrame to transform
        columns: Columns to transform
        lag: Lag for differencing
        
    Returns:
        Transformed DataFrame
    """
    pipeline = TransformationPipeline()
    return pipeline.add_step(columns, ['diff'], {'diff_lag': lag}).fit_transform(df)

def seasonal_difference_transform(df: pd.DataFrame, columns: List[str], 
                                 period: int = 12) -> pd.DataFrame:
    """
    Apply seasonal differencing to specified columns.
    
    Args:
        df: DataFrame to transform
        columns: Columns to transform
        period: Seasonal period
        
    Returns:
        Transformed DataFrame
    """
    pipeline = TransformationPipeline()
    return pipeline.add_step(
        columns, ['seasonal_diff'], {'seasonal_period': period}
    ).fit_transform(df)

def normalize_columns(df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, TransformationPipeline]:
    """
    Normalize specified columns to [0, 1] range and return the pipeline for inverse transform.
    
    Args:
        df: DataFrame to transform
        columns: Columns to normalize
        
    Returns:
        Tuple of (transformed DataFrame, pipeline used)
    """
    pipeline = TransformationPipeline()
    transformed_df = pipeline.add_step(columns, ['normalize']).fit_transform(df)
    return transformed_df, pipeline

def standardize_columns(df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, TransformationPipeline]:
    """
    Standardize specified columns and return the pipeline for inverse transform.
    
    Args:
        df: DataFrame to transform
        columns: Columns to standardize
        
    Returns:
        Tuple of (transformed DataFrame, pipeline used)
    """
    pipeline = TransformationPipeline()
    transformed_df = pipeline.add_step(columns, ['standardize']).fit_transform(df)
    return transformed_df, pipeline