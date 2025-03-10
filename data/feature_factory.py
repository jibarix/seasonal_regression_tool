"""
Feature factory module for time series analysis.
Creates standard and specialized features for economic time series.
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Union, Optional, Tuple
from datetime import datetime
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings

# Setup logging
logger = logging.getLogger(__name__)


class FeatureFactory:
    """
    Class for creating and managing features for time series analysis.
    Provides methods for lag creation, seasonal components, interactions,
    and feature selection.
    """
    
    def __init__(self, date_col: str = 'date'):
        """
        Initialize the feature factory.
        
        Args:
            date_col: Name of the date column
        """
        self.date_col = date_col
        self.created_features = {}
        self.dropped_features = []
        
    def create_lags(self, df: pd.DataFrame, variables: List[str], 
                   max_lag: int = 6, drop_na: bool = False) -> pd.DataFrame:
        """
        Create lagged versions of specified variables.
        
        Args:
            df: DataFrame with time series data
            variables: List of variables to create lags for
            max_lag: Maximum number of lags to create
            drop_na: Whether to drop rows with NaN values
            
        Returns:
            DataFrame with added lag features
        """
        if max_lag < 1:
            logger.warning("max_lag must be at least 1")
            return df.copy()
            
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Sort by date if date column exists
        if self.date_col in result.columns:
            result = result.sort_values(self.date_col)
            
        # Create lag features for each variable
        lag_features = []
        
        for var in variables:
            if var not in result.columns:
                logger.warning(f"Variable '{var}' not found in DataFrame")
                continue
                
            # Create lags from 1 to max_lag
            for lag in range(1, max_lag + 1):
                lag_name = f"{var}_lag{lag}"
                result[lag_name] = result[var].shift(lag)
                lag_features.append(lag_name)
                
                logger.debug(f"Created lag feature: {lag_name}")
        
        # Track created features
        self.created_features['lags'] = lag_features
        
        # Drop rows with NaN values if requested
        if drop_na:
            orig_len = len(result)
            result = result.dropna()
            dropped = orig_len - len(result)
            logger.info(f"Dropped {dropped} rows with NaN values")
            
        logger.info(f"Created {len(lag_features)} lag features")
        return result
    
    def create_diff_features(self, df: pd.DataFrame, variables: List[str],
                            periods: List[int] = [1], drop_na: bool = False) -> pd.DataFrame:
        """
        Create differenced features for specified variables.
        
        Args:
            df: DataFrame with time series data
            variables: List of variables to create differences for
            periods: List of periods to difference
            drop_na: Whether to drop rows with NaN values
            
        Returns:
            DataFrame with added difference features
        """
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Sort by date if date column exists
        if self.date_col in result.columns:
            result = result.sort_values(self.date_col)
            
        # Create difference features for each variable and period
        diff_features = []
        
        for var in variables:
            if var not in result.columns:
                logger.warning(f"Variable '{var}' not found in DataFrame")
                continue
                
            # Create differences for each period
            for period in periods:
                diff_name = f"{var}_diff{period}"
                result[diff_name] = result[var].diff(period)
                diff_features.append(diff_name)
                
                logger.debug(f"Created difference feature: {diff_name}")
        
        # Track created features
        self.created_features['diffs'] = diff_features
        
        # Drop rows with NaN values if requested
        if drop_na:
            orig_len = len(result)
            result = result.dropna()
            dropped = orig_len - len(result)
            logger.info(f"Dropped {dropped} rows with NaN values")
            
        logger.info(f"Created {len(diff_features)} difference features")
        return result
        
    def create_pct_change_features(self, df: pd.DataFrame, variables: List[str],
                                  periods: List[int] = [1, 12], 
                                  drop_na: bool = False) -> pd.DataFrame:
        """
        Create percentage change features for specified variables.
        
        Args:
            df: DataFrame with time series data
            variables: List of variables to create percentage changes for
            periods: List of periods for percentage change
            drop_na: Whether to drop rows with NaN values
            
        Returns:
            DataFrame with added percentage change features
        """
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Sort by date if date column exists
        if self.date_col in result.columns:
            result = result.sort_values(self.date_col)
            
        # Create percentage change features for each variable and period
        pct_features = []
        
        for var in variables:
            if var not in result.columns:
                logger.warning(f"Variable '{var}' not found in DataFrame")
                continue
                
            # Create percentage changes for each period
            for period in periods:
                pct_name = f"{var}_pct{period}"
                
                # Handle zeros and prevent division by zero
                with np.errstate(divide='ignore', invalid='ignore'):
                    pct_change = result[var].pct_change(period)
                    
                # Replace infinite values with NaN
                result[pct_name] = pct_change.replace([np.inf, -np.inf], np.nan)
                
                pct_features.append(pct_name)
                logger.debug(f"Created percentage change feature: {pct_name}")
        
        # Track created features
        self.created_features['pct_changes'] = pct_features
        
        # Drop rows with NaN values if requested
        if drop_na:
            orig_len = len(result)
            result = result.dropna()
            dropped = orig_len - len(result)
            logger.info(f"Dropped {dropped} rows with NaN values")
            
        logger.info(f"Created {len(pct_features)} percentage change features")
        return result
        
    def create_rolling_features(self, df: pd.DataFrame, variables: List[str],
                               windows: List[int] = [3, 6, 12],
                               functions: List[str] = ['mean', 'std'],
                               drop_na: bool = False) -> pd.DataFrame:
        """
        Create rolling window features for specified variables.
        
        Args:
            df: DataFrame with time series data
            variables: List of variables to create rolling features for
            windows: List of window sizes
            functions: List of functions to apply (mean, std, min, max, etc.)
            drop_na: Whether to drop rows with NaN values
            
        Returns:
            DataFrame with added rolling window features
        """
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Sort by date if date column exists
        if self.date_col in result.columns:
            result = result.sort_values(self.date_col)
            
        # Create rolling features for each variable, window size, and function
        rolling_features = []
        
        for var in variables:
            if var not in result.columns:
                logger.warning(f"Variable '{var}' not found in DataFrame")
                continue
                
            # Create rolling features for each window size and function
            for window in windows:
                for func in functions:
                    roll_name = f"{var}_roll{window}_{func}"
                    
                    try:
                        # Get rolling window
                        rolling = result[var].rolling(window=window, min_periods=1)
                        
                        # Apply function
                        if func == 'mean':
                            result[roll_name] = rolling.mean()
                        elif func == 'std':
                            result[roll_name] = rolling.std()
                        elif func == 'min':
                            result[roll_name] = rolling.min()
                        elif func == 'max':
                            result[roll_name] = rolling.max()
                        elif func == 'median':
                            result[roll_name] = rolling.median()
                        elif func == 'sum':
                            result[roll_name] = rolling.sum()
                        else:
                            logger.warning(f"Unknown rolling function: {func}")
                            continue
                            
                        rolling_features.append(roll_name)
                        logger.debug(f"Created rolling feature: {roll_name}")
                        
                    except Exception as e:
                        logger.error(f"Error creating rolling feature {roll_name}: {e}")
        
        # Track created features
        self.created_features['rolling'] = rolling_features
        
        # Drop rows with NaN values if requested
        if drop_na:
            orig_len = len(result)
            result = result.dropna()
            dropped = orig_len - len(result)
            logger.info(f"Dropped {dropped} rows with NaN values")
            
        logger.info(f"Created {len(rolling_features)} rolling window features")
        return result
        
    def add_seasonal_dummies(self, df: pd.DataFrame,
                            dummy_type: str = 'month',
                            drop_first: bool = True) -> pd.DataFrame:
        """
        Add seasonal dummy variables (monthly, quarterly) to DataFrame.
        
        Args:
            df: DataFrame with time series data
            dummy_type: Type of dummy variables ('month', 'quarter', 'day_of_week')
            drop_first: Whether to drop the first dummy variable
            
        Returns:
            DataFrame with added seasonal dummy variables
        """
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Ensure date column is datetime
        if self.date_col in result.columns:
            result[self.date_col] = pd.to_datetime(result[self.date_col])
        else:
            logger.warning(f"Date column '{self.date_col}' not found")
            return result
            
        fourier_features = []
        
        # Create a time index starting from 0
        time_idx = np.arange(len(result))
        
        # Calculate Fourier terms for each harmonic
        for i in range(1, harmonics + 1):
            # Sine term
            sin_name = f'sin_{period}_{i}'
            result[sin_name] = np.sin(2 * np.pi * i * time_idx / period)
            fourier_features.append(sin_name)
            
            # Cosine term
            cos_name = f'cos_{period}_{i}'
            result[cos_name] = np.cos(2 * np.pi * i * time_idx / period)
            fourier_features.append(cos_name)
            
        # Track created features
        self.created_features['fourier'] = fourier_features
        
        logger.info(f"Created {len(fourier_features)} Fourier terms")
        return result
    
    def add_interaction_terms(self, df: pd.DataFrame, var1: List[str], 
                             var2: List[str]) -> pd.DataFrame:
        """
        Add interaction terms between two sets of variables.
        
        Args:
            df: DataFrame with time series data
            var1: First set of variables
            var2: Second set of variables
            
        Returns:
            DataFrame with added interaction terms
        """
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        interaction_features = []
        
        # Create interaction terms for each pair of variables
        for v1 in var1:
            if v1 not in result.columns:
                logger.warning(f"Variable '{v1}' not found in DataFrame")
                continue
                
            for v2 in var2:
                if v2 not in result.columns:
                    logger.warning(f"Variable '{v2}' not found in DataFrame")
                    continue
                    
                # Create interaction name
                interaction_name = f"{v1}_x_{v2}"
                
                # Calculate interaction
                result[interaction_name] = result[v1] * result[v2]
                interaction_features.append(interaction_name)
                
                logger.debug(f"Created interaction feature: {interaction_name}")
        
        # Track created features
        self.created_features['interactions'] = interaction_features
        
        logger.info(f"Created {len(interaction_features)} interaction terms")
        return result
    
    def add_event_dummies(self, df: pd.DataFrame, 
                         events: Dict[str, Union[str, List[str]]],
                         window: int = 0) -> pd.DataFrame:
        """
        Add dummy variables for economic events or disruptions.
        
        Args:
            df: DataFrame with time series data
            events: Dictionary mapping event names to dates or date ranges
            window: Number of periods before and after to include
            
        Returns:
            DataFrame with added event dummy variables
        """
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Ensure date column is datetime
        if self.date_col in result.columns:
            result[self.date_col] = pd.to_datetime(result[self.date_col])
        else:
            logger.warning(f"Date column '{self.date_col}' not found")
            return result
        
        event_features = []
        
        # Create dummy variables for each event
        for event_name, event_dates in events.items():
            dummy_name = f"event_{event_name}"
            
            # Initialize dummy variable with zeros
            result[dummy_name] = 0
            
            # Handle single date or list of dates
            if isinstance(event_dates, str):
                event_dates = [event_dates]
                
            # Set dummy variable for each date (and window if specified)
            for date_str in event_dates:
                try:
                    event_date = pd.to_datetime(date_str)
                    
                    if window > 0:
                        # Find dates within window of event
                        for i in range(-window, window + 1):
                            window_date = event_date + pd.Timedelta(days=i)
                            mask = (result[self.date_col].dt.date == window_date.date())
                            result.loc[mask, dummy_name] = 1
                    else:
                        # Exact date match
                        mask = (result[self.date_col].dt.date == event_date.date())
                        result.loc[mask, dummy_name] = 1
                        
                except Exception as e:
                    logger.error(f"Error processing event date {date_str}: {e}")
            
            event_features.append(dummy_name)
            logger.debug(f"Created event dummy: {dummy_name}")
        
        # Track created features
        self.created_features['events'] = event_features
        
        logger.info(f"Created {len(event_features)} event dummy variables")
        return result
    
    def create_economic_indicators(self, df: pd.DataFrame, 
                                  variables: List[str]) -> pd.DataFrame:
        """
        Create standard economic indicators from variables.
        
        Args:
            df: DataFrame with time series data
            variables: Base variables to use
            
        Returns:
            DataFrame with added economic indicators
        """
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Sort by date if date column exists
        if self.date_col in result.columns:
            result = result.sort_values(self.date_col)
        
        # First create some basic transformations
        # 1. Year-over-year percentage changes
        result = self.create_pct_change_features(result, variables, [12])
        
        # 2. Month-over-month percentage changes
        result = self.create_pct_change_features(result, variables, [1])
        
        # 3. Rolling 3-month and 6-month averages
        result = self.create_rolling_features(result, variables, [3, 6], ['mean'])
        
        # 4. Volatility (rolling standard deviation)
        result = self.create_rolling_features(result, variables, [6], ['std'])
        
        # Track specific economic indicators
        econ_indicators = []
        
        # Create more specialized economic indicators
        for var in variables:
            if var not in result.columns:
                continue
                
            # Calculate growth rates
            growth_name = f"{var}_growth_rate"
            if f"{var}_pct1" in result.columns:
                result[growth_name] = result[f"{var}_pct1"]
                econ_indicators.append(growth_name)
                
            # Calculate momentum (difference between short and long-term trends)
            momentum_name = f"{var}_momentum"
            if f"{var}_roll3_mean" in result.columns and f"{var}_roll6_mean" in result.columns:
                result[momentum_name] = result[f"{var}_roll3_mean"] - result[f"{var}_roll6_mean"]
                econ_indicators.append(momentum_name)
            
            # Calculate trend strength
            trend_name = f"{var}_trend_strength"
            if f"{var}_roll6_mean" in result.columns and f"{var}_roll6_std" in result.columns:
                # Avoid division by zero
                with np.errstate(divide='ignore', invalid='ignore'):
                    result[trend_name] = result[f"{var}_roll6_mean"] / result[f"{var}_roll6_std"]
                    result[trend_name] = result[trend_name].replace([np.inf, -np.inf], np.nan)
                econ_indicators.append(trend_name)
        
        # Track created features
        self.created_features['economic_indicators'] = econ_indicators
        
        logger.info(f"Created {len(econ_indicators)} specialized economic indicators")
        return result
    
    def remove_multicollinearity(self, df: pd.DataFrame, 
                                threshold: float = 10.0,
                                exclude_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Remove features with high multicollinearity using Variance Inflation Factor.
        
        Args:
            df: DataFrame with features
            threshold: VIF threshold above which features are removed
            exclude_cols: Columns to exclude from VIF calculation
            
        Returns:
            DataFrame with multicollinear features removed
        """
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Exclude specific columns
        if exclude_cols is None:
            exclude_cols = []
            
        # Always exclude date column
        if self.date_col in result.columns:
            exclude_cols.append(self.date_col)
            
        # Get numeric columns for VIF calculation
        numeric_cols = result.select_dtypes(include=['number']).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if len(numeric_cols) < 2:
            logger.warning("Not enough numeric columns for VIF calculation")
            return result
            
        # Create X matrix for VIF calculation
        X = result[numeric_cols].copy()
        
        # Fill NaN values (required for VIF calculation)
        X = X.fillna(X.mean())
        
        # Iteratively remove features with high VIF
        dropped_cols = []
        max_iter = 100  # Safety limit
        
        for _ in range(max_iter):
            if len(X.columns) < 2:
                break
                
            # Calculate VIF for each feature
            vif_data = self._calculate_vif(X)
            
            # Find max VIF
            max_vif_row = vif_data.loc[vif_data['VIF'].idxmax()]
            max_vif = max_vif_row['VIF']
            max_vif_feature = max_vif_row['Feature']
            
            # Check if max VIF exceeds threshold
            if max_vif < threshold:
                break
                
            # Remove feature with highest VIF
            logger.info(f"Removing {max_vif_feature} due to high VIF: {max_vif:.2f}")
            X = X.drop(columns=[max_vif_feature])
            dropped_cols.append(max_vif_feature)
        
        # Update tracking
        self.dropped_features.extend(dropped_cols)
        
        # Remove dropped columns from result
        result = result.drop(columns=dropped_cols)
        
        logger.info(f"Removed {len(dropped_cols)} features due to multicollinearity")
        return result
    
    def _calculate_vif(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Variance Inflation Factor for features.
        
        Args:
            X: Feature matrix
            
        Returns:
            DataFrame with VIF values for each feature
        """
        # Initialize VIF DataFrame
        vif_df = pd.DataFrame()
        vif_df['Feature'] = X.columns
        
        # Calculate VIF for each feature
        vif_values = []
        
        for i in range(X.shape[1]):
            try:
                vif = variance_inflation_factor(X.values, i)
                
                # Handle infinity and very large values
                if np.isinf(vif) or vif > 1e6:
                    vif = 1e6
                
                vif_values.append(vif)
                
            except Exception as e:
                logger.warning(f"Error calculating VIF for {X.columns[i]}: {e}")
                vif_values.append(float('nan'))
        
        vif_df['VIF'] = vif_values
        
        # Sort by descending VIF
        vif_df = vif_df.sort_values('VIF', ascending=False)
        
        return vif_df
    
    def select_features(self, df: pd.DataFrame, 
                        target: str,
                        max_features: Optional[int] = None,
                        method: str = 'correlation') -> Tuple[pd.DataFrame, List[str]]:
        """
        Select most relevant features for a target variable.
        
        Args:
            df: DataFrame with features
            target: Target variable name
            max_features: Maximum number of features to select
            method: Feature selection method ('correlation', 'mutual_info')
            
        Returns:
            Tuple of (DataFrame with selected features, list of selected feature names)
        """
        # Check if target exists
        if target not in df.columns:
            logger.error(f"Target variable '{target}' not found in DataFrame")
            return df.copy(), []
            
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Get numeric columns (excluding target)
        numeric_cols = result.select_dtypes(include=['number']).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != target]
        
        if not numeric_cols:
            logger.warning("No numeric features available for selection")
            return result, []
            
        selected_features = []
        
        if method == 'correlation':
            # Calculate absolute correlation with target
            corr_with_target = result[numeric_cols].corrwith(result[target]).abs()
            
            # Sort features by correlation
            sorted_features = corr_with_target.sort_values(ascending=False)
            
            # Select top features
            if max_features is not None:
                selected_features = sorted_features.index[:max_features].tolist()
            else:
                selected_features = sorted_features.index.tolist()
                
            logger.info(f"Selected {len(selected_features)} features using correlation method")
            
        elif method == 'mutual_info':
            try:
                from sklearn.feature_selection import mutual_info_regression
                
                # Get feature matrix and target
                X = result[numeric_cols].fillna(0)
                y = result[target].fillna(0)
                
                # Calculate mutual information
                mi_scores = mutual_info_regression(X, y)
                
                # Create DataFrame with scores
                mi_df = pd.DataFrame({'Feature': numeric_cols, 'MI_Score': mi_scores})
                sorted_mi = mi_df.sort_values('MI_Score', ascending=False)
                
                # Select top features
                if max_features is not None:
                    selected_features = sorted_mi['Feature'].iloc[:max_features].tolist()
                else:
                    selected_features = sorted_mi['Feature'].tolist()
                    
                logger.info(f"Selected {len(selected_features)} features using mutual information")
                
            except Exception as e:
                logger.error(f"Error calculating mutual information: {e}")
                # Fall back to correlation method
                logger.info("Falling back to correlation method")
                return self.select_features(df, target, max_features, 'correlation')
                
        else:
            logger.warning(f"Unknown feature selection method: {method}")
            return result, numeric_cols
        
        # Include target and date column
        keep_cols = selected_features.copy()
        
        if self.date_col in result.columns:
            keep_cols.append(self.date_col)
            
        keep_cols.append(target)
        
        # Return DataFrame with selected features only
        result = result[keep_cols]
        
        return result, selected_features
    
    def get_feature_summary(self) -> Dict:
        """
        Get summary of created and dropped features.
        
        Returns:
            Dictionary with feature summary
        """
        return {
            'created': self.created_features,
            'dropped': self.dropped_features
        }


# Helper functions for common feature creation patterns

def create_standard_features(df: pd.DataFrame, 
                            variables: List[str],
                            date_col: str = 'date') -> pd.DataFrame:
    """
    Create a standard set of time series features.
    
    Args:
        df: DataFrame with time series data
        variables: Variables to create features for
        date_col: Name of date column
        
    Returns:
        DataFrame with added features
    """
    factory = FeatureFactory(date_col)
    
    # Add lags 1, 2, 3
    result = factory.create_lags(df, variables, max_lag=3)
    
    # Add percentage changes
    result = factory.create_pct_change_features(result, variables, periods=[1, 12])
    
    # Add rolling statistics
    result = factory.create_rolling_features(result, variables, windows=[3, 6, 12], functions=['mean'])
    
    # Add seasonal components
    result = factory.add_seasonal_dummies(result, dummy_type='month')
    
    return result

def create_economic_feature_set(df: pd.DataFrame,
                               variables: List[str],
                               date_col: str = 'date',
                               remove_collinear: bool = True) -> pd.DataFrame:
    """
    Create a comprehensive set of economic features with optional multicollinearity removal.
    
    Args:
        df: DataFrame with time series data
        variables: Variables to create features for
        date_col: Name of date column
        remove_collinear: Whether to remove multicollinear features
        
    Returns:
        DataFrame with added features
    """
    factory = FeatureFactory(date_col)
    
    # Create standard feature set
    result = create_standard_features(df, variables, date_col)
    
    # Add economic indicators
    result = factory.create_economic_indicators(result, variables)
    
    # Add Fourier terms for seasonal patterns
    result = factory.add_fourier_terms(result, period=12, harmonics=2)
    
    # Remove multicollinearity if requested
    if remove_collinear:
        result = factory.remove_multicollinearity(result, threshold=10.0)
    
    return result column '{self.date_col}' not found")
            return result
        
        dummy_features = []
        
        # Create dummy variables based on type
        if dummy_type == 'month':
            # Extract month (1-12)
            months = result[self.date_col].dt.month
            
            # Create dummies
            month_dummies = pd.get_dummies(months, prefix='month', drop_first=drop_first)
            
            # Add to result
            for col in month_dummies.columns:
                result[col] = month_dummies[col]
                dummy_features.append(col)
                
            logger.info(f"Created {len(dummy_features)} month dummy variables")
            
        elif dummy_type == 'quarter':
            # Extract quarter (1-4)
            quarters = result[self.date_col].dt.quarter
            
            # Create dummies
            quarter_dummies = pd.get_dummies(quarters, prefix='quarter', drop_first=drop_first)
            
            # Add to result
            for col in quarter_dummies.columns:
                result[col] = quarter_dummies[col]
                dummy_features.append(col)
                
            logger.info(f"Created {len(dummy_features)} quarter dummy variables")
            
        elif dummy_type == 'day_of_week':
            # Extract day of week (0-6)
            day_of_week = result[self.date_col].dt.dayofweek
            
            # Create dummies
            dow_dummies = pd.get_dummies(day_of_week, prefix='dow', drop_first=drop_first)
            
            # Add to result
            for col in dow_dummies.columns:
                result[col] = dow_dummies[col]
                dummy_features.append(col)
                
            logger.info(f"Created {len(dummy_features)} day of week dummy variables")
            
        else:
            logger.warning(f"Unknown dummy type: {dummy_type}")
            
        # Track created features
        self.created_features[f'{dummy_type}_dummies'] = dummy_features
        
        return result
        
    def add_fourier_terms(self, df: pd.DataFrame, period: int = 12, 
                         harmonics: int = 2) -> pd.DataFrame:
        """
        Add Fourier terms for seasonal patterns.
        
        Args:
            df: DataFrame with time series data
            period: Length of seasonal cycle (e.g., 12 for monthly data with annual cycle)
            harmonics: Number of harmonics to include
            
        Returns:
            DataFrame with added Fourier terms
        """
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Ensure date column is datetime
        if self.date_col in result.columns:
            result[self.date_col] = pd.to_datetime(result[self.date_col])
        else:
            logger.warning(f"Date