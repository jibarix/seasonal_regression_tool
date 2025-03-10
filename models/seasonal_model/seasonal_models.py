"""
Concrete implementations of seasonal time series models.
Provides different approaches to modeling seasonality.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Union, Optional, Tuple, Any
import statsmodels.api as sm
from datetime import datetime

# Import from project modules
from seasonal_base import BaseSeasonalModel

# Setup logging
logger = logging.getLogger(__name__)


class DummyVariableSeasonalModel(BaseSeasonalModel):
    """
    Seasonal model using dummy variables (one-hot encoding) for seasons.
    Supports monthly and quarterly seasonality.
    """
    
    def __init__(self, name: str = None, date_col: str = 'date',
                 seasonality_type: str = 'monthly', drop_first: bool = True,
                 **kwargs):
        """
        Initialize the dummy variable seasonal model.
        
        Args:
            name: Model name
            date_col: Name of the date column
            seasonality_type: Type of seasonality ('monthly', 'quarterly')
            drop_first: Whether to drop the first dummy to avoid collinearity
            **kwargs: Additional parameters passed to BaseSeasonalModel
        """
        super().__init__(
            name=name or f"DummyVar_{seasonality_type}",
            date_col=date_col,
            seasonality_type=seasonality_type,
            **kwargs
        )
        
        self.drop_first = drop_first
        self.model_params.update({'drop_first': drop_first})
        
    def add_seasonal_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add seasonal dummy variables to the feature matrix.
        
        Args:
            X: Feature matrix
            
        Returns:
            Feature matrix with added seasonal dummy variables
        """
        result = X.copy()
        
        # Ensure date column is datetime
        if self.date_col in result.columns:
            result[self.date_col] = pd.to_datetime(result[self.date_col])
            
            # Extract month/quarter based on seasonality type
            if self.seasonality_type == 'monthly':
                result['month'] = result[self.date_col].dt.month
                
                # Create dummy variables, optionally dropping the first
                start_idx = 2 if self.drop_first else 1
                end_idx = 13  # 12 months + 1
                
                for month in range(start_idx, end_idx):
                    dummy_name = f'month_{month}'
                    result[dummy_name] = (result['month'] == month).astype(int)
                    
                    # Track as seasonal feature
                    if dummy_name not in self.feature_groups['seasonal_features']:
                        self.feature_groups['seasonal_features'].append(dummy_name)
                
            elif self.seasonality_type == 'quarterly':
                result['quarter'] = result[self.date_col].dt.quarter
                
                # Create dummy variables, optionally dropping the first
                start_idx = 2 if self.drop_first else 1
                end_idx = 5  # 4 quarters + 1
                
                for quarter in range(start_idx, end_idx):
                    dummy_name = f'quarter_{quarter}'
                    result[dummy_name] = (result['quarter'] == quarter).astype(int)
                    
                    # Track as seasonal feature
                    if dummy_name not in self.feature_groups['seasonal_features']:
                        self.feature_groups['seasonal_features'].append(dummy_name)
        
        return result
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'DummyVariableSeasonalModel':
        """
        Fit the dummy variable seasonal model.
        
        Args:
            X: Feature matrix
            y: Target variable
            **kwargs: Additional model-specific parameters
            
        Returns:
            Self for method chaining
        """
        # Preprocess data
        processed_X, processed_y = self.preprocess(X, y, fit=True)
        
        # Add constant term for intercept
        processed_X_with_const = sm.add_constant(processed_X)
        
        # Fit OLS model
        self.model = sm.OLS(processed_y, processed_X_with_const).fit()
        
        # Extract and store seasonal components
        self._extract_seasonal_coefficients()
        
        self.is_fitted = True
        self.metadata['updated_at'] = datetime.now().isoformat()
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using the fitted model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Preprocess data
        processed_X, _ = self.preprocess(X, fit=False)
        
        # Add constant term for intercept
        processed_X_with_const = sm.add_constant(processed_X)
        
        # Generate predictions
        predictions = self.model.predict(processed_X_with_const)
        
        return predictions
    
    def _extract_seasonal_coefficients(self) -> None:
        """
        Extract seasonal coefficients from the fitted model.
        Updates self.seasonal_components with the coefficients.
        """
        if not self.is_fitted or self.model is None:
            return
        
        # Initialize seasonal components dictionary
        seasonal_components = {}
        
        # Get all coefficients
        coefficients = self.model.params
        
        # Extract seasonal coefficients based on seasonality type
        if self.seasonality_type == 'monthly':
            # Extract monthly coefficients
            monthly_coef = []
            
            # Get base effect (intercept or first month)
            if self.drop_first:
                # First month is the reference (intercept)
                base_effect = coefficients.get('const', 0)
                monthly_coef.append(base_effect)
                
                # Get effects for months 2-12
                for month in range(2, 13):
                    dummy_name = f'month_{month}'
                    if dummy_name in coefficients:
                        # Add to base effect
                        monthly_coef.append(base_effect + coefficients[dummy_name])
                    else:
                        monthly_coef.append(base_effect)
            else:
                # Each month has its own dummy
                for month in range(1, 13):
                    dummy_name = f'month_{month}'
                    if dummy_name in coefficients:
                        monthly_coef.append(coefficients[dummy_name])
                    else:
                        monthly_coef.append(0)
            
            seasonal_components['seasonal'] = np.array(monthly_coef)
            
        elif self.seasonality_type == 'quarterly':
            # Extract quarterly coefficients
            quarterly_coef = []
            
            # Get base effect (intercept or first quarter)
            if self.drop_first:
                # First quarter is the reference (intercept)
                base_effect = coefficients.get('const', 0)
                quarterly_coef.append(base_effect)
                
                # Get effects for quarters 2-4
                for quarter in range(2, 5):
                    dummy_name = f'quarter_{quarter}'
                    if dummy_name in coefficients:
                        # Add to base effect
                        quarterly_coef.append(base_effect + coefficients[dummy_name])
                    else:
                        quarterly_coef.append(base_effect)
            else:
                # Each quarter has its own dummy
                for quarter in range(1, 5):
                    dummy_name = f'quarter_{quarter}'
                    if dummy_name in coefficients:
                        quarterly_coef.append(coefficients[dummy_name])
                    else:
                        quarterly_coef.append(0)
            
            seasonal_components['seasonal'] = np.array(quarterly_coef)
        
        # Store components
        self.seasonal_components = seasonal_components
    
    def extract_seasonal_components(self, X: pd.DataFrame = None) -> Dict[str, np.ndarray]:
        """
        Extract seasonal components from the fitted model.
        
        Args:
            X: Optional feature matrix for prediction (not used in this model)
            
        Returns:
            Dictionary mapping component names to arrays of values
        """
        if not self.is_fitted or self.model is None:
            return {}
        
        # If components haven't been extracted yet, do it now
        if self.seasonal_components is None:
            self._extract_seasonal_coefficients()
        
        return self.seasonal_components


class FourierSeasonalModel(BaseSeasonalModel):
    """
    Seasonal model using Fourier series (sine and cosine terms) for smooth seasonality.
    """
    
    def __init__(self, name: str = None, date_col: str = 'date',
                 seasonality_type: str = 'monthly', harmonics: int = 2,
                 **kwargs):
        """
        Initialize the Fourier seasonal model.
        
        Args:
            name: Model name
            date_col: Name of the date column
            seasonality_type: Type of seasonality ('monthly', 'quarterly')
            harmonics: Number of harmonics to include
            **kwargs: Additional parameters passed to BaseSeasonalModel
        """
        super().__init__(
            name=name or f"Fourier_{seasonality_type}_{harmonics}harm",
            date_col=date_col, 
            seasonality_type=seasonality_type,
            **kwargs
        )
        
        self.harmonics = harmonics
        self.model_params.update({'harmonics': harmonics})
    
    def add_seasonal_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add Fourier terms for seasonality to the feature matrix.
        
        Args:
            X: Feature matrix
            
        Returns:
            Feature matrix with added Fourier terms
        """
        result = X.copy()
        
        # Add trend variables - first convert dates to numeric values if date exists
        if self.date_col in result.columns:
            dates = pd.to_datetime(result[self.date_col])
            min_date = dates.min()
            
            # Calculate trend in monthly units
            result['trend'] = (dates - min_date).dt.days / 30.0
            result['trend_squared'] = result['trend'] ** 2
            
            # Track trend features
            for feature in ['trend', 'trend_squared']:
                if feature not in self.feature_groups['trend_features']:
                    self.feature_groups['trend_features'].append(feature)
            
            # Extract month for seasonal components
            month_num = dates.dt.month
            
            # Create Fourier terms
            period = self.seasonal_period  # From base class (12 for monthly, 4 for quarterly)
            
            for harm in range(1, self.harmonics + 1):
                # Create sine and cosine components
                sin_name = f'sin_h{harm}'
                cos_name = f'cos_h{harm}'
                
                result[sin_name] = np.sin(2 * np.pi * harm * month_num / period)
                result[cos_name] = np.cos(2 * np.pi * harm * month_num / period)
                
                # Track seasonal features
                if sin_name not in self.feature_groups['seasonal_features']:
                    self.feature_groups['seasonal_features'].append(sin_name)
                    self.feature_groups['seasonal_features'].append(cos_name)
        
        return result
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'FourierSeasonalModel':
        """
        Fit the Fourier seasonal model.
        
        Args:
            X: Feature matrix
            y: Target variable
            **kwargs: Additional model-specific parameters
            
        Returns:
            Self for method chaining
        """
        # Preprocess data
        processed_X, processed_y = self.preprocess(X, y, fit=True)
        
        # Add constant term for intercept
        processed_X_with_const = sm.add_constant(processed_X)
        
        # Fit OLS model
        self.model = sm.OLS(processed_y, processed_X_with_const).fit()
        
        # Set flag
        self.is_fitted = True
        self.metadata['updated_at'] = datetime.now().isoformat()
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using the fitted model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Preprocess data
        processed_X, _ = self.preprocess(X, fit=False)
        
        # Add constant term for intercept
        processed_X_with_const = sm.add_constant(processed_X)
        
        # Generate predictions
        predictions = self.model.predict(processed_X_with_const)
        
        return predictions
    
    def extract_seasonal_components(self, X: pd.DataFrame = None) -> Dict[str, np.ndarray]:
        """
        Extract seasonal components from the fitted model.
        For Fourier models, reconstruct the seasonal pattern from sine/cosine coefficients.
        
        Args:
            X: Optional feature matrix for prediction context
            
        Returns:
            Dictionary mapping component names to arrays of values
        """
        if not self.is_fitted or self.model is None:
            return {}
        
        # Get coefficients
        coefficients = self.model.params
        
        # For monthly data, reconstruct the seasonal pattern for each month
        if self.seasonality_type == 'monthly':
            months = np.arange(1, 13)
            period = 12
        elif self.seasonality_type == 'quarterly':
            months = np.arange(1, 5)
            period = 4
        else:
            return {}
        
        # Initialize seasonal component
        seasonal = np.zeros(len(months))
        
        # Add contribution from each harmonic
        for harm in range(1, self.harmonics + 1):
            sin_name = f'sin_h{harm}'
            cos_name = f'cos_h{harm}'
            
            if sin_name in coefficients and cos_name in coefficients:
                sin_coef = coefficients[sin_name]
                cos_coef = coefficients[cos_name]
                
                # Calculate harmonic contribution for each month
                for i, month in enumerate(months):
                    seasonal[i] += sin_coef * np.sin(2 * np.pi * harm * month / period)
                    seasonal[i] += cos_coef * np.cos(2 * np.pi * harm * month / period)
        
        return {'seasonal': seasonal}


class HybridSeasonalModel(BaseSeasonalModel):
    """
    Hybrid seasonal model that combines dummy variables and trend components.
    Good for capturing both discrete seasonal effects and smooth trends.
    """
    
    def __init__(self, name: str = None, date_col: str = 'date',
                 seasonality_type: str = 'monthly', 
                 include_trend: bool = True,
                 trend_degree: int = 2,
                 **kwargs):
        """
        Initialize the hybrid seasonal model.
        
        Args:
            name: Model name
            date_col: Name of the date column
            seasonality_type: Type of seasonality ('monthly', 'quarterly')
            include_trend: Whether to include trend components
            trend_degree: Polynomial degree for trend (1=linear, 2=quadratic, etc.)
            **kwargs: Additional parameters passed to BaseSeasonalModel
        """
        super().__init__(
            name=name or f"Hybrid_{seasonality_type}",
            date_col=date_col,
            seasonality_type=seasonality_type,
            **kwargs
        )
        
        self.include_trend = include_trend
        self.trend_degree = trend_degree
        self.model_params.update({
            'include_trend': include_trend,
            'trend_degree': trend_degree
        })
    
    def add_seasonal_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add both dummy variables and trend components to the feature matrix.
        
        Args:
            X: Feature matrix
            
        Returns:
            Feature matrix with added seasonal and trend features
        """
        result = X.copy()
        
        # Ensure date column is datetime
        if self.date_col in result.columns:
            result[self.date_col] = pd.to_datetime(result[self.date_col])
            
            # Add seasonal dummies
            if self.seasonality_type == 'monthly':
                result['month'] = result[self.date_col].dt.month
                
                # Create month dummies (drop first to avoid collinearity)
                for month in range(2, 13):
                    dummy_name = f'month_{month}'
                    result[dummy_name] = (result['month'] == month).astype(int)
                    
                    # Track seasonal feature
                    if dummy_name not in self.feature_groups['seasonal_features']:
                        self.feature_groups['seasonal_features'].append(dummy_name)
                
            elif self.seasonality_type == 'quarterly':
                result['quarter'] = result[self.date_col].dt.quarter
                
                # Create quarter dummies (drop first to avoid collinearity)
                for quarter in range(2, 5):
                    dummy_name = f'quarter_{quarter}'
                    result[dummy_name] = (result['quarter'] == quarter).astype(int)
                    
                    # Track seasonal feature
                    if dummy_name not in self.feature_groups['seasonal_features']:
                        self.feature_groups['seasonal_features'].append(dummy_name)
            
            # Add trend components if requested
            if self.include_trend:
                # Calculate months since start
                min_date = result[self.date_col].min()
                result['trend'] = (result[self.date_col] - min_date).dt.days / 30.0
                
                # Add polynomial trend terms
                for degree in range(2, self.trend_degree + 1):
                    result[f'trend_{degree}'] = result['trend'] ** degree
                
                # Track trend features
                trend_features = ['trend'] + [f'trend_{d}' for d in range(2, self.trend_degree + 1)]
                for feature in trend_features:
                    if feature not in self.feature_groups['trend_features']:
                        self.feature_groups['trend_features'].append(feature)
        
        return result
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'HybridSeasonalModel':
        """
        Fit the hybrid seasonal model.
        
        Args:
            X: Feature matrix
            y: Target variable
            **kwargs: Additional model-specific parameters
            
        Returns:
            Self for method chaining
        """
        # Preprocess data
        processed_X, processed_y = self.preprocess(X, y, fit=True)
        
        # Add constant term for intercept
        processed_X_with_const = sm.add_constant(processed_X)
        
        # Fit OLS model
        self.model = sm.OLS(processed_y, processed_X_with_const).fit()
        
        # Extract and store seasonal components
        self._extract_components()
        
        # Set flag
        self.is_fitted = True
        self.metadata['updated_at'] = datetime.now().isoformat()
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using the fitted model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Preprocess data
        processed_X, _ = self.preprocess(X, fit=False)
        
        # Add constant term for intercept
        processed_X_with_const = sm.add_constant(processed_X)
        
        # Generate predictions
        predictions = self.model.predict(processed_X_with_const)
        
        return predictions
    
    def _extract_components(self) -> None:
        """
        Extract seasonal and trend components from the fitted model.
        Updates self.seasonal_components with the extracted components.
        """
        if not self.is_fitted or self.model is None:
            return
        
        # Initialize components dictionary
        components = {}
        
        # Get coefficients
        coefficients = self.model.params
        
        # Extract seasonal components (similar to DummyVariableSeasonalModel)
        if self.seasonality_type == 'monthly':
            # Extract monthly effects
            monthly_coef = []
            
            # First month is the reference (intercept)
            base_effect = coefficients.get('const', 0)
            monthly_coef.append(base_effect)
            
            # Get effects for months 2-12
            for month in range(2, 13):
                dummy_name = f'month_{month}'
                if dummy_name in coefficients:
                    # Add to base effect
                    monthly_coef.append(base_effect + coefficients[dummy_name])
                else:
                    monthly_coef.append(base_effect)
            
            components['seasonal'] = np.array(monthly_coef)
            
        elif self.seasonality_type == 'quarterly':
            # Extract quarterly effects
            quarterly_coef = []
            
            # First quarter is the reference (intercept)
            base_effect = coefficients.get('const', 0)
            quarterly_coef.append(base_effect)
            
            # Get effects for quarters 2-4
            for quarter in range(2, 5):
                dummy_name = f'quarter_{quarter}'
                if dummy_name in coefficients:
                    # Add to base effect
                    quarterly_coef.append(base_effect + coefficients[dummy_name])
                else:
                    quarterly_coef.append(base_effect)
            
            components['seasonal'] = np.array(quarterly_coef)
        
        # Extract trend components if included
        if self.include_trend:
            # Get trend coefficients
            trend_coefs = {}
            
            for feature in self.feature_groups['trend_features']:
                if feature in coefficients:
                    trend_coefs[feature] = coefficients[feature]
            
            components['trend_coefficients'] = trend_coefs
        
        # Store components
        self.seasonal_components = components
    
    def extract_seasonal_components(self, X: pd.DataFrame = None) -> Dict[str, np.ndarray]:
        """
        Extract seasonal components from the fitted model.
        
        Args:
            X: Optional feature matrix for prediction context
            
        Returns:
            Dictionary mapping component names to arrays of values
        """
        if not self.is_fitted or self.model is None:
            return {}
        
        # If components haven't been extracted yet, do it now
        if self.seasonal_components is None:
            self._extract_components()
        
        return self.seasonal_components


def add_month_dummies(df_base: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """
    Add monthly dummy variables (11 dummies) to the base features.
    
    Args:
        df_base: DataFrame with base features
        date_col: Name of date column
        
    Returns:
        DataFrame with month dummies added
    """
    # Create a copy to avoid modifying the input
    df = df_base.copy()
    
    # Extract month and create dummies - ensure numeric month values
    months = pd.to_datetime(df[date_col]).dt.month.astype(int)
    month_dummies = pd.get_dummies(months, prefix='month', drop_first=True)
    
    # Ensure dummy columns have numeric types
    for col in month_dummies.columns:
        month_dummies[col] = month_dummies[col].astype(float)
    
    # Add to dataframe
    result = pd.concat([df, month_dummies], axis=1)
    
    return result


def add_quarter_dummies(df_base: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """
    Add quarterly dummy variables (3 dummies) to the base features.
    
    Args:
        df_base: DataFrame with base features
        date_col: Name of date column
        
    Returns:
        DataFrame with quarter dummies added
    """
    # Create a copy to avoid modifying the input
    df = df_base.copy()
    
    # Extract quarter and create dummies - ensure numeric quarter values
    quarters = pd.to_datetime(df[date_col]).dt.quarter.astype(int)
    quarter_dummies = pd.get_dummies(quarters, prefix='quarter', drop_first=True)
    
    # Ensure dummy columns have numeric types
    for col in quarter_dummies.columns:
        quarter_dummies[col] = quarter_dummies[col].astype(float)
    
    # Add to dataframe
    result = pd.concat([df, quarter_dummies], axis=1)
    
    return result


def add_seasonal_components(df_base: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """
    Add trigonometric seasonal components and trend variables.
    
    Args:
        df_base: DataFrame with base features
        date_col: Name of date column
        
    Returns:
        DataFrame with trigonometric seasonal components added
    """
    # Create a copy to avoid modifying the input
    df = df_base.copy()
    
    # Add trend variables - first convert dates to numeric values
    dates = pd.to_datetime(df[date_col])
    min_date = dates.min()
    df['trend'] = (dates - min_date).dt.days / 30  # Trend in months
    df['trend_squared'] = df['trend'] ** 2
    
    # Extract month for seasonal components
    month_num = dates.dt.month
    
    # Create sine and cosine components for annual seasonality
    df['sin_annual'] = np.sin(2 * np.pi * month_num / 12)
    df['cos_annual'] = np.cos(2 * np.pi * month_num / 12)
    
    # Create sine and cosine components for semi-annual seasonality
    df['sin_semiannual'] = np.sin(4 * np.pi * month_num / 12)
    df['cos_semiannual'] = np.cos(4 * np.pi * month_num / 12)
    
    return df