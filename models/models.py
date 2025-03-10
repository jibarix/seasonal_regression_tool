"""
Abstract base model class for time series forecasting models.
Provides a standard interface and common functionality.
"""
import os
import pickle
import json
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Union, Optional, Tuple, Any
from abc import ABC, abstractmethod
from datetime import datetime

# Setup logging
logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for time series forecasting models.
    All model implementations should inherit from this class.
    """
    
    def __init__(self, name: str = None, date_col: str = 'date'):
        """
        Initialize the base model.
        
        Args:
            name: Model name
            date_col: Name of the date column
        """
        # Set model name or generate a default
        if name is None:
            self.name = f"{self.__class__.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            self.name = name
            
        self.date_col = date_col
        self.feature_names = []
        self.is_fitted = False
        self.model_params = {}
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'updated_at': None,
            'transformations': [],
            'training_data_info': {},
            'performance_metrics': {}
        }
        self.model = None
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'BaseModel':
        """
        Fit the model to the training data.
        
        Args:
            X: Feature matrix
            y: Target variable
            **kwargs: Additional model-specific parameters
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using the fitted model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions
        """
        pass
    
    def preprocess(self, X: pd.DataFrame, y: Optional[pd.Series] = None, 
                  fit: bool = False) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Preprocess data before model fitting or prediction.
        
        Args:
            X: Feature matrix
            y: Target variable (optional)
            fit: Whether this is for fitting (True) or prediction (False)
            
        Returns:
            Tuple of (processed_X, processed_y)
        """
        processed_X = X.copy()
        processed_y = y.copy() if y is not None else None
        
        # Store feature names if fitting
        if fit:
            self.feature_names = list(processed_X.columns)
            
            # Log data info
            self.metadata['training_data_info'] = {
                'n_samples': len(processed_X),
                'n_features': len(self.feature_names),
                'feature_names': self.feature_names,
                'date_range': [
                    processed_X[self.date_col].min().isoformat() if self.date_col in processed_X else None,
                    processed_X[self.date_col].max().isoformat() if self.date_col in processed_X else None
                ]
            }
        else:
            # Check for missing features in prediction data
            missing_features = [f for f in self.feature_names if f not in processed_X.columns]
            if missing_features:
                logger.warning(f"Missing features in prediction data: {missing_features}")
                
                # Add missing features with zeros
                for feature in missing_features:
                    processed_X[feature] = 0
        
        # Handle missing values
        processed_X = self._handle_missing_values(processed_X)
        
        if processed_y is not None:
            # Handle missing values in target
            processed_y = processed_y.fillna(processed_y.mean())
        
        return processed_X, processed_y
    
    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the feature matrix.
        
        Args:
            X: Feature matrix
            
        Returns:
            Feature matrix with missing values handled
        """
        # Create a copy to avoid modifying the original
        result = X.copy()
        
        # Get numeric columns
        numeric_cols = result.select_dtypes(include=['number']).columns.tolist()
        
        # Remove date column if present
        if self.date_col in numeric_cols:
            numeric_cols.remove(self.date_col)
        
        # Impute missing values with column means
        for col in numeric_cols:
            missing_count = result[col].isna().sum()
            if missing_count > 0:
                mean_val = result[col].mean()
                result[col] = result[col].fillna(mean_val)
                logger.debug(f"Imputed {missing_count} missing values in '{col}'")
        
        return result
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            X: Feature matrix
            y: True target values
            
        Returns:
            Dictionary of performance metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        # Make predictions
        predictions = self.predict(X)
        
        # Calculate various metrics
        metrics = self._calculate_metrics(y, predictions)
        
        # Store metrics in metadata
        self.metadata['performance_metrics'] = metrics
        self.metadata['updated_at'] = datetime.now().isoformat()
        
        return metrics
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate performance metrics for predictions.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        # Ensure arrays have the same shape
        if len(y_true) != len(y_pred):
            raise ValueError(f"Length mismatch: y_true ({len(y_true)}) and y_pred ({len(y_pred)})")
        
        # Calculate error terms
        errors = y_true - y_pred
        abs_errors = np.abs(errors)
        squared_errors = errors ** 2
        
        # Basic metrics
        metrics = {
            'mae': np.mean(abs_errors),  # Mean Absolute Error
            'rmse': np.sqrt(np.mean(squared_errors)),  # Root Mean Squared Error
            'mse': np.mean(squared_errors),  # Mean Squared Error
            'median_ae': np.median(abs_errors),  # Median Absolute Error
            'max_error': np.max(abs_errors),  # Maximum Error
        }
        
        # Calculate MAPE if no zero values in y_true
        if not np.any(y_true == 0):
            mape = np.mean(np.abs(errors / y_true)) * 100
            metrics['mape'] = mape  # Mean Absolute Percentage Error
        else:
            # Alternative for data with zeros: sMAPE (Symmetric MAPE)
            with np.errstate(divide='ignore', invalid='ignore'):
                smape = np.mean(2 * abs_errors / (np.abs(y_true) + np.abs(y_pred))) * 100
                metrics['smape'] = np.nan_to_num(smape)  # Replace NaN with 0
        
        # R-squared (coefficient of determination)
        y_mean = np.mean(y_true)
        tss = np.sum((y_true - y_mean) ** 2)  # Total sum of squares
        rss = np.sum(squared_errors)  # Residual sum of squares
        
        if tss > 0:
            metrics['r_squared'] = 1 - (rss / tss)
        else:
            metrics['r_squared'] = 0
        
        # Directional accuracy (percentage of correct direction predictions)
        if len(y_true) > 1:
            actual_diff = np.diff(y_true)
            pred_diff = np.diff(y_pred)
            
            direction_matches = (actual_diff * pred_diff) > 0
            metrics['directional_accuracy'] = np.mean(direction_matches) * 100
        
        return metrics
    
    def save(self, directory: str = 'models') -> str:
        """
        Save the model to disk.
        
        Args:
            directory: Directory to save the model
            
        Returns:
            Path to the saved model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save an unfitted model")
        
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Base filename
        filename = f"{self.name}"
        model_path = os.path.join(directory, f"{filename}.pkl")
        metadata_path = os.path.join(directory, f"{filename}_metadata.json")
        
        # Update metadata
        self.metadata['updated_at'] = datetime.now().isoformat()
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
        return model_path
    
    @classmethod
    def load(cls, model_path: str) -> 'BaseModel':
        """
        Load a model from disk.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Loaded model instance
        """
        # Extract directory and filename
        directory = os.path.dirname(model_path)
        filename = os.path.basename(model_path).replace('.pkl', '')
        metadata_path = os.path.join(directory, f"{filename}_metadata.json")
        
        # Create instance
        instance = cls()
        
        # Load model
        with open(model_path, 'rb') as f:
            instance.model = pickle.load(f)
        
        # Load metadata if exists
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                instance.metadata = json.load(f)
        
        # Set feature names and other attributes from metadata
        if 'training_data_info' in instance.metadata:
            if 'feature_names' in instance.metadata['training_data_info']:
                instance.feature_names = instance.metadata['training_data_info']['feature_names']
        
        # Set fitted flag
        instance.is_fitted = True
        instance.name = filename
        
        logger.info(f"Model loaded from {model_path}")
        return instance
    
    def get_params(self) -> Dict:
        """
        Get model parameters.
        
        Returns:
            Dictionary of model parameters
        """
        return self.model_params
    
    def set_params(self, **params) -> 'BaseModel':
        """
        Set model parameters.
        
        Args:
            **params: Parameters to set
            
        Returns:
            Self for method chaining
        """
        self.model_params.update(params)
        return self
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance if the model supports it.
        
        Returns:
            DataFrame with feature importance or None
        """
        # Default implementation returns None
        # Should be overridden by subclasses that support feature importance
        return None
    
    def __repr__(self) -> str:
        """String representation of the model."""
        status = "fitted" if self.is_fitted else "not fitted"
        return f"{self.__class__.__name__}(name='{self.name}', {status})"
    
    def summary(self) -> str:
        """Get a summary of the model."""
        lines = [f"Model: {self.__class__.__name__}", f"Name: {self.name}"]
        
        if self.is_fitted:
            lines.append("Status: Fitted")
            
            # Add feature info
            if self.feature_names:
                lines.append(f"Number of features: {len(self.feature_names)}")
            
            # Add performance metrics if available
            if 'performance_metrics' in self.metadata and self.metadata['performance_metrics']:
                lines.append("\nPerformance Metrics:")
                for metric, value in self.metadata['performance_metrics'].items():
                    lines.append(f"  {metric}: {value:.4f}")
        else:
            lines.append("Status: Not fitted")
        
        return "\n".join(lines)