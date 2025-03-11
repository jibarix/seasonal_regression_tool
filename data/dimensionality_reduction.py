"""
Dimensionality reduction module for time series analysis.
Provides PCA with time series specific considerations.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Union, Optional, Tuple, Any
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# Setup logging
logger = logging.getLogger(__name__)


class PCAnalyzer:
    """
    Class for PCA analysis with proper handling of time series data.
    Provides functionality for dimensionality reduction and feature importance.
    """
    
    def __init__(self, standardize: bool = True, 
                 n_components: Optional[int] = None,
                 variance_threshold: float = 0.95,
                 random_state: int = 42):
        """
        Initialize PCA analyzer with options for standardization and components selection.
        
        Args:
            standardize: Whether to standardize data before PCA
            n_components: Number of components to keep (None for auto-selection)
            variance_threshold: Minimum variance to explain when auto-selecting components
            random_state: Random seed for reproducibility
        """
        self.standardize = standardize
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.random_state = random_state
        
        self.scaler = StandardScaler() if standardize else None
        self.pca = None
        self.is_fitted = False
        
        # For storing feature information
        self.feature_names = []
        self.exclude_cols = []
        self.metadata = {
            'n_samples': 0,
            'n_features': 0,
            'n_components': 0,
            'explained_variance_ratio': [],
            'cumulative_variance_ratio': []
        }
    
    def fit(self, df: pd.DataFrame, columns: Optional[List[str]] = None, 
            exclude_cols: Optional[List[str]] = None) -> 'PCAnalyzer':
        """
        Fit PCA to data, excluding specified columns.
        
        Args:
            df: DataFrame with features
            columns: Columns to use (None for all numeric columns)
            exclude_cols: Columns to exclude (e.g., date, ID)
            
        Returns:
            Self for method chaining
        """
        # Select columns for PCA
        if columns is None:
            # Use all numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            # Exclude specified columns
            if exclude_cols is not None:
                numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
                
            columns = numeric_cols
        else:
            # Ensure all specified columns exist
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                logger.warning(f"Columns not found in DataFrame: {missing_cols}")
                columns = [col for col in columns if col in df.columns]
        
        # Save feature names and excluded columns
        self.feature_names = columns
        self.exclude_cols = exclude_cols or []
        
        # Check if we have enough columns for PCA
        if len(columns) < 2:
            logger.error("Not enough columns for PCA (need at least 2)")
            raise ValueError("Not enough columns for PCA (need at least 2)")
        
        # Extract feature matrix
        X = df[columns].values
        
        # Handle missing values if any
        if np.isnan(X).any():
            logger.warning("Missing values found in data. Filling with column means.")
            # Fill missing values with column means
            col_means = np.nanmean(X, axis=0)
            for i, mean_val in enumerate(col_means):
                mask = np.isnan(X[:, i])
                X[mask, i] = mean_val
        
        # Store metadata
        self.metadata['n_samples'] = X.shape[0]
        self.metadata['n_features'] = X.shape[1]
        
        # Create PCA pipeline
        if self.standardize:
            # Standardize then apply PCA
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Determine number of components if not specified
            if self.n_components is None:
                # First fit with all components
                temp_pca = PCA(random_state=self.random_state)
                temp_pca.fit(X_scaled)
                
                # Determine number of components based on variance threshold
                explained_variance_ratio = temp_pca.explained_variance_ratio_
                cumulative_variance = np.cumsum(explained_variance_ratio)
                
                # Find minimum components to explain desired variance
                n_components = np.argmax(cumulative_variance >= self.variance_threshold) + 1
                n_components = min(n_components, len(columns) - 1)  # Ensure we don't use all components
                
                logger.info(f"Auto-selected {n_components} components to explain {self.variance_threshold*100:.1f}% variance")
                self.n_components = n_components
            
            # Create PCA with determined number of components
            self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
            self.pca.fit(X_scaled)
            
        else:
            # Apply PCA without standardization
            # Determine number of components if not specified
            if self.n_components is None:
                # First fit with all components
                temp_pca = PCA(random_state=self.random_state)
                temp_pca.fit(X)
                
                # Determine number of components based on variance threshold
                explained_variance_ratio = temp_pca.explained_variance_ratio_
                cumulative_variance = np.cumsum(explained_variance_ratio)
                
                # Find minimum components to explain desired variance
                n_components = np.argmax(cumulative_variance >= self.variance_threshold) + 1
                n_components = min(n_components, len(columns) - 1)  # Ensure we don't use all components
                
                logger.info(f"Auto-selected {n_components} components to explain {self.variance_threshold*100:.1f}% variance")
                self.n_components = n_components
            
            # Create PCA with determined number of components
            self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
            self.pca.fit(X)
        
        # Update metadata
        self.metadata['n_components'] = self.pca.n_components_
        self.metadata['explained_variance_ratio'] = self.pca.explained_variance_ratio_.tolist()
        self.metadata['cumulative_variance_ratio'] = np.cumsum(self.pca.explained_variance_ratio_).tolist()
        
        # Set fitted flag
        self.is_fitted = True
        
        return self
    
    def transform(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Transform data to PCA space.
        
        Args:
            df: DataFrame to transform
            columns: Columns to use (None for using same columns as in fit)
            
        Returns:
            DataFrame with PCA components
        """
        if not self.is_fitted:
            raise ValueError("PCA must be fitted before transform")
        
        # Use columns from fit if not specified
        if columns is None:
            columns = self.feature_names
        else:
            # Check for consistency with fitted columns
            missing_cols = [col for col in self.feature_names if col not in columns]
            if missing_cols:
                logger.warning(f"Missing columns from fit: {missing_cols}")
        
        # Extract feature matrix
        X = df[columns].values
        
        # Handle missing values if any
        if np.isnan(X).any():
            logger.warning("Missing values found in data. Filling with column means.")
            # Fill missing values with column means
            col_means = np.nanmean(X, axis=0)
            for i, mean_val in enumerate(col_means):
                mask = np.isnan(X[:, i])
                X[mask, i] = mean_val
        
        # Apply transformation
        if self.standardize:
            X_scaled = self.scaler.transform(X)
            X_pca = self.pca.transform(X_scaled)
        else:
            X_pca = self.pca.transform(X)
        
        # Create DataFrame with PCA results
        pca_df = pd.DataFrame(
            X_pca, 
            columns=[f'PC{i+1}' for i in range(self.pca.n_components_)],
            index=df.index
        )
        
        return pca_df
    
    def fit_transform(self, df: pd.DataFrame, columns: Optional[List[str]] = None,
                     exclude_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            df: DataFrame with features
            columns: Columns to use (None for all numeric columns)
            exclude_cols: Columns to exclude (e.g., date, ID)
            
        Returns:
            DataFrame with PCA components
        """
        self.fit(df, columns, exclude_cols)
        return self.transform(df, self.feature_names)
    
    def get_loadings(self) -> pd.DataFrame:
        """
        Get the loadings matrix showing how original variables map to components.
        
        Returns:
            DataFrame with loadings (features x components)
        """
        if not self.is_fitted:
            raise ValueError("PCA must be fitted before getting loadings")
        
        # Get components matrix (also called loadings)
        loadings = self.pca.components_.T
        
        # Create DataFrame with loadings
        loadings_df = pd.DataFrame(
            loadings,
            columns=[f'PC{i+1}' for i in range(self.pca.n_components_)],
            index=self.feature_names
        )
        
        return loadings_df
    
    def get_explained_variance(self) -> Dict[str, List[float]]:
        """
        Get explained variance ratio for each component.
        
        Returns:
            Dictionary with explained variance information
        """
        if not self.is_fitted:
            raise ValueError("PCA must be fitted before getting explained variance")
        
        return {
            'explained_variance_ratio': self.pca.explained_variance_ratio_.tolist(),
            'cumulative_variance': np.cumsum(self.pca.explained_variance_ratio_).tolist()
        }
    
    def get_feature_importance(self, feature_names: Optional[List[str]] = None,
                              n_components: Optional[int] = None) -> pd.DataFrame:
        """
        Calculate importance of original features based on PCA loadings.
        
        Args:
            feature_names: Names of original features (None for using names from fit)
            n_components: Number of top components to consider (None for all)
            
        Returns:
            DataFrame with feature importance scores
        """
        if not self.is_fitted:
            raise ValueError("PCA must be fitted before getting feature importance")
        
        # Use feature names from fit if not specified
        if feature_names is None:
            feature_names = self.feature_names
        
        # Determine number of components to use
        if n_components is None:
            n_components = self.pca.n_components_
        else:
            n_components = min(n_components, self.pca.n_components_)
        
        # Get loadings for top components
        loadings = self.pca.components_[:n_components].T
        explained_variance = self.pca.explained_variance_ratio_[:n_components]
        
        # Calculate weighted importance score
        # For each feature, sum the squared loading multiplied by the variance explained
        importance_scores = np.zeros(len(feature_names))
        
        for i in range(n_components):
            importance_scores += (loadings[:, i] ** 2) * explained_variance[i]
        
        # Normalize scores to sum to 1
        importance_scores /= np.sum(importance_scores)
        
        # Create DataFrame with importance scores
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        })
        
        # Sort by importance (descending)
        importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
        
        return importance_df
    
    def get_feature_contributions(self, component_idx: int = 0) -> pd.DataFrame:
        """
        Get contributions of original features to a specific principal component.
        
        Args:
            component_idx: Index of the component (0-based)
            
        Returns:
            DataFrame with feature contributions
        """
        if not self.is_fitted:
            raise ValueError("PCA must be fitted before getting feature contributions")
        
        if component_idx >= self.pca.n_components_:
            raise ValueError(f"Component index {component_idx} out of range (0-{self.pca.n_components_-1})")
        
        # Get loadings for the specified component
        loadings = self.pca.components_[component_idx]
        
        # Square loadings to get contribution (removes sign)
        contributions = loadings ** 2
        
        # Normalize to sum to 1
        contributions = contributions / np.sum(contributions)
        
        # Create DataFrame with contributions
        contribution_df = pd.DataFrame({
            'feature': self.feature_names,
            'loading': loadings,
            'contribution': contributions
        })
        
        # Sort by absolute loading (descending)
        contribution_df = contribution_df.sort_values('contribution', ascending=False).reset_index(drop=True)
        
        return contribution_df
    
    def plot_explained_variance(self, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot explained variance by component.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if not self.is_fitted:
            raise ValueError("PCA must be fitted before plotting")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get explained variance data
        explained_variance = self.pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        # Component indices (1-based for display)
        components = np.arange(1, len(explained_variance) + 1)
        
        # Plot individual and cumulative explained variance
        ax.bar(components, explained_variance, alpha=0.7, label='Individual')
        ax.step(components, cumulative_variance, where='mid', label='Cumulative', color='red')
        ax.axhline(y=self.variance_threshold, color='green', linestyle='--', 
                  label=f'{self.variance_threshold*100:.0f}% Threshold')
        
        # Add labels and title
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Explained Variance Ratio')
        ax.set_title('Explained Variance by Principal Component')
        ax.set_xticks(components)
        ax.set_ylim(0, 1.05)
        
        # Add threshold marker
        threshold_component = np.argmax(cumulative_variance >= self.variance_threshold) + 1
        ax.axvline(x=threshold_component, color='green', linestyle='--', alpha=0.7)
        
        # Add percentage annotations
        for i, (var, cum_var) in enumerate(zip(explained_variance, cumulative_variance)):
            ax.annotate(f'{var:.1%}', xy=(i+1, var), xytext=(0, 5), 
                       textcoords='offset points', ha='center')
            
            if i == len(explained_variance)-1 or i == threshold_component-1:
                ax.annotate(f'{cum_var:.1%}', xy=(i+1, cum_var), xytext=(0, -15), 
                           textcoords='offset points', ha='center', color='red')
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig
    
    def plot_loadings(self, n_components: int = 2, top_n_features: int = 10,
                     figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot component loadings for top features.
        
        Args:
            n_components: Number of components to plot
            top_n_features: Number of top features to include for each component
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if not self.is_fitted:
            raise ValueError("PCA must be fitted before plotting loadings")
        
        # Limit to available components
        n_components = min(n_components, self.pca.n_components_)
        
        # Create figure
        fig, axes = plt.subplots(1, n_components, figsize=figsize, sharey=True)
        
        # Handle case of single component
        if n_components == 1:
            axes = [axes]
        
        # For each component
        for i in range(n_components):
            # Get top features for this component
            component_loadings = self.pca.components_[i]
            feature_loadings = list(zip(self.feature_names, component_loadings))
            
            # Sort by absolute loading value
            sorted_loadings = sorted(feature_loadings, key=lambda x: abs(x[1]), reverse=True)
            
            # Take top N features
            top_features = sorted_loadings[:top_n_features]
            
            # Extract names and loadings
            feature_names = [f[0] for f in top_features]
            loadings = [f[1] for f in top_features]
            
            # Plot horizontal bar chart
            colors = ['steelblue' if l > 0 else 'tomato' for l in loadings]
            axes[i].barh(range(len(feature_names)), loadings, color=colors)
            
            # Add feature names
            axes[i].set_yticks(range(len(feature_names)))
            axes[i].set_yticklabels(feature_names)
            
            # Add component title and explained variance
            explained_var = self.pca.explained_variance_ratio_[i]
            axes[i].set_title(f'PC{i+1} ({explained_var:.1%} variance)')
            
            # Add vertical line at 0
            axes[i].axvline(x=0, color='gray', linestyle='--', alpha=0.7)
            
            # Add grid
            axes[i].grid(True, alpha=0.3)
        
        # Add common y-label
        fig.text(0.01, 0.5, 'Feature', va='center', rotation='vertical')
        
        # Add common x-label
        fig.text(0.5, 0.01, 'Loading', ha='center')
        
        # Add overall title
        fig.suptitle('PCA Component Loadings', fontsize=14)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, bottom=0.1)
        
        return fig
    
    def plot_biplot(self, transformed_data: pd.DataFrame, 
                   pc_x: int = 1, pc_y: int = 2,
                   n_features: int = 10,
                   figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
        """
        Create a biplot of data points and feature loadings.
        
        Args:
            transformed_data: DataFrame with transformed data (from transform())
            pc_x: Component for x-axis (1-based)
            pc_y: Component for y-axis (1-based)
            n_features: Number of top features to display
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if not self.is_fitted:
            raise ValueError("PCA must be fitted before creating biplot")
        
        # Check component indices
        if pc_x < 1 or pc_x > self.pca.n_components_ or pc_y < 1 or pc_y > self.pca.n_components_:
            raise ValueError(f"Component indices must be between 1 and {self.pca.n_components_}")
        
        # Convert to 0-based indices
        pc_x = pc_x - 1
        pc_y = pc_y - 1
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot transformed data points
        x_col = f'PC{pc_x+1}'
        y_col = f'PC{pc_y+1}'
        
        if x_col not in transformed_data.columns or y_col not in transformed_data.columns:
            raise ValueError(f"Transformed data must contain {x_col} and {y_col}")
            
        ax.scatter(transformed_data[x_col], transformed_data[y_col], alpha=0.7, s=30)
        
        # Get scaling factor for feature vectors
        # This is to make the arrows visible on the same scale as the data points
        scale_x = np.max(np.abs(transformed_data[x_col])) / np.max(np.abs(self.pca.components_[pc_x])) * 0.7
        scale_y = np.max(np.abs(transformed_data[y_col])) / np.max(np.abs(self.pca.components_[pc_y])) * 0.7
        
        # Calculate importance of features for these two components
        feature_importance = np.sqrt(self.pca.components_[pc_x]**2 + self.pca.components_[pc_y]**2)
        
        # Get top features
        indices = np.argsort(feature_importance)[-n_features:]
        
        # Plot feature vectors
        for i in indices:
            ax.arrow(0, 0, 
                    self.pca.components_[pc_x, i] * scale_x, 
                    self.pca.components_[pc_y, i] * scale_y,
                    head_width=0.05, head_length=0.05, fc='red', ec='red')
            
            # Add feature name
            ax.text(self.pca.components_[pc_x, i] * scale_x * 1.15, 
                   self.pca.components_[pc_y, i] * scale_y * 1.15, 
                   self.feature_names[i], color='red')
        
        # Add labels and title
        explained_var_x = self.pca.explained_variance_ratio_[pc_x]
        explained_var_y = self.pca.explained_variance_ratio_[pc_y]
        
        ax.set_xlabel(f'PC{pc_x+1} ({explained_var_x:.1%} variance)')
        ax.set_ylabel(f'PC{pc_y+1} ({explained_var_y:.1%} variance)')
        ax.set_title(f'PCA Biplot: PC{pc_x+1} vs PC{pc_y+1}', fontsize=14)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add origin lines
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        return fig
    
    def save(self, filepath: str) -> None:
        """
        Save the PCA model to a file.
        
        Args:
            filepath: Path to save the model
        """
        import joblib
        
        if not self.is_fitted:
            raise ValueError("Cannot save an unfitted model")
        
        # Prepare model data
        model_data = {
            'scaler': self.scaler,
            'pca': self.pca,
            'standardize': self.standardize,
            'n_components': self.n_components,
            'variance_threshold': self.variance_threshold,
            'random_state': self.random_state,
            'feature_names': self.feature_names,
            'exclude_cols': self.exclude_cols,
            'metadata': self.metadata,
            'is_fitted': self.is_fitted
        }
        
        # Save to file
        joblib.dump(model_data, filepath)
        logger.info(f"PCA model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'PCAnalyzer':
        """
        Load a PCA model from a file.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Loaded PCAnalyzer instance
        """
        import joblib
        
        # Load model data
        model_data = joblib.load(filepath)
        
        # Create instance with same parameters
        instance = cls(
            standardize=model_data['standardize'],
            n_components=model_data['n_components'],
            variance_threshold=model_data['variance_threshold'],
            random_state=model_data['random_state']
        )
        
        # Restore model state
        instance.scaler = model_data['scaler']
        instance.pca = model_data['pca']
        instance.feature_names = model_data['feature_names']
        instance.exclude_cols = model_data['exclude_cols']
        instance.metadata = model_data['metadata']
        instance.is_fitted = model_data['is_fitted']
        
        logger.info(f"PCA model loaded from {filepath}")
        return instance
    
    def __repr__(self) -> str:
        """String representation of the PCAnalyzer."""
        status = "fitted" if self.is_fitted else "not fitted"
        return f"PCAnalyzer(n_components={self.n_components}, standardize={self.standardize}, {status})"


class TimeSeriesPCA(PCAnalyzer):
    """
    Extension of PCAnalyzer specifically for time series data.
    Handles proper train/test splitting and time-aware transformations.
    """
    
    def __init__(self, standardize: bool = True, 
                 n_components: Optional[int] = None,
                 variance_threshold: float = 0.95,
                 random_state: int = 42,
                 date_col: str = 'date'):
        """
        Initialize TimeSeriesPCA.
        
        Args:
            standardize: Whether to standardize data before PCA
            n_components: Number of components to keep (None for auto-selection)
            variance_threshold: Minimum variance to explain when auto-selecting components
            random_state: Random seed for reproducibility
            date_col: Name of date column
        """
        super().__init__(standardize, n_components, variance_threshold, random_state)
        self.date_col = date_col
    
    def fit_with_time_split(self, df: pd.DataFrame, 
                          columns: Optional[List[str]] = None,
                          exclude_cols: Optional[List[str]] = None,
                          train_ratio: float = 0.8) -> 'TimeSeriesPCA':
        """
        Fit PCA using only a training portion of the time series.
        
        Args:
            df: DataFrame with features
            columns: Columns to use (None for all numeric columns)
            exclude_cols: Columns to exclude (e.g., date, ID)
            train_ratio: Proportion of data to use for training
            
        Returns:
            Self for method chaining
        """
        # Ensure date column exists
        if self.date_col not in df.columns:
            raise ValueError(f"Date column '{self.date_col}' not found in DataFrame")
        
        # Sort by date
        df_sorted = df.sort_values(self.date_col).reset_index(drop=True)
        
        # Split into train and test
        train_size = int(len(df_sorted) * train_ratio)
        df_train = df_sorted.iloc[:train_size].copy()
        
        logger.info(f"Fitting PCA on training portion ({train_size} of {len(df_sorted)} rows)")
        
        # Fit on training data
        return super().fit(df_train, columns, exclude_cols)
    
    def transform_with_plotting(self, df: pd.DataFrame,
                              columns: Optional[List[str]] = None,
                              plot_components: List[int] = [1, 2],
                              figsize: Tuple[int, int] = (12, 6)) -> Tuple[pd.DataFrame, plt.Figure]:
        """
        Transform data and create time series plot of components.
        
        Args:
            df: DataFrame to transform
            columns: Columns to use (None for using same columns as in fit)
            plot_components: List of components to plot (1-based indices)
            figsize: Figure size
            
        Returns:
            Tuple of (transformed DataFrame, matplotlib figure)
        """
        if not self.is_fitted:
            raise ValueError("PCA must be fitted before transform_with_plotting")
        
        # Ensure date column exists
        if self.date_col not in df.columns:
            raise ValueError(f"Date column '{self.date_col}' not found in DataFrame")
        
        # Transform data
        pca_df = self.transform(df, columns)
        
        # Add date column
        pca_df[self.date_col] = df[self.date_col].values
        
        # Create figure for time series plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Sort by date
        pca_df_sorted = pca_df.sort_values(self.date_col)
        
        # Plot each requested component
        for pc in plot_components:
            if pc < 1 or pc > self.pca.n_components_:
                logger.warning(f"Component {pc} out of range, skipping")
                continue
                
            comp_col = f'PC{pc}'
            ax.plot(pca_df_sorted[self.date_col], pca_df_sorted[comp_col], 
                   label=f'{comp_col} ({self.pca.explained_variance_ratio_[pc-1]:.1%} var)')
        
        # Add labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Component Value')
        ax.set_title('PCA Components Over Time')
        
        # Format x-axis as dates
        fig.autofmt_xdate()
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return pca_df, fig
    
    def get_temporal_stability(self, df: pd.DataFrame,
                              columns: Optional[List[str]] = None,
                              window_size: int = 52,
                              step_size: int = 26) -> Dict[str, Any]:
        """
        Assess stability of PCA components over time using rolling windows.
        
        Args:
            df: DataFrame with time series data
            columns: Columns to use (None for using same columns as in fit)
            window_size: Size of rolling window
            step_size: Step size for rolling window
            
        Returns:
            Dictionary with stability metrics
        """
        if not self.is_fitted:
            raise ValueError("PCA must be fitted before assessing temporal stability")
        
        # Ensure date column exists
        if self.date_col not in df.columns:
            raise ValueError(f"Date column '{self.date_col}' not found in DataFrame")
        
        # Use columns from fit if not specified
        if columns is None:
            columns = self.feature_names
        
        # Sort by date
        df_sorted = df.sort_values(self.date_col).reset_index(drop=True)
        
        # Check if we have enough data for multiple windows
        if len(df_sorted) < window_size * 2:
            logger.warning(f"Not enough data for multiple windows (need at least {window_size*2} rows)")
            return {'error': 'Not enough data for multiple windows'}
        
        # Create windows
        windows = []
        for start in range(0, len(df_sorted) - window_size + 1, step_size):
            end = start + window_size
            windows.append((start, end))
        
        # Apply PCA to each window
        window_results = []
        loading_matrices = []
        
        for i, (start, end) in enumerate(windows):
            window_df = df_sorted.iloc[start:end]
            
            try:
                # Create a new PCA analyzer with same settings
                window_pca = PCAnalyzer(
                    standardize=self.standardize,
                    n_components=self.n_components,
                    variance_threshold=self.variance_threshold,
                    random_state=self.random_state
                )
                
                # Fit PCA to window
                window_pca.fit(window_df, columns, exclude_cols=self.exclude_cols)
                
                # Get loadings
                loadings = window_pca.get_loadings()
                loading_matrices.append(loadings.values)
                
                # Store window info
                window_results.append({
                    'window_id': i,
                    'start_idx': start,
                    'end_idx': end,
                    'start_date': window_df[self.date_col].iloc[0],
                    'end_date': window_df[self.date_col].iloc[-1],
                    'explained_variance': window_pca.pca.explained_variance_ratio_.tolist()
                })
                
            except Exception as e:
                logger.warning(f"Error applying PCA to window {i}: {e}")
        
        if not window_results:
            return {'error': 'Could not apply PCA to any window'}
        
        # Calculate stability metrics
        stability_metrics = {}
        
        # 1. Variance in explained variance across windows
        all_exp_var = np.array([w['explained_variance'] for w in window_results])
        stability_metrics['explained_variance_stability'] = {
            'mean': np.mean(all_exp_var, axis=0).tolist(),
            'std': np.std(all_exp_var, axis=0).tolist(),
            'cv': (np.std(all_exp_var, axis=0) / np.mean(all_exp_var, axis=0)).tolist()
        }
        
        # 2. Loading similarity across windows
        if len(loading_matrices) >= 2:
            # For each pair of windows, calculate loadings similarity
            # Using RV coefficient (vector correlation)
            similarity_scores = []
            
            for i in range(len(loading_matrices)):
                for j in range(i+1, len(loading_matrices)):
                    # Get loadings for two windows
                    loadings_i = loading_matrices[i]
                    loadings_j = loading_matrices[j]
                    
                    # Calculate similarity (modified RV coefficient)
                    # Higher value means more similar loadings
                    similarity = self._calculate_loading_similarity(loadings_i, loadings_j)
                    
                    similarity_scores.append({
                        'window_i': i,
                        'window_j': j,
                        'similarity': similarity
                    })
            
            # Calculate average similarity
            stability_metrics['loading_similarity'] = {
                'scores': similarity_scores,
                'mean': np.mean([s['similarity'] for s in similarity_scores]),
                'min': np.min([s['similarity'] for s in similarity_scores]),
                'max': np.max([s['similarity'] for s in similarity_scores])
            }
        
        # 3. Feature importance stability
        feature_importance_by_window = []
        
        for i, (start, end) in enumerate(windows):
            window_df = df_sorted.iloc[start:end]
            
            try:
                # Create a new PCA analyzer
                window_pca = PCAnalyzer(
                    standardize=self.standardize,
                    n_components=self.n_components,
                    variance_threshold=self.variance_threshold,
                    random_state=self.random_state
                )
                
                # Fit PCA to window
                window_pca.fit(window_df, columns, exclude_cols=self.exclude_cols)
                
                # Get feature importance
                importance = window_pca.get_feature_importance()
                
                # Store as dictionary for easier comparison
                importance_dict = dict(zip(importance['feature'], importance['importance']))
                feature_importance_by_window.append(importance_dict)
                
            except Exception as e:
                logger.warning(f"Error calculating feature importance for window {i}: {e}")
        
        if feature_importance_by_window:
            # Calculate stability of feature importance
            all_features = set()
            for imp in feature_importance_by_window:
                all_features.update(imp.keys())
            
            # For each feature, calculate importance stability
            feature_stability = {}
            
            for feature in all_features:
                # Get importance across windows
                importance_values = [imp.get(feature, 0) for imp in feature_importance_by_window]
                
                # Calculate statistics
                feature_stability[feature] = {
                    'mean': np.mean(importance_values),
                    'std': np.std(importance_values),
                    'cv': np.std(importance_values) / np.mean(importance_values) if np.mean(importance_values) > 0 else 0
                }
            
            # Sort features by mean importance
            sorted_features = sorted(feature_stability.items(), 
                                    key=lambda x: x[1]['mean'], 
                                    reverse=True)
            
            stability_metrics['feature_importance_stability'] = {
                'features': {f[0]: f[1] for f in sorted_features},
                'average_cv': np.mean([f[1]['cv'] for f in sorted_features])
            }
        
        # 4. Overall stability assessment
        if 'loading_similarity' in stability_metrics:
            mean_similarity = stability_metrics['loading_similarity']['mean']
            
            if mean_similarity > 0.8:
                stability_metrics['overall_stability'] = 'High'
            elif mean_similarity > 0.5:
                stability_metrics['overall_stability'] = 'Medium'
            else:
                stability_metrics['overall_stability'] = 'Low'
        
        # Store window info
        stability_metrics['windows'] = window_results
        
        return stability_metrics
    
    def _calculate_loading_similarity(self, loadings_i: np.ndarray, loadings_j: np.ndarray) -> float:
        """
        Calculate similarity between two loading matrices.
        Uses a modified version of the RV coefficient for matrix correlation.
        
        Args:
            loadings_i: First loading matrix
            loadings_j: Second loading matrix
            
        Returns:
            Similarity score (0-1, higher is more similar)
        """
        # Handle potential sign flips in PCA components
        # For each column in j, try both original and flipped version
        best_similarity = 0
        
        # For each component in the second loading matrix
        for col in range(loadings_j.shape[1]):
            # Try both original and flipped version
            for flip in [1, -1]:
                # Create modified version with potential flip
                loadings_j_mod = loadings_j.copy()
                loadings_j_mod[:, col] *= flip
                
                # Calculate correlation for each feature
                feature_similarities = []
                
                for row in range(loadings_i.shape[0]):
                    # Correlation between feature loadings across components
                    correlation = np.corrcoef(loadings_i[row, :], loadings_j_mod[row, :])[0, 1]
                    feature_similarities.append(abs(correlation))  # Take absolute value
                
                # Average similarity across features
                avg_similarity = np.mean(feature_similarities)
                
                # Update best similarity
                best_similarity = max(best_similarity, avg_similarity)
        
        return best_similarity


def apply_pca_with_date_preservation(df: pd.DataFrame, 
                                   columns: Optional[List[str]] = None,
                                   date_col: str = 'date',
                                   n_components: Optional[int] = None,
                                   variance_threshold: float = 0.95,
                                   standardize: bool = True) -> Tuple[pd.DataFrame, PCAnalyzer]:
    """
    Apply PCA to a DataFrame and preserve date column.
    
    Args:
        df: DataFrame to transform
        columns: Columns to use for PCA (None for all numeric columns)
        date_col: Name of date column
        n_components: Number of components (None for auto-selection)
        variance_threshold: Threshold for explained variance
        standardize: Whether to standardize data
        
    Returns:
        Tuple of (transformed DataFrame with date, fitted PCAnalyzer)
    """
    # Check if date column exists
    if date_col not in df.columns:
        logger.warning(f"Date column '{date_col}' not found in DataFrame")
    
    # Create PCAnalyzer
    analyzer = PCAnalyzer(
        standardize=standardize,
        n_components=n_components,
        variance_threshold=variance_threshold
    )
    
    # Exclude date column from PCA
    exclude_cols = [date_col]
    
    # Fit and transform
    pca_df = analyzer.fit_transform(df, columns, exclude_cols)
    
    # Add date column back
    if date_col in df.columns:
        pca_df[date_col] = df[date_col].values
    
    return pca_df, analyzer