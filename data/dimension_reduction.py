"""
Dimension reduction module for time series feature selection.
Provides functionality for PCA and feature selection methods.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

# Setup logging
logger = logging.getLogger(__name__)

def perform_pca(df: pd.DataFrame, 
                numeric_cols: List[str], 
                variance_threshold: float = 0.95,
                standardize: bool = True,
                n_components: Optional[int] = None) -> Tuple[pd.DataFrame, PCA, StandardScaler, List[str]]:
    """
    Perform PCA on selected numeric columns with standardization option.
    
    Args:
        df: DataFrame with features
        numeric_cols: List of numeric columns to include in PCA
        variance_threshold: Minimum cumulative explained variance (if n_components not specified)
        standardize: Whether to standardize the data before PCA
        n_components: Number of components to keep (overrides variance_threshold if specified)
        
    Returns:
        Tuple of (DataFrame with PCA components, fitted PCA object, 
                 fitted StandardScaler, list of feature names used)
    """
    if df.empty or not numeric_cols:
        logger.warning("Empty DataFrame or no numeric columns provided")
        return df.copy(), None, None, []
    
    # Filter to only include columns that exist in the DataFrame
    valid_cols = [col for col in numeric_cols if col in df.columns]
    if not valid_cols:
        logger.warning("None of the provided numeric columns exist in the DataFrame")
        return df.copy(), None, None, []
    
    # Extract features
    features = df[valid_cols].copy()
    
    # Handle missing values
    features = features.fillna(features.mean())
    
    # Initialize scaler
    scaler = None
    
    # Standardize if requested
    if standardize:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
    else:
        features_scaled = features.values
    
    # Determine number of components
    if n_components is None:
        # Start with min(n_samples, n_features) components
        n_components = min(features_scaled.shape[0], features_scaled.shape[1])
    
    # Initialize and fit PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(features_scaled)
    
    # If variance threshold is specified, reduce components further
    if n_components is None and variance_threshold < 1.0:
        # Find number of components needed to reach variance threshold
        explained_variance_ratio_cumsum = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(explained_variance_ratio_cumsum >= variance_threshold) + 1
        
        # Apply PCA again with the determined number of components
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(features_scaled)
    
    # Create DataFrame with principal components
    pc_columns = [f'PC{i+1}' for i in range(principal_components.shape[1])]
    pc_df = pd.DataFrame(data=principal_components, columns=pc_columns, index=df.index)
    
    # Log explained variance
    logger.info(f"PCA explained variance: {pca.explained_variance_ratio_}")
    logger.info(f"Total explained variance: {sum(pca.explained_variance_ratio_):.4f}")
    
    return pc_df, pca, scaler, valid_cols

def analyze_pca_loadings(pca: PCA, feature_names: List[str], 
                         top_n_components: int = 5, top_n_features: int = 5) -> Dict:
    """
    Analyze PCA loadings to determine feature importance for each component.
    
    Args:
        pca: Fitted PCA object
        feature_names: Names of original features
        top_n_components: Number of top components to analyze
        top_n_features: Number of top features to report per component
        
    Returns:
        Dictionary with component analysis
    """
    if pca is None or not hasattr(pca, 'components_'):
        logger.warning("Invalid PCA object provided")
        return {}
    
    # Limit number of components to analyze
    n_components = min(top_n_components, pca.n_components_)
    
    result = {}
    
    # For each component, get the most important features
    for i in range(n_components):
        # Get absolute loadings
        abs_loadings = np.abs(pca.components_[i])
        
        # Get indices of top features
        top_indices = abs_loadings.argsort()[-top_n_features:][::-1]
        
        # Get feature names and loadings
        top_features = [(feature_names[idx], pca.components_[i][idx]) for idx in top_indices]
        
        # Add to result
        result[f'PC{i+1}'] = {
            'explained_variance': pca.explained_variance_ratio_[i],
            'top_features': top_features
        }
    
    return result

def select_features_mutual_info(df: pd.DataFrame, 
                               target_col: str, 
                               feature_cols: List[str],
                               top_n: int = 10) -> List[str]:
    """
    Select features based on mutual information with target.
    
    Args:
        df: DataFrame with features and target
        target_col: Name of target column
        feature_cols: List of feature columns to consider
        top_n: Number of top features to select
        
    Returns:
        List of selected feature names
    """
    if df.empty or target_col not in df.columns:
        logger.warning("Empty DataFrame or target column not found")
        return []
    
    # Filter to only include columns that exist in the DataFrame
    valid_cols = [col for col in feature_cols if col in df.columns]
    if not valid_cols:
        logger.warning("None of the provided feature columns exist in the DataFrame")
        return []
    
    # Extract features and target
    X = df[valid_cols].copy()
    y = df[target_col].copy()
    
    # Handle missing values
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())
    
    # Calculate mutual information
    mi_scores = mutual_info_regression(X, y)
    mi_scores = pd.Series(mi_scores, index=valid_cols)
    
    # Select top features
    top_features = mi_scores.sort_values(ascending=False).head(top_n).index.tolist()
    
    return top_features

def select_features_f_regression(df: pd.DataFrame, 
                                target_col: str, 
                                feature_cols: List[str],
                                top_n: int = 10) -> List[str]:
    """
    Select features based on F-regression test with target.
    
    Args:
        df: DataFrame with features and target
        target_col: Name of target column
        feature_cols: List of feature columns to consider
        top_n: Number of top features to select
        
    Returns:
        List of selected feature names
    """
    if df.empty or target_col not in df.columns:
        logger.warning("Empty DataFrame or target column not found")
        return []
    
    # Filter to only include columns that exist in the DataFrame
    valid_cols = [col for col in feature_cols if col in df.columns]
    if not valid_cols:
        logger.warning("None of the provided feature columns exist in the DataFrame")
        return []
    
    # Extract features and target
    X = df[valid_cols].copy()
    y = df[target_col].copy()
    
    # Handle missing values
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())
    
    # Calculate F-statistics
    f_values, p_values = f_regression(X, y)
    f_scores = pd.Series(f_values, index=valid_cols)
    
    # Select top features
    top_features = f_scores.sort_values(ascending=False).head(top_n).index.tolist()
    
    return top_features

def select_features_lasso(df: pd.DataFrame, 
                         target_col: str, 
                         feature_cols: List[str],
                         alpha: float = 0.01) -> List[str]:
    """
    Select features using Lasso regression.
    
    Args:
        df: DataFrame with features and target
        target_col: Name of target column
        feature_cols: List of feature columns to consider
        alpha: Regularization strength
        
    Returns:
        List of selected feature names
    """
    if df.empty or target_col not in df.columns:
        logger.warning("Empty DataFrame or target column not found")
        return []
    
    # Filter to only include columns that exist in the DataFrame
    valid_cols = [col for col in feature_cols if col in df.columns]
    if not valid_cols:
        logger.warning("None of the provided feature columns exist in the DataFrame")
        return []
    
    # Extract features and target
    X = df[valid_cols].copy()
    y = df[target_col].copy()
    
    # Handle missing values
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit Lasso model
    lasso = Lasso(alpha=alpha, random_state=42)
    lasso.fit(X_scaled, y)
    
    # Get feature coefficients
    coef = pd.Series(lasso.coef_, index=valid_cols)
    
    # Select features with non-zero coefficients
    selected_features = coef[coef != 0].index.tolist()
    
    return selected_features

def reduce_dimensions(df: pd.DataFrame, 
                     target_col: str,
                     feature_cols: Optional[List[str]] = None,
                     date_col: str = 'date',
                     use_pca: bool = True,
                     pca_variance_threshold: float = 0.95,
                     feature_selection_method: str = 'mutual_info',
                     top_n_features: int = 10,
                     lasso_alpha: float = 0.01,
                     data_loader: Optional[Any] = None,
                     frequency: Optional[str] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Main function to reduce dimensions and select features.
    
    Args:
        df: DataFrame with features and target
        target_col: Name of target column
        feature_cols: List of feature columns (None for auto-detect)
        date_col: Name of date column
        use_pca: Whether to use PCA
        pca_variance_threshold: Minimum cumulative explained variance
        feature_selection_method: Method for feature selection
        top_n_features: Number of top features to select
        lasso_alpha: Alpha parameter for Lasso feature selection
        data_loader: Optional DataLoader instance for missing value handling
        frequency: Optional frequency of data for group-based imputation
        
    Returns:
        Tuple of (DataFrame with selected features, dictionary with metadata)
    """
    if df.empty:
        logger.warning("Empty DataFrame provided")
        return df.copy(), {}
    
    # Auto-detect feature columns if not provided
    if feature_cols is None:
        # Get numeric columns
        feature_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # Remove date column and target column if they exist
        if date_col in feature_cols:
            feature_cols.remove(date_col)
        if target_col in feature_cols:
            feature_cols.remove(target_col)
            
        # Remove date-related columns
        date_related_cols = ['year', 'month', 'quarter', 'month_end', 'quarter_end', 
                           'Q1', 'Q2', 'Q3', 'Q4', 'day_of_year', 'day_of_month', 
                           'week_of_year']
        feature_cols = [col for col in feature_cols if col not in date_related_cols]
    
    # Filter out highly correlated features before PCA
    orig_feature_count = len(feature_cols)
    feature_df = df[feature_cols].copy()
    
    # Handle missing values in features consistently
    if data_loader is not None and frequency is not None:
        if frequency in ['monthly', 'quarterly']:
            fill_groups = ['year', 'month' if frequency == 'monthly' else 'quarter']
            if all(group in df.columns for group in fill_groups):
                # Create a temporary df with features and grouping columns
                temp_df = feature_df.copy()
                for group in fill_groups:
                    if group not in temp_df.columns and group in df.columns:
                        temp_df[group] = df[group]
                feature_df = data_loader.handle_missing_values(temp_df, method='mean', fill_groups=fill_groups)
                # Remove the grouping columns after imputation
                feature_df = feature_df[feature_cols]
            else:
                feature_df = data_loader.handle_missing_values(feature_df, method='mean')
        else:
            feature_df = data_loader.handle_missing_values(feature_df, method='mean')
    else:
        # Simple imputation as fallback
        feature_df = feature_df.fillna(feature_df.mean())
    
    # Simple correlation filtering
    corr_matrix = feature_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    
    if to_drop:
        logger.info(f"Removing {len(to_drop)} highly correlated features")
        reduced_features = [col for col in feature_cols if col not in to_drop]
        feature_cols = reduced_features
    
    logger.info(f"Feature count reduced from {orig_feature_count} to {len(feature_cols)} after correlation filtering")
    
    metadata = {
        'original_features': feature_cols.copy(),
        'pca_applied': False,
        'feature_selection_method': feature_selection_method,
        'selected_features': []
    }
    
    # Create result DataFrame with date and target
    result_cols = [date_col, target_col] if date_col in df.columns else [target_col]
    result = df[result_cols].copy()
    
    # Apply PCA if requested
    if use_pca:
        # Calculate initial PCA to determine optimal number of components
        pc_df, pca_obj, scaler, valid_features = perform_pca(
            df, feature_cols, variance_threshold=1.0  # Get all components first
        )
        
        if pca_obj is not None:
            # Get components that explain up to threshold variance
            explained_variance = np.cumsum(pca_obj.explained_variance_ratio_)
            n_components = np.argmax(explained_variance >= pca_variance_threshold) + 1
            
            logger.info(f"Using {n_components} PCA components to explain {pca_variance_threshold*100:.1f}% of variance")
            
            # Now perform PCA with the determined number of components
            pc_df, pca, scaler, valid_features = perform_pca(
                df, feature_cols, n_components=n_components
            )
            
            if pca is not None:
                # Add PCA components to result
                result = pd.concat([result, pc_df], axis=1)
                
                # Analyze PCA loadings
                pca_analysis = analyze_pca_loadings(pca, valid_features)
                
                # Update metadata
                metadata['pca_applied'] = True
                metadata['pca_components'] = pc_df.columns.tolist()
                metadata['pca_explained_variance'] = pca.explained_variance_ratio_.tolist()
                metadata['pca_loading_analysis'] = pca_analysis
                metadata['n_components_used'] = n_components
                metadata['pca'] = pca
                
                # Use PCA components for feature selection
                feature_cols = pc_df.columns.tolist()
    
    # Apply feature selection - now use result directly which contains the PCA components
    selected_features = []
    
    if feature_selection_method == 'mutual_info':
        selected_features = select_features_mutual_info(
            result, 
            target_col, 
            feature_cols,
            top_n=top_n_features
        )
    elif feature_selection_method == 'f_regression':
        selected_features = select_features_f_regression(
            result, 
            target_col, 
            feature_cols,
            top_n=top_n_features
        )
    elif feature_selection_method == 'lasso':
        selected_features = select_features_lasso(
            result, 
            target_col, 
            feature_cols,
            alpha=lasso_alpha
        )
    else:
        logger.warning(f"Unknown feature selection method: {feature_selection_method}")
    
    # Update metadata
    metadata['selected_features'] = selected_features
    
    # If no features were selected but we have PCA components, use the top components
    if not selected_features and 'pca_components' in metadata:
        selected_features = metadata['pca_components'][:min(top_n_features, len(metadata['pca_components']))]
        metadata['selected_features'] = selected_features
        logger.info(f"No features selected, using top {len(selected_features)} PCA components")
    
    return result, metadata

def plot_pca_variance(pca: PCA, max_components: int = 20) -> None:
    """
    Plot PCA explained variance.
    
    Args:
        pca: Fitted PCA object
        max_components: Maximum number of components to plot
    """
    if pca is None or not hasattr(pca, 'explained_variance_ratio_'):
        logger.warning("Invalid PCA object provided")
        return
    
    # Limit number of components to plot
    n_components = min(max_components, len(pca.explained_variance_ratio_))
    
    # Calculate cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_[:n_components])
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Plot individual explained variance
    plt.bar(range(1, n_components + 1), pca.explained_variance_ratio_[:n_components], 
            alpha=0.6, label='Individual explained variance')
    
    # Plot cumulative explained variance
    plt.step(range(1, n_components + 1), cumulative_variance, where='mid', 
             label='Cumulative explained variance', color='red')
    
    # Add horizontal line at variance threshold
    plt.axhline(y=0.95, linestyle='--', color='green', label='95% Variance threshold')
    
    # Add labels and title
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('PCA Explained Variance')
    plt.xticks(range(1, n_components + 1))
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()