"""
Feature selection module for time series analysis.
Provides methods for selecting optimal features, especially after PCA.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Union, Optional, Tuple, Any
import matplotlib.pyplot as plt
from sklearn.feature_selection import (
    SelectKBest, f_regression, mutual_info_regression, RFE
)
from sklearn.linear_model import Lasso, LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings

# Local imports - try to import PCAnalyzer
try:
    from data.dimensionality_reduction import PCAnalyzer
except ImportError:
    # If not available, define a placeholder
    PCAnalyzer = Any

# Setup logging
logger = logging.getLogger(__name__)


def select_features_from_pca(pca_analyzer: PCAnalyzer, 
                           threshold: float = 0.3, 
                           top_n_components: int = 3) -> List[str]:
    """
    Select original features with highest loadings on top PCA components.
    
    Args:
        pca_analyzer: Fitted PCAnalyzer object
        threshold: Minimum absolute loading to consider important
        top_n_components: Number of top components to consider
        
    Returns:
        List of selected feature names
    """
    if not hasattr(pca_analyzer, 'is_fitted') or not pca_analyzer.is_fitted:
        raise ValueError("PCA analyzer must be fitted before selecting features")
    
    # Limit to available components
    n_components = min(top_n_components, pca_analyzer.pca.n_components_)
    
    # Get loadings
    loadings = pca_analyzer.get_loadings()
    
    # Get feature importance
    importance = pca_analyzer.get_feature_importance(n_components=n_components)
    
    # Select features that exceed threshold in at least one top component
    selected_features = set()
    
    # Method 1: Based on loading threshold
    for component in range(n_components):
        component_col = f'PC{component+1}'
        
        # Find features with loading above threshold (absolute value)
        important_features = loadings[loadings[component_col].abs() >= threshold].index.tolist()
        selected_features.update(important_features)
    
    # Method 2: Based on overall importance
    # Get top features based on cumulative importance
    importance_threshold = importance['importance'].mean() + importance['importance'].std()
    important_features = importance[importance['importance'] >= importance_threshold]['feature'].tolist()
    selected_features.update(important_features)
    
    # Convert to list and sort by original feature names
    selected_features = sorted(list(selected_features))
    
    logger.info(f"Selected {len(selected_features)} features from PCA analysis")
    return selected_features


def select_features_by_correlation(df: pd.DataFrame, 
                                 target_col: str, 
                                 threshold: float = 0.5,
                                 exclude_cols: Optional[List[str]] = None) -> List[str]:
    """
    Select features based on correlation with target variable.
    
    Args:
        df: DataFrame with features and target
        target_col: Name of target column
        threshold: Minimum absolute correlation to select feature
        exclude_cols: Columns to exclude from selection
        
    Returns:
        List of selected feature names
    """
    # Check if target column exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    
    # Determine columns to consider
    if exclude_cols is None:
        exclude_cols = []
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Exclude target and specified columns
    feature_cols = [col for col in numeric_cols if col != target_col and col not in exclude_cols]
    
    if not feature_cols:
        logger.warning("No numeric feature columns found")
        return []
    
    # Calculate correlation with target
    correlations = df[feature_cols].corrwith(df[target_col])
    
    # Select features with correlation above threshold (absolute value)
    selected_features = correlations[correlations.abs() >= threshold].index.tolist()
    
    # Sort by absolute correlation (descending)
    selected_features = sorted(selected_features, 
                             key=lambda x: abs(correlations[x]), 
                             reverse=True)
    
    logger.info(f"Selected {len(selected_features)} features with correlation >= {threshold}")
    return selected_features


def select_features_vif(df: pd.DataFrame, 
                      threshold: float = 10.0,
                      exclude_cols: Optional[List[str]] = None) -> List[str]:
    """
    Select features by removing those with high multicollinearity using VIF.
    
    Args:
        df: DataFrame with features
        threshold: Maximum VIF value to keep feature
        exclude_cols: Columns to exclude from VIF calculation
        
    Returns:
        List of selected feature names
    """
    # Determine columns to consider
    if exclude_cols is None:
        exclude_cols = []
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Exclude specified columns
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    if len(feature_cols) < 2:
        logger.warning("Need at least 2 features for VIF calculation")
        return feature_cols
    
    # Create a copy of the feature DataFrame
    X = df[feature_cols].copy()
    
    # Handle missing values
    X.fillna(X.mean(), inplace=True)
    
    # Standardize for better VIF stability
    X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
    
    # Iteratively remove features with high VIF
    selected_features = feature_cols.copy()
    max_vif = float('inf')
    
    while max_vif > threshold and len(selected_features) >= 2:
        # Calculate VIF for each feature
        vif_data = pd.DataFrame()
        vif_data['feature'] = selected_features
        
        # Calculate VIF for each feature
        vif_values = []
        for i in range(len(selected_features)):
            try:
                vif = variance_inflation_factor(X[selected_features].values, i)
                vif_values.append(vif)
            except Exception as e:
                logger.warning(f"Error calculating VIF for {selected_features[i]}: {e}")
                vif_values.append(float('inf'))
        
        vif_data['VIF'] = vif_values
        
        # Find the feature with the highest VIF
        max_vif = vif_data['VIF'].max()
        
        if max_vif > threshold:
            # Get the feature with highest VIF
            remove_feature = vif_data.loc[vif_data['VIF'].idxmax(), 'feature']
            
            # Remove the feature
            selected_features.remove(remove_feature)
            logger.debug(f"Removed feature '{remove_feature}' with VIF {max_vif:.2f}")
    
    logger.info(f"Selected {len(selected_features)} features with VIF < {threshold}")
    return selected_features


def select_features_lasso(df: pd.DataFrame,
                        target_col: str,
                        alpha: Optional[float] = None,
                        exclude_cols: Optional[List[str]] = None,
                        cv: int = 5,
                        standardize: bool = True) -> List[str]:
    """
    Select features using Lasso regression.
    
    Args:
        df: DataFrame with features and target
        target_col: Name of target column
        alpha: Lasso regularization parameter (None for CV-based selection)
        exclude_cols: Columns to exclude from selection
        cv: Number of cross-validation folds if alpha is None
        standardize: Whether to standardize features
        
    Returns:
        List of selected feature names
    """
    # Check if target column exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    
    # Determine columns to consider
    if exclude_cols is None:
        exclude_cols = []
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Exclude target and specified columns
    feature_cols = [col for col in numeric_cols if col != target_col and col not in exclude_cols]
    
    if not feature_cols:
        logger.warning("No numeric feature columns found")
        return []
    
    # Create X and y
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Handle missing values
    X.fillna(X.mean(), inplace=True)
    
    # Feature standardization
    if standardize:
        X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
    
    # Choose Lasso model based on alpha
    if alpha is None:
        # Use cross-validation to find optimal alpha
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = LassoCV(cv=cv, random_state=42)
            model.fit(X, y)
            alpha = model.alpha_
            logger.info(f"Selected optimal alpha: {alpha:.6f}")
    else:
        # Use specified alpha
        model = Lasso(alpha=alpha, random_state=42)
        model.fit(X, y)
    
    # Get feature coefficients
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': model.coef_
    })
    
    # Select features with non-zero coefficients
    selected_features = feature_importance[feature_importance['coefficient'] != 0]['feature'].tolist()
    
    # Sort by absolute coefficient (descending)
    selected_features = sorted(selected_features, 
                             key=lambda x: abs(model.coef_[feature_cols.index(x)]), 
                             reverse=True)
    
    logger.info(f"Selected {len(selected_features)} features using Lasso (alpha={alpha:.6f})")
    return selected_features


def select_features_random_forest(df: pd.DataFrame,
                                target_col: str,
                                n_estimators: int = 100,
                                threshold: float = 0.01,
                                exclude_cols: Optional[List[str]] = None) -> List[str]:
    """
    Select features using Random Forest feature importance.
    
    Args:
        df: DataFrame with features and target
        target_col: Name of target column
        n_estimators: Number of trees in the forest
        threshold: Minimum importance to select feature
        exclude_cols: Columns to exclude from selection
        
    Returns:
        List of selected feature names
    """
    # Check if target column exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    
    # Determine columns to consider
    if exclude_cols is None:
        exclude_cols = []
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Exclude target and specified columns
    feature_cols = [col for col in numeric_cols if col != target_col and col not in exclude_cols]
    
    if not feature_cols:
        logger.warning("No numeric feature columns found")
        return []
    
    # Create X and y
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Handle missing values
    X.fillna(X.mean(), inplace=True)
    
    # Train Random Forest
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X, y)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    })
    
    # Select features with importance above threshold
    selected_features = feature_importance[feature_importance['importance'] >= threshold]['feature'].tolist()
    
    # Sort by importance (descending)
    selected_features = sorted(selected_features, 
                             key=lambda x: model.feature_importances_[feature_cols.index(x)], 
                             reverse=True)
    
    logger.info(f"Selected {len(selected_features)} features using Random Forest")
    return selected_features


def select_features_from_combined_pca(df: pd.DataFrame, 
                                    target_col: str,
                                    n_components: int = 3,
                                    threshold: float = 0.01,
                                    date_col: str = 'date',
                                    exclude_cols: Optional[List[str]] = None,
                                    standardize: bool = True) -> Tuple[List[str], PCAnalyzer]:
    """
    Select features by combining PCA with target correlation.
    
    This approach:
    1. Performs PCA on features
    2. Transforms data to component space
    3. Calculates correlation of components with target
    4. Identifies which original features drive the target-correlated components
    
    Args:
        df: DataFrame with features and target
        target_col: Name of target column
        n_components: Number of principal components to consider
        threshold: Minimum importance threshold for selection
        date_col: Name of date column
        exclude_cols: Additional columns to exclude
        standardize: Whether to standardize data before PCA
        
    Returns:
        Tuple of (selected feature names, fitted PCAnalyzer)
    """
    # Import PCAnalyzer for actual use
    try:
        from data.dimensionality_reduction import PCAnalyzer
    except ImportError:
        raise ImportError("PCAnalyzer not found. Make sure dimensionality_reduction.py is available.")
    
    # Check if target column exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    
    # Prepare exclusion list
    if exclude_cols is None:
        exclude_cols = []
    
    # Always exclude target and date columns
    exclude_cols = exclude_cols + [target_col, date_col]
    exclude_cols = list(set(exclude_cols))  # Remove duplicates
    
    # Create PCA analyzer
    pca_analyzer = PCAnalyzer(
        standardize=standardize,
        n_components=n_components,
        variance_threshold=0.95
    )
    
    # Fit PCA
    pca_analyzer.fit(df, exclude_cols=exclude_cols)
    
    # Transform data to PCA space
    pca_df = pca_analyzer.transform(df)
    
    # Add target column for correlation calculation
    pca_df[target_col] = df[target_col].values
    
    # Calculate correlation of each component with target
    component_correlation = pca_df.corrwith(pca_df[target_col])
    component_correlation = component_correlation.drop(target_col)
    
    # Get loadings
    loadings = pca_analyzer.get_loadings()
    
    # Calculate weighted feature importance
    n_features = len(pca_analyzer.feature_names)
    feature_importance = np.zeros(n_features)
    
    for i, comp in enumerate(component_correlation.index):
        # Weight loadings by component correlation (absolute value)
        weight = abs(component_correlation[comp])
        for j, feature in enumerate(pca_analyzer.feature_names):
            # Use absolute loading value
            loading = abs(loadings.loc[feature, comp])
            feature_importance[j] += loading * weight
    
    # Normalize to sum to 1
    feature_importance = feature_importance / feature_importance.sum()
    
    # Create DataFrame with importance
    importance_df = pd.DataFrame({
        'feature': pca_analyzer.feature_names,
        'importance': feature_importance
    })
    
    # Sort by importance (descending)
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # Select features above threshold
    selected_features = importance_df[importance_df['importance'] >= threshold]['feature'].tolist()
    
    logger.info(f"Selected {len(selected_features)} features using PCA with target correlation")
    return selected_features, pca_analyzer


def find_optimal_feature_set(X: pd.DataFrame, 
                           y: pd.Series,
                           estimator: Any,
                           cv: Any,
                           scoring: str = 'neg_mean_squared_error',
                           feature_step: int = 1,
                           max_features: Optional[int] = None) -> Tuple[List[str], Dict[int, float]]:
    """
    Find optimal feature subset using cross-validation.
    
    Args:
        X: Feature matrix
        y: Target variable
        estimator: Sklearn-compatible estimator
        cv: Cross-validation strategy
        scoring: Scoring metric
        feature_step: Step size for feature count
        max_features: Maximum number of features to try
        
    Returns:
        Tuple of (optimal feature names, scores by feature count)
    """
    from sklearn.feature_selection import RFECV
    from sklearn.model_selection import cross_val_score
    
    # Limit max features
    if max_features is None:
        max_features = X.shape[1]
    else:
        max_features = min(max_features, X.shape[1])
    
    # Create feature selector
    selector = RFECV(
        estimator=estimator,
        step=feature_step,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )
    
    # Fit selector
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        selector.fit(X, y)
    
    # Get optimal features
    feature_names = X.columns[selector.support_].tolist()
    
    # Get scores for each number of features
    n_features_grid = np.arange(feature_step, max_features + 1, feature_step)
    scores_by_features = {}
    
    for n_features in n_features_grid:
        # Skip if we have insufficient features
        if n_features > len(feature_names):
            continue
            
        # Select top n features
        top_features = feature_names[:n_features]
        
        # Calculate cross-validation score
        scores = cross_val_score(
            estimator=estimator,
            X=X[top_features],
            y=y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )
        
        # Store average score
        scores_by_features[n_features] = scores.mean()
    
    logger.info(f"Optimal feature count: {len(feature_names)}")
    return feature_names, scores_by_features


def select_pca_economic_features(df: pd.DataFrame,
                               target_col: str,
                               date_col: str = 'date',
                               max_features: int = 10,
                               n_components: Optional[int] = None,
                               exclude_cols: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Special feature selection for economic time series based on PCA and domain knowledge.
    
    This applies a specialized workflow for economic indicators:
    1. Performs PCA to understand major drivers of variation
    2. Correlates components with target
    3. Selects diverse features representing different economic aspects
    4. Ensures representation of leading, coincident, and lagging indicators
    
    Args:
        df: DataFrame with economic indicators
        target_col: Target variable name
        date_col: Date column name
        max_features: Maximum features to select
        n_components: Number of PCA components (None for auto-selection)
        exclude_cols: Columns to exclude
        
    Returns:
        Dictionary with selected features and analysis results
    """
    # Import PCAnalyzer
    try:
        from data.dimensionality_reduction import PCAnalyzer
    except ImportError:
        raise ImportError("PCAnalyzer not found. Make sure dimensionality_reduction.py is available.")
    
    # Check if target column exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    
    # Prepare columns to exclude
    if exclude_cols is None:
        exclude_cols = []
    exclude_cols = exclude_cols + [target_col, date_col]
    exclude_cols = list(set(exclude_cols))  # Remove duplicates
    
    # Extract time-related features for domain knowledge analysis
    # Assumes columns follow pattern: category_name_metric
    # E.g., 'unemployment_rate_rate', 'housing_starts_count'
    feature_info = {}
    for col in df.columns:
        if col in exclude_cols:
            continue
            
        # Try to parse category from column name
        parts = col.split('_')
        if len(parts) >= 2:
            category = '_'.join(parts[:-1])  # Everything except last part
            metric = parts[-1]
            
            if category not in feature_info:
                feature_info[category] = []
                
            feature_info[category].append(col)
    
    # Perform PCA
    pca_analyzer = PCAnalyzer(
        standardize=True,
        n_components=n_components,
        variance_threshold=0.95
    )
    
    # Fit PCA
    pca_analyzer.fit(df, exclude_cols=exclude_cols)
    
    # Transform data to PCA space
    pca_df = pca_analyzer.transform(df)
    
    # Add target column for correlation calculation
    pca_df[target_col] = df[target_col].values
    
    # Calculate correlation of each component with target
    component_correlation = pca_df.corrwith(pca_df[target_col])
    component_correlation = component_correlation.drop(target_col)
    
    # Get top components by absolute correlation with target
    top_components = component_correlation.abs().sort_values(ascending=False).index.tolist()
    
    # Get feature contributions for each top component
    feature_scores = {}
    
    for comp_idx, comp_name in enumerate(top_components):
        # Get component index (0-based)
        comp_idx_0 = int(comp_name.replace('PC', '')) - 1
        
        # Get feature contributions to this component
        contributions = pca_analyzer.get_feature_contributions(comp_idx_0)
        
        # Weight by correlation with target (absolute value)
        weight = abs(component_correlation[comp_name])
        
        # Update feature scores
        for _, row in contributions.iterrows():
            feature = row['feature']
            contrib = row['contribution'] * weight
            
            if feature not in feature_scores:
                feature_scores[feature] = 0
                
            # Score decays with component order
            feature_scores[feature] += contrib * (0.7 ** comp_idx)
    
    # Create DataFrame with scores
    scores_df = pd.DataFrame({
        'feature': list(feature_scores.keys()),
        'score': list(feature_scores.values())
    })
    
    # Sort by score (descending)
    scores_df = scores_df.sort_values('score', ascending=False)
    
    # Ensure category diversity
    categories = {}
    for feature in scores_df['feature']:
        # Try to parse category
        parts = feature.split('_')
        if len(parts) >= 2:
            category = '_'.join(parts[:-1])
            
            if category not in categories:
                categories[category] = []
                
            categories[category].append(feature)
    
    # Strategy: Select top feature from each major category
    selected_features = []
    selected_categories = set()
    
    # First pass: Get top feature from each category
    for feature in scores_df['feature']:
        parts = feature.split('_')
        if len(parts) >= 2:
            category = '_'.join(parts[:-1])
            
            if category not in selected_categories:
                selected_features.append(feature)
                selected_categories.add(category)
                
                # Break if we've reached max features
                if len(selected_features) >= max_features:
                    break
    
    # Second pass: Fill remaining slots with top-scoring features
    if len(selected_features) < max_features:
        remaining_features = [f for f in scores_df['feature'] 
                             if f not in selected_features]
        
        # Add top remaining features up to max_features
        selected_features.extend(remaining_features[:max_features - len(selected_features)])
    
    # Create result dictionary
    result = {
        'selected_features': selected_features,
        'feature_scores': scores_df.to_dict('records'),
        'pca_analyzer': pca_analyzer,
        'component_correlation': component_correlation.to_dict(),
        'categories': categories
    }
    
    logger.info(f"Selected {len(selected_features)} economic features using PCA analysis")
    return result


def select_features_for_time_series(df: pd.DataFrame,
                                  target_col: str,
                                  date_col: str = 'date',
                                  lag_features: bool = True,
                                  max_lag: int = 3,
                                  seasonal_lag: bool = True,
                                  include_pca: bool = True,
                                  max_features: int = 10) -> List[str]:
    """
    Select features specifically for time series forecasting.
    
    This combines PCA, correlation analysis, and adds lag features.
    
    Args:
        df: DataFrame with time series data
        target_col: Target variable name
        date_col: Date column name
        lag_features: Whether to include lag features
        max_lag: Maximum lag to include
        seasonal_lag: Whether to include seasonal lag (same period last year)
        include_pca: Whether to use PCA for feature selection
        max_features: Maximum features to select
        
    Returns:
        List of selected feature names
    """
    # Check required imports
    if include_pca:
        try:
            from data.dimensionality_reduction import PCAnalyzer
        except ImportError:
            logger.warning("PCAnalyzer not found, falling back to correlation-based selection")
            include_pca = False
    
    # Create a copy of the DataFrame
    df_work = df.copy()
    
    # Exclude date column and other obvious non-feature columns
    exclude_cols = [date_col]
    if 'year' in df_work.columns:
        exclude_cols.append('year')
    if 'month' in df_work.columns:
        exclude_cols.append('month')
    if 'day' in df_work.columns:
        exclude_cols.append('day')
    
    # Add lag features if requested
    lag_cols = []
    if lag_features:
        try:
            from data.feature_factory import FeatureFactory
            
            # Create lags for target variable
            factory = FeatureFactory(date_col=date_col)
            df_work = factory.create_lags(df_work, [target_col], max_lag=max_lag)
            
            # Lag columns follow pattern: {target_col}_lag{lag}
            lag_cols = [f"{target_col}_lag{lag}" for lag in range(1, max_lag + 1)]
            
            # Add seasonal lag if requested
            if seasonal_lag:
                if date_col in df_work.columns:
                    # Get unique years
                    years = df_work[date_col].dt.year.unique()
                    
                    if len(years) >= 2:
                        # Create seasonal lag feature (same period last year)
                        df_work = factory.create_pct_change_features(df_work, [target_col], periods=[12])
                        lag_cols.append(f"{target_col}_pct12")
        except Exception as e:
            logger.warning(f"Error creating lag features: {e}")
    
    # Select features
    if include_pca:
        # Use PCA-based selection
        selection_result = select_pca_economic_features(
            df=df_work,
            target_col=target_col,
            date_col=date_col,
            max_features=max_features - len(lag_cols),  # Leave room for lag features
            exclude_cols=exclude_cols + lag_cols
        )
        
        # Get selected features
        selected_features = selection_result['selected_features']
        
    else:
        # Fall back to correlation-based selection
        selected_features = select_features_by_correlation(
            df=df_work,
            target_col=target_col,
            threshold=0.3,
            exclude_cols=exclude_cols + lag_cols
        )
        
        # Limit to max features
        selected_features = selected_features[:max_features - len(lag_cols)]
    
    # Add lag features
    if lag_features and lag_cols:
        # Check which lag columns actually exist
        existing_lag_cols = [col for col in lag_cols if col in df_work.columns]
        selected_features = existing_lag_cols + selected_features
        
        # Ensure we don't exceed max_features
        selected_features = selected_features[:max_features]
    
    logger.info(f"Selected {len(selected_features)} features for time series forecasting")
    return selected_features


def plot_feature_importances(importances: pd.DataFrame, 
                           title: str = 'Feature Importance',
                           figsize: Tuple[int, int] = (10, 6),
                           top_n: Optional[int] = None) -> plt.Figure:
    """
    Plot feature importances.
    
    Args:
        importances: DataFrame with 'feature' and 'importance' columns
        title: Plot title
        figsize: Figure size
        top_n: Number of top features to plot (None for all)
        
    Returns:
        Matplotlib figure
    """
    # Validate input
    if 'feature' not in importances.columns or 'importance' not in importances.columns:
        raise ValueError("importances DataFrame must have 'feature' and 'importance' columns")
    
    # Sort by importance
    plot_df = importances.sort_values('importance', ascending=False).copy()
    
    # Limit to top_n if specified
    if top_n is not None:
        plot_df = plot_df.head(top_n)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot horizontal bar chart
    bars = ax.barh(plot_df['feature'], plot_df['importance'], color='steelblue')
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width * 1.01
        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.3f}',
               va='center')
    
    # Add labels and title
    ax.set_xlabel('Importance')
    ax.set_title(title)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


def feature_stability_analysis(df: pd.DataFrame,
                             target_col: str,
                             feature_selector: Callable,
                             n_bootstrap: int = 20,
                             sample_fraction: float = 0.8,
                             date_col: Optional[str] = None,
                             random_state: int = 42) -> Dict[str, Any]:
    """
    Analyze feature selection stability through bootstrapping.
    
    Args:
        df: DataFrame with features and target
        target_col: Target column name
        feature_selector: Function that takes df and returns selected features
        n_bootstrap: Number of bootstrap samples
        sample_fraction: Fraction of data to use in each bootstrap sample
        date_col: Date column for time-aware sampling (None for random sampling)
        random_state: Random seed
        
    Returns:
        Dictionary with stability analysis results
    """
    # Set random seed for reproducibility
    np.random.seed(random_state)
    
    # Create a copy of the DataFrame
    df_work = df.copy()
    
    # Track selected features across bootstrap samples
    all_selections = []
    all_features = set()
    
    for i in range(n_bootstrap):
        try:
            # Create bootstrap sample
            if date_col is not None and date_col in df_work.columns:
                # Time-aware sampling
                # Sort by date
                df_sorted = df_work.sort_values(date_col)
                
                # Select contiguous time period
                n_samples = int(len(df_sorted) * sample_fraction)
                max_start = len(df_sorted) - n_samples
                
                if max_start > 0:
                    start_idx = np.random.randint(0, max_start)
                    bootstrap_sample = df_sorted.iloc[start_idx:start_idx + n_samples].copy()
                else:
                    # If dataset is too small, use random sampling instead
                    bootstrap_sample = df_sorted.sample(
                        frac=sample_fraction, 
                        random_state=random_state + i
                    )
            else:
                # Random sampling
                bootstrap_sample = df_work.sample(
                    frac=sample_fraction, 
                    random_state=random_state + i
                )
            
            # Apply feature selection
            selected_features = feature_selector(bootstrap_sample)
            
            # Store results
            all_selections.append(selected_features)
            all_features.update(selected_features)
            
        except Exception as e:
            logger.warning(f"Error in bootstrap sample {i}: {e}")
    
    # Calculate feature selection frequency
    feature_freq = {}
    for feature in all_features:
        # Count how often this feature was selected
        count = sum(1 for selection in all_selections if feature in selection)
        frequency = count / len(all_selections)
        feature_freq[feature] = frequency
    
    # Create DataFrame with frequencies
    stability_df = pd.DataFrame({
        'feature': list(feature_freq.keys()),
        'frequency': list(feature_freq.values())
    })
    
    # Sort by frequency (descending)
    stability_df = stability_df.sort_values('frequency', ascending=False)
    
    # Calculate overall stability metrics
    if len(all_selections) > 1:
        # Average Jaccard similarity between all pairs of selections
        n_comparisons = 0
        jaccard_sum = 0
        
        for i in range(len(all_selections)):
            for j in range(i+1, len(all_selections)):
                set_i = set(all_selections[i])
                set_j = set(all_selections[j])
                
                if set_i or set_j:  # Avoid division by zero
                    jaccard = len(set_i.intersection(set_j)) / len(set_i.union(set_j))
                    jaccard_sum += jaccard
                    n_comparisons += 1
        
        avg_jaccard = jaccard_sum / n_comparisons if n_comparisons > 0 else 0
        
        # Calculate average selection size
        avg_size = np.mean([len(selection) for selection in all_selections])
        
        # Calculate feature consistency (how many features were selected in all samples)
        consistent_features = [f for f, freq in feature_freq.items() if freq == 1.0]
        
        # Overall stability score (0-100)
        stability_score = avg_jaccard * 100
    else:
        avg_jaccard = 0
        avg_size = 0
        consistent_features = []
        stability_score = 0
    
    # Create result dictionary
    result = {
        'stability_df': stability_df,
        'avg_jaccard': avg_jaccard,
        'avg_selection_size': avg_size,
        'consistent_features': consistent_features,
        'stability_score': stability_score,
        'n_bootstrap': len(all_selections),
        'all_selections': all_selections
    }
    
    logger.info(f"Feature stability analysis completed: stability score {stability_score:.1f}")
    return result


def evaluate_feature_selection(df: pd.DataFrame,
                             target_col: str,
                             feature_selectors: Dict[str, Callable],
                             estimator: Any,
                             cv: Any,
                             scoring: str = 'neg_mean_squared_error',
                             date_col: Optional[str] = None) -> pd.DataFrame:
    """
    Evaluate multiple feature selection methods using cross-validation.
    
    Args:
        df: DataFrame with features and target
        target_col: Target column name
        feature_selectors: Dictionary mapping method names to feature selection functions
        estimator: Estimator to use for evaluation
        cv: Cross-validation strategy
        scoring: Scoring metric
        date_col: Date column for time series CV (None for standard CV)
        
    Returns:
        DataFrame with evaluation results
    """
    from sklearn.model_selection import cross_val_score
    
    # Check if sklearn_pandas is available for enhancing DataFrames
    try:
        from sklearn_pandas import DataFrameMapper
        has_sklearn_pandas = True
    except ImportError:
        has_sklearn_pandas = False
    
    # Create results list
    results = []
    
    # Evaluate each feature selection method
    for method_name, selector_func in feature_selectors.items():
        try:
            # Apply feature selection
            logger.info(f"Evaluating feature selection method: {method_name}")
            selected_features = selector_func(df)
            
            # Ensure we have features
            if not selected_features:
                logger.warning(f"No features selected by {method_name}")
                continue
                
            # Prepare data
            X = df[selected_features].copy()
            y = df[target_col].copy()
            
            # Handle missing values
            X = X.fillna(X.mean())
            
            # Set up cross-validation
            if date_col is not None and date_col in df.columns:
                # Use time series CV
                try:
                    from sklearn.model_selection import TimeSeriesSplit
                    
                    # Sort by date
                    sort_idx = df[date_col].argsort()
                    X = X.iloc[sort_idx].reset_index(drop=True)
                    y = y.iloc[sort_idx].reset_index(drop=True)
                    
                    # Use TimeSeriesSplit if cv is an integer
                    if isinstance(cv, int):
                        cv = TimeSeriesSplit(n_splits=cv)
                        
                except ImportError:
                    logger.warning("TimeSeriesSplit not available, using standard CV")
            
            # Perform cross-validation
            scores = cross_val_score(
                estimator=estimator,
                X=X,
                y=y,
                cv=cv,
                scoring=scoring,
                n_jobs=-1
            )
            
            # Record results
            results.append({
                'method': method_name,
                'n_features': len(selected_features),
                'features': selected_features,
                'mean_score': scores.mean(),
                'std_score': scores.std(),
                'min_score': scores.min(),
                'max_score': scores.max()
            })
            
        except Exception as e:
            logger.error(f"Error evaluating {method_name}: {e}")
    
    # Create DataFrame with results
    if results:
        results_df = pd.DataFrame(results)
        
        # Sort by score (higher is better, as sklearn multiplies metrics like MSE by -1)
        results_df = results_df.sort_values('mean_score', ascending=False)
        
        return results_df
    else:
        logger.warning("No results from feature selection evaluation")
        return pd.DataFrame()


def compare_feature_selection_over_time(df: pd.DataFrame,
                                      target_col: str,
                                      feature_selector: Callable,
                                      date_col: str = 'date',
                                      window_size: int = 52,
                                      step_size: int = 12) -> Dict[str, Any]:
    """
    Compare feature selection stability over time using a rolling window approach.
    
    Args:
        df: DataFrame with features and target
        target_col: Target column name
        feature_selector: Function that takes df and returns selected features
        date_col: Date column name
        window_size: Size of rolling window
        step_size: Step size for rolling window
        
    Returns:
        Dictionary with time-based stability analysis
    """
    # Check if date column exists
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in DataFrame")
    
    # Sort by date
    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    
    # Create rolling windows
    windows = []
    for start in range(0, len(df_sorted) - window_size + 1, step_size):
        end = start + window_size
        windows.append((start, end))
    
    # Track selected features for each window
    window_selections = []
    window_info = []
    all_features = set()
    
    for i, (start, end) in enumerate(windows):
        window_df = df_sorted.iloc[start:end].copy()
        
        try:
            # Apply feature selection
            selected_features = feature_selector(window_df)
            
            # Store results
            window_selections.append(selected_features)
            window_info.append({
                'window_id': i,
                'start_idx': start,
                'end_idx': end,
                'start_date': window_df[date_col].iloc[0],
                'end_date': window_df[date_col].iloc[-1],
                'n_features': len(selected_features),
                'features': selected_features
            })
            
            all_features.update(selected_features)
            
        except Exception as e:
            logger.warning(f"Error in window {i}: {e}")
    
    # Calculate feature selection frequency over time
    feature_freq_over_time = {}
    for feature in all_features:
        # For each window, check if feature was selected
        frequencies = []
        for selection in window_selections:
            frequencies.append(1 if feature in selection else 0)
            
        feature_freq_over_time[feature] = frequencies
    
    # Calculate Jaccard similarity between consecutive windows
    jaccard_similarities = []
    
    for i in range(1, len(window_selections)):
        prev_set = set(window_selections[i-1])
        curr_set = set(window_selections[i])
        
        if prev_set or curr_set:  # Avoid division by zero
            jaccard = len(prev_set.intersection(curr_set)) / len(prev_set.union(curr_set))
        else:
            jaccard = 1.0  # Both empty = perfect similarity
            
        jaccard_similarities.append({
            'window_id': i,
            'prev_window': i-1,
            'jaccard': jaccard,
            'start_date': window_info[i]['start_date'],
            'end_date': window_info[i]['end_date']
        })
    
    # Create feature frequency DataFrame
    freq_df = pd.DataFrame(feature_freq_over_time)
    
    # Add window dates
    freq_df['start_date'] = [info['start_date'] for info in window_info]
    freq_df['end_date'] = [info['end_date'] for info in window_info]
    
    # Create result dictionary
    result = {
        'window_info': window_info,
        'feature_freq_over_time': freq_df,
        'jaccard_similarities': jaccard_similarities,
        'overall_stability': np.mean([j['jaccard'] for j in jaccard_similarities]) if jaccard_similarities else 0,
        'all_features': list(all_features)
    }
    
    logger.info(f"Feature selection stability over time analyzed across {len(windows)} windows")
    return result


# Feature Selection Factory

class FeatureSelector:
    """
    Factory class for feature selection methods.
    Provides a consistent interface for different selection approaches.
    """
    
    def __init__(self, df: pd.DataFrame, 
                target_col: str,
                date_col: str = 'date',
                exclude_cols: Optional[List[str]] = None):
        """
        Initialize the feature selector.
        
        Args:
            df: DataFrame with features and target
            target_col: Target column name
            date_col: Date column name
            exclude_cols: Columns to exclude from selection
        """
        self.df = df.copy()
        self.target_col = target_col
        self.date_col = date_col
        self.exclude_cols = exclude_cols or []
        
        # Always exclude date and target columns
        if date_col not in self.exclude_cols:
            self.exclude_cols.append(date_col)
        if target_col not in self.exclude_cols:
            self.exclude_cols.append(target_col)
            
        # Get available numeric columns
        self.numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        self.feature_cols = [col for col in self.numeric_cols 
                           if col not in self.exclude_cols]
        
        logger.info(f"FeatureSelector initialized with {len(self.feature_cols)} potential features")
    
    def by_correlation(self, threshold: float = 0.3, abs_corr: bool = True) -> List[str]:
        """
        Select features by correlation with target.
        
        Args:
            threshold: Minimum correlation threshold
            abs_corr: Whether to use absolute correlation
            
        Returns:
            List of selected feature names
        """
        # Calculate correlation with target
        correlations = self.df[self.feature_cols].corrwith(self.df[self.target_col])
        
        if abs_corr:
            # Use absolute correlation
            mask = correlations.abs() >= threshold
        else:
            # Use raw correlation (direction matters)
            mask = correlations >= threshold
            
        # Get selected features
        selected_features = correlations[mask].index.tolist()
        
        # Sort by absolute correlation (descending)
        selected_features = sorted(selected_features, 
                                key=lambda x: abs(correlations[x]), 
                                reverse=True)
        
        logger.info(f"Selected {len(selected_features)} features by correlation (threshold: {threshold})")
        return selected_features
    
    def by_vif(self, threshold: float = 10.0) -> List[str]:
        """
        Select features by removing those with high VIF.
        
        Args:
            threshold: Maximum VIF threshold
            
        Returns:
            List of selected feature names
        """
        return select_features_vif(self.df, threshold, self.exclude_cols)
    
    def by_lasso(self, alpha: Optional[float] = None, cv: int = 5) -> List[str]:
        """
        Select features using Lasso regression.
        
        Args:
            alpha: Lasso regularization parameter (None for CV-based selection)
            cv: Number of cross-validation folds if alpha is None
            
        Returns:
            List of selected feature names
        """
        return select_features_lasso(self.df, self.target_col, alpha, self.exclude_cols, cv)
    
    def by_random_forest(self, n_estimators: int = 100, 
                       threshold: float = 0.01) -> List[str]:
        """
        Select features using Random Forest importance.
        
        Args:
            n_estimators: Number of trees
            threshold: Minimum importance threshold
            
        Returns:
            List of selected feature names
        """
        return select_features_random_forest(
            self.df, self.target_col, n_estimators, threshold, self.exclude_cols
        )
    
    def by_pca(self, n_components: int = 3, threshold: float = 0.3) -> List[str]:
        """
        Select features based on PCA loadings.
        
        Args:
            n_components: Number of components to consider
            threshold: Minimum loading threshold
            
        Returns:
            List of selected feature names
        """
        try:
            # Import PCAnalyzer
            from data.dimensionality_reduction import PCAnalyzer
            
            # Create analyzer
            analyzer = PCAnalyzer(
                standardize=True,
                n_components=n_components
            )
            
            # Fit PCA
            analyzer.fit(self.df, exclude_cols=self.exclude_cols)
            
            # Select features
            return select_features_from_pca(analyzer, threshold, n_components)
            
        except ImportError:
            logger.warning("PCAnalyzer not available, falling back to correlation-based selection")
            return self.by_correlation(threshold)
    
    def by_pca_with_target(self, n_components: int = 3, 
                         threshold: float = 0.01) -> List[str]:
        """
        Select features by combining PCA with target correlation.
        
        Args:
            n_components: Number of components
            threshold: Minimum importance threshold
            
        Returns:
            List of selected feature names
        """
        try:
            # Use PCA with target correlation
            selected_features, _ = select_features_from_combined_pca(
                self.df, self.target_col, n_components, threshold,
                self.date_col, self.exclude_cols
            )
            return selected_features
            
        except ImportError:
            logger.warning("PCAnalyzer not available, falling back to correlation-based selection")
            return self.by_correlation(threshold)
    
    def for_time_series(self, lag_features: bool = True, max_lag: int = 3,
                      seasonal_lag: bool = True, include_pca: bool = True,
                      max_features: int = 10) -> List[str]:
        """
        Select features specifically for time series forecasting.
        
        Args:
            lag_features: Whether to include lag features
            max_lag: Maximum lag to include
            seasonal_lag: Whether to include seasonal lag
            include_pca: Whether to use PCA
            max_features: Maximum features to select
            
        Returns:
            List of selected feature names
        """
        return select_features_for_time_series(
            self.df, self.target_col, self.date_col,
            lag_features, max_lag, seasonal_lag, include_pca, max_features
        )
    
    def for_economic_indicators(self, max_features: int = 10,
                             n_components: Optional[int] = None) -> List[str]:
        """
        Select features for economic indicators using PCA-based approach.
        
        Args:
            max_features: Maximum features to select
            n_components: Number of PCA components
            
        Returns:
            List of selected feature names
        """
        try:
            # Use economic-specific selection
            selection_result = select_pca_economic_features(
                self.df, self.target_col, self.date_col,
                max_features, n_components, self.exclude_cols
            )
            return selection_result['selected_features']
            
        except ImportError:
            logger.warning("PCAnalyzer not available, falling back to correlation-based selection")
            return self.by_correlation(threshold=0.3)[:max_features]
    
    def evaluate_methods(self, estimator: Any, cv: Any = 5,
                        scoring: str = 'neg_mean_squared_error') -> pd.DataFrame:
        """
        Evaluate multiple feature selection methods.
        
        Args:
            estimator: Estimator to use for evaluation
            cv: Cross-validation strategy
            scoring: Scoring metric
            
        Returns:
            DataFrame with evaluation results
        """
        # Define feature selectors
        feature_selectors = {
            'correlation': lambda df: self.by_correlation(threshold=0.3),
            'vif': lambda df: self.by_vif(threshold=10.0),
            'lasso': lambda df: self.by_lasso(),
            'random_forest': lambda df: self.by_random_forest(),
            'pca': lambda df: self.by_pca(),
            'pca_with_target': lambda df: self.by_pca_with_target(),
            'time_series': lambda df: self.for_time_series()
        }
        
        # Evaluate methods
        return evaluate_feature_selection(
            self.df, self.target_col, feature_selectors,
            estimator, cv, scoring, self.date_col
        )
    
    def find_best_method(self, estimator: Any, cv: Any = 5,
                       scoring: str = 'neg_mean_squared_error') -> Tuple[str, List[str]]:
        """
        Find the best feature selection method based on cross-validation.
        
        Args:
            estimator: Estimator to use for evaluation
            cv: Cross-validation strategy
            scoring: Scoring metric
            
        Returns:
            Tuple of (best method name, selected features)
        """
        # Evaluate methods
        results = self.evaluate_methods(estimator, cv, scoring)
        
        if results.empty:
            logger.warning("No results from method evaluation")
            return 'correlation', self.by_correlation()
        
        # Get best method
        best_method = results.iloc[0]['method']
        best_features = results.iloc[0]['features']
        
        logger.info(f"Best method: {best_method} with score {results.iloc[0]['mean_score']:.4f}")
        return best_method, best_features