"""
Visualization module for time series analysis.
Provides functions for plotting time series data, model results,
feature importance, and PCA components.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Union, Optional, Tuple, Any
import logging
from sklearn.decomposition import PCA
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator

# Setup logging
logger = logging.getLogger(__name__)

def set_plotting_style():
    """Set consistent style for all plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("notebook", font_scale=1.1)
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    
def plot_time_series(df: pd.DataFrame, 
                    columns: List[str], 
                    date_col: str = 'date',
                    title: str = 'Time Series Plot',
                    save_path: Optional[str] = None):
    """
    Plot time series data.
    
    Args:
        df: DataFrame with time series data
        columns: List of columns to plot
        date_col: Name of date column
        title: Plot title
        save_path: Optional path to save the plot
    """
    set_plotting_style()
    
    if date_col not in df.columns:
        logger.warning(f"Date column '{date_col}' not found in DataFrame")
        return
    
    plt.figure(figsize=(12, 6))
    
    dates = pd.to_datetime(df[date_col])
    
    for col in columns:
        if col in df.columns:
            plt.plot(dates, df[col], marker='.', markersize=3, label=col)
        else:
            logger.warning(f"Column '{col}' not found in DataFrame")
    
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    
    # Format x-axis date labels
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_seasonal_decomposition(df: pd.DataFrame, 
                               column: str, 
                               date_col: str = 'date',
                               frequency: int = 12,  # Default for monthly data
                               model: str = 'additive',
                               save_path: Optional[str] = None):
    """
    Plot seasonal decomposition of time series.
    
    Args:
        df: DataFrame with time series data
        column: Column to decompose
        date_col: Name of date column
        frequency: Frequency of the time series
        model: Type of seasonal model ('additive' or 'multiplicative')
        save_path: Optional path to save the plot
    """
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
    except ImportError:
        logger.error("statsmodels is required for seasonal decomposition")
        return
    
    set_plotting_style()
    
    if column not in df.columns:
        logger.warning(f"Column '{column}' not found in DataFrame")
        return
    
    if date_col not in df.columns:
        logger.warning(f"Date column '{date_col}' not found in DataFrame")
        return
    
    # Set datetime index
    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.set_index(date_col)
    
    # Handle missing values
    data[column] = data[column].fillna(data[column].interpolate())
    
    # Perform decomposition
    decomposition = seasonal_decompose(data[column], model=model, period=frequency)
    
    # Plot decomposition
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    
    # Original
    decomposition.observed.plot(ax=ax1, title='Original')
    ax1.set_ylabel('Value')
    
    # Trend
    decomposition.trend.plot(ax=ax2, title='Trend')
    ax2.set_ylabel('Value')
    
    # Seasonal
    decomposition.seasonal.plot(ax=ax3, title='Seasonality')
    ax3.set_ylabel('Value')
    
    # Residual
    decomposition.resid.plot(ax=ax4, title='Residuals')
    ax4.set_ylabel('Value')
    
    # Format dates and layout
    plt.xlabel('Date')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.suptitle(f'Seasonal Decomposition of {column} (Model: {model})', y=1.02, fontsize=16)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_correlation_matrix(df: pd.DataFrame, 
                           columns: Optional[List[str]] = None,
                           title: str = 'Feature Correlation Matrix',
                           save_path: Optional[str] = None):
    """
    Plot correlation matrix of features.
    
    Args:
        df: DataFrame with features
        columns: Specific columns to include (None for all numeric)
        title: Plot title
        save_path: Optional path to save the plot
    """
    set_plotting_style()
    
    # Select columns
    if columns is None:
        # Use all numeric columns
        data = df.select_dtypes(include=['number'])
    else:
        # Filter to specified columns that exist
        valid_cols = [col for col in columns if col in df.columns]
        if not valid_cols:
            logger.warning("None of the specified columns found in DataFrame")
            return
        data = df[valid_cols]
    
    # Calculate correlation matrix
    corr_matrix = data.corr()
    
    # Plot correlation matrix
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, annot=False, cbar_kws={"shrink": .5})
    
    plt.title(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_feature_importance(importances: pd.Series, 
                           title: str = 'Feature Importance',
                           top_n: int = 20,
                           save_path: Optional[str] = None):
    """
    Plot feature importance from model.
    
    Args:
        importances: Series with feature importances
        title: Plot title
        top_n: Number of top features to show
        save_path: Optional path to save the plot
    """
    set_plotting_style()
    
    # Handle empty input
    if importances.empty:
        logger.warning("Empty feature importance Series")
        return
    
    # Sort importances and take top N
    if len(importances) > top_n:
        top_importances = importances.sort_values(ascending=False).head(top_n)
    else:
        top_importances = importances.sort_values(ascending=False)
    
    # Create horizontal bar plot
    plt.figure(figsize=(10, max(6, len(top_importances) * 0.4)))
    ax = top_importances.plot(kind='barh', color='steelblue')
    
    # Add value labels
    for i, v in enumerate(top_importances):
        ax.text(v, i, f'{v:.4f}', va='center', fontsize=9)
    
    plt.title(title, fontsize=14)
    plt.xlabel('Importance')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_pca_components(pca: PCA, 
                       feature_names: List[str],
                       n_components: int = 2,
                       n_features: int = 10,
                       save_path: Optional[str] = None):
    """
    Plot PCA component loadings.
    
    Args:
        pca: Fitted PCA object
        feature_names: Names of original features
        n_components: Number of components to plot
        n_features: Number of top features to show per component
        save_path: Optional path to save the plot
    """
    set_plotting_style()
    
    if pca is None or not hasattr(pca, 'components_'):
        logger.warning("Invalid PCA object provided")
        return
    
    # Limit number of components to plot
    n_components = min(n_components, pca.n_components_)
    
    # Prepare figure with multiple subplots
    fig, axes = plt.subplots(n_components, 1, figsize=(10, 4 * n_components), squeeze=False)
    
    # Plot each component
    for i in range(n_components):
        # Get component loadings
        loadings = pca.components_[i]
        
        # Get indices of top features by magnitude
        top_indices = np.abs(loadings).argsort()[-n_features:][::-1]
        
        # Get feature names and loadings
        top_features = [feature_names[idx] for idx in top_indices]
        top_loadings = loadings[top_indices]
        
        # Create bar plot
        ax = axes[i, 0]
        bars = ax.barh(top_features, top_loadings, color=['red' if x < 0 else 'blue' for x in top_loadings])
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            label_pos = width - 0.1 if width < 0 else width + 0.1
            ax.text(label_pos, bar.get_y() + bar.get_height()/2, f'{width:.2f}',
                   va='center', ha='right' if width < 0 else 'left', fontsize=8)
        
        # Add labels and title
        ax.set_xlabel('Loading Value')
        ax.set_title(f'PC{i+1} ({pca.explained_variance_ratio_[i]:.2%} variance explained)')
        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.7)
    
    plt.suptitle('PCA Component Loadings', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_model_predictions(y_true: pd.Series, 
                          y_pred: np.ndarray, 
                          date_index: Optional[pd.DatetimeIndex] = None,
                          test_start_idx: Optional[int] = None,
                          title: str = 'Model Predictions vs Actual',
                          save_path: Optional[str] = None):
    """
    Plot model predictions against actual values with train/test split.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values (can be shorter than y_true if only test set predictions)
        date_index: Optional DatetimeIndex for time-based plotting
        test_start_idx: Index where test set begins (for vertical line)
        title: Plot title
        save_path: Optional path to save the plot
    """
    set_plotting_style()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # If we have a time index, use it
    if date_index is not None:
        x_axis = date_index
        xlabel = 'Date'
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        if len(date_index) > 60:  # If we have many points
            ax.xaxis.set_major_locator(mdates.YearLocator())
        plt.xticks(rotation=45)
    else:
        # Otherwise use numeric index
        x_axis = np.arange(len(y_true))
        xlabel = 'Index'
    
    # Plot actual values
    ax.plot(x_axis, y_true, 'b-', label='Actual', alpha=0.7)
    
    # Plot predictions - handle case where predictions might be only for test set
    if len(y_pred) < len(y_true) and test_start_idx is not None:
        # We only have predictions for test set
        pred_x = x_axis[test_start_idx:test_start_idx + len(y_pred)]
        ax.plot(pred_x, y_pred, 'r-', label='Predicted', alpha=0.7)
    else:
        # We have predictions for the entire dataset
        ax.plot(x_axis[:len(y_pred)], y_pred, 'r-', label='Predicted', alpha=0.7)
    
    # Add vertical line for train/test split if specified
    if test_start_idx is not None:
        if date_index is not None:
            split_point = date_index[test_start_idx]
        else:
            split_point = test_start_idx
            
        ax.axvline(x=split_point, color='green', linestyle='--', label='Train/Test Split')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_forecast_error_distribution(y_true: pd.Series, 
                                   y_pred: np.ndarray,
                                   title: str = 'Forecast Error Distribution',
                                   save_path: Optional[str] = None):
    """
    Plot distribution of forecast errors.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        title: Plot title
        save_path: Optional path to save the plot
    """
    set_plotting_style()
    
    # Calculate errors
    errors = y_true - y_pred
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Histogram of errors
    sns.histplot(errors, kde=True, color='steelblue')
    
    # Add mean and zero lines
    plt.axvline(x=errors.mean(), color='red', linestyle='-', label=f'Mean Error: {errors.mean():.4f}')
    plt.axvline(x=0, color='green', linestyle='--', label='Zero Error')
    
    # Add labels
    plt.xlabel('Forecast Error')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_rolling_window_forecast(y_true: pd.Series,
                                predictions: List[np.ndarray],
                                window_size: int,
                                step_size: int = 1,
                                date_index: Optional[pd.DatetimeIndex] = None,
                                title: str = 'Rolling Window Forecast',
                                save_path: Optional[str] = None):
    """
    Plot rolling window forecast with multiple prediction windows.
    
    Args:
        y_true: True target values
        predictions: List of prediction arrays for each window
        window_size: Size of each forecast window
        step_size: Steps between consecutive windows
        date_index: Optional DatetimeIndex for time-based plotting
        title: Plot title
        save_path: Optional path to save the plot
    """
    set_plotting_style()
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot actual values
    if date_index is not None:
        plt.plot(date_index, y_true, 'k-', linewidth=2, label='Actual', alpha=0.7)
        x_label = 'Date'
        
        # Format x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        plt.xticks(rotation=45)
    else:
        plt.plot(y_true.index, y_true, 'k-', linewidth=2, label='Actual', alpha=0.7)
        x_label = 'Index'
    
    # Plot each forecast window
    colors = plt.cm.tab10.colors
    for i, window_pred in enumerate(predictions):
        start_idx = i * step_size
        end_idx = start_idx + len(window_pred)
        
        if date_index is not None:
            window_x = date_index[start_idx:end_idx]
        else:
            window_x = y_true.index[start_idx:end_idx]
        
        # Only label first few windows to avoid cluttering legend
        if i < 5:
            plt.plot(window_x, window_pred, 'o-', color=colors[i % len(colors)], 
                    alpha=0.5, linewidth=1, label=f'Window {i+1}')
        else:
            plt.plot(window_x, window_pred, 'o-', color=colors[i % len(colors)], 
                    alpha=0.5, linewidth=1)
    
    plt.xlabel(x_label)
    plt.ylabel('Value')
    plt.title(title)
    plt.legend(loc='best')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def create_diagnostic_plots(df: pd.DataFrame,
                           target_col: str,
                           date_col: str = 'date',
                           save_dir: Optional[str] = None):
    """
    Create a set of diagnostic plots for exploratory data analysis.
    
    Args:
        df: DataFrame with time series data
        target_col: Name of target column
        date_col: Name of date column
        save_dir: Optional directory to save plots
    """
    # Ensure we have required columns
    if target_col not in df.columns:
        logger.warning(f"Target column '{target_col}' not found in DataFrame")
        return
    
    if date_col not in df.columns:
        logger.warning(f"Date column '{date_col}' not found in DataFrame")
        return
    
    # Convert date column to datetime
    df[date_col] = pd.to_datetime(df[date_col])
    
    # 1. Time series plot of target variable
    plot_time_series(
        df, [target_col], date_col=date_col,
        title=f'Time Series of {target_col}',
        save_path=f'{save_dir}/time_series_{target_col}.png' if save_dir else None
    )
    
    # 2. Seasonal decomposition if we have enough data
    if len(df) >= 24:  # Need at least 2 periods for monthly data
        try:
            plot_seasonal_decomposition(
                df, target_col, date_col=date_col,
                save_path=f'{save_dir}/seasonal_decomposition_{target_col}.png' if save_dir else None
            )
        except Exception as e:
            logger.warning(f"Could not create seasonal decomposition: {e}")
    
    # 3. Correlation matrix of numeric variables
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if len(numeric_cols) > 1:
        plot_correlation_matrix(
            df, numeric_cols,
            title='Feature Correlation Matrix',
            save_path=f'{save_dir}/correlation_matrix.png' if save_dir else None
        )
    
    # 4. Distribution of target variable
    plt.figure(figsize=(10, 6))
    sns.histplot(df[target_col].dropna(), kde=True, color='steelblue')
    plt.title(f'Distribution of {target_col}')
    plt.xlabel(target_col)
    plt.tight_layout()
    if save_dir:
        plt.savefig(f'{save_dir}/distribution_{target_col}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. Box plot by year if we have multiple years
    try:
        df['year'] = df[date_col].dt.year
        years = df['year'].unique()
        
        if len(years) > 1:
            plt.figure(figsize=(12, 6))
            sns.boxplot(x='year', y=target_col, data=df, color='steelblue')
            plt.title(f'{target_col} by Year')
            plt.xticks(rotation=45)
            plt.tight_layout()
            if save_dir:
                plt.savefig(f'{save_dir}/boxplot_year_{target_col}.png', dpi=300, bbox_inches='tight')
            plt.show()
    except Exception as e:
        logger.warning(f"Could not create year box plot: {e}")
    
    # 6. Monthly pattern if we have monthly data
    try:
        df['month'] = df[date_col].dt.month
        
        # Create month name labels
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='month', y=target_col, data=df, color='steelblue')
        plt.title(f'Monthly Pattern of {target_col}')
        plt.xticks(range(12), month_names, rotation=45)
        plt.tight_layout()
        if save_dir:
            plt.savefig(f'{save_dir}/monthly_pattern_{target_col}.png', dpi=300, bbox_inches='tight')
        plt.show()
    except Exception as e:
        logger.warning(f"Could not create monthly pattern plot: {e}")