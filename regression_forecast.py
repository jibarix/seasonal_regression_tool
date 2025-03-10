"""
Module for running regression models with different seasonality representations.
Each model will use a mutually exclusive approach to seasonality to allow fair comparison.
"""
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='regression.log',
    filemode='w'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)
logger = logging.getLogger('regression')

def get_available_files():
    """Get list of CSV files in the exports folder."""
    export_path = Path("exports")
    if not export_path.exists():
        logger.info("Creating exports directory...")
        export_path.mkdir(exist_ok=True)
        return []
    
    csv_files = list(export_path.glob("*.csv"))
    return [file.name for file in csv_files]

def select_file(files):
    """Ask user to select a file from the list."""
    if not files:
        logger.error("No CSV files found in the exports directory.")
        return None
    
    print("\nAvailable CSV files:")
    for i, file in enumerate(files, 1):
        print(f"{i}. {file}")
    
    while True:
        try:
            choice = int(input("\nSelect a file (enter number): "))
            if 1 <= choice <= len(files):
                return files[choice - 1]
            print(f"Please enter a number between 1 and {len(files)}")
        except ValueError:
            print("Please enter a valid number")

def get_numeric_columns(df):
    """Get list of numeric columns from DataFrame."""
    return df.select_dtypes(include=['number']).columns.tolist()

def select_target_variable(columns):
    """Ask user to select a target variable."""
    print("\nAvailable numeric columns:")
    for i, col in enumerate(columns, 1):
        print(f"{i}. {col}")
    
    while True:
        try:
            choice = int(input("\nSelect target variable (enter number): "))
            if 1 <= choice <= len(columns):
                return columns[choice - 1]
            print(f"Please enter a number between 1 and {len(columns)}")
        except ValueError:
            print("Please enter a valid number")

def prepare_base_features(df, target_var, date_col='date'):
    """
    Prepare base features excluding any seasonality variables.
    This ensures the baseline feature set is the same across all models.
    
    Returns:
        DataFrame with base features only (no seasonality)
    """
    # Make a deep copy to avoid modifying the original
    df_base = df.copy()
    
    # Ensure date column is datetime
    df_base[date_col] = pd.to_datetime(df_base[date_col])
    
    # Exclude any existing seasonality columns
    seasonal_columns = [
        'year', 'month', 'quarter', 'month_name', 'period', 
        'month_end', 'quarter_end', 'Q1', 'Q2', 'Q3', 'Q4', 
        'sin_annual', 'cos_annual', 'sin_semiannual', 'cos_semiannual',
        'trend', 'trend_squared'
    ]
    
    # Also exclude columns that start with 'month_', 'quarter_', etc.
    seasonal_prefixes = ['month_', 'quarter_', 'week_', 'day_']
    
    # Find all columns to drop
    cols_to_drop = []
    for col in df_base.columns:
        if col in seasonal_columns:
            cols_to_drop.append(col)
        else:
            for prefix in seasonal_prefixes:
                if str(col).startswith(prefix):
                    cols_to_drop.append(col)
                    break
    
    # Drop seasonal columns if they exist
    existing_cols_to_drop = [col for col in cols_to_drop if col in df_base.columns]
    if existing_cols_to_drop:
        logger.info(f"Removing existing seasonality columns: {existing_cols_to_drop}")
        df_base = df_base.drop(columns=existing_cols_to_drop)
    
    # Also drop the target variable from features
    if target_var in df_base.columns:
        df_base = df_base.drop(columns=[target_var])
    
    # Ensure all columns are numeric (except date)
    # This prevents statsmodels errors related to object dtypes
    for col in df_base.columns:
        if col != date_col:
            try:
                df_base[col] = pd.to_numeric(df_base[col], errors='coerce')
            except:
                logger.warning(f"Column {col} could not be converted to numeric and will be dropped")
                df_base = df_base.drop(columns=[col])
    
    return df_base

def add_month_dummies(df_base, date_col='date'):
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
    
    # Add to dataframe without temporary columns
    result = pd.concat([df, month_dummies], axis=1)
    
    return result

def add_quarter_dummies(df_base, date_col='date'):
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
    
    # Add to dataframe without temporary columns
    result = pd.concat([df, quarter_dummies], axis=1)
    
    return result

def add_seasonal_components(df_base, date_col='date'):
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

def run_regression(X, y):
    """Run linear regression and return results."""
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()
    return model

def check_multicollinearity(X, vif_threshold=10):
    """
    Check for multicollinearity in features using VIF.
    
    Args:
        X: Feature matrix
        vif_threshold: Threshold for high VIF
        
    Returns:
        Tuple of (clean_X, removed_features)
    """
    # Start with all features
    X_clean = X.copy()
    removed_features = []
    
    # Iteratively check and remove high VIF features
    max_iter = 100  # Safety limit
    for i in range(max_iter):
        # Calculate VIF
        vif_data = calculate_vif(X_clean)
        
        # Find highest VIF
        max_vif = vif_data["VIF"].max()
        
        # If all VIFs are acceptable, break
        if max_vif < vif_threshold:
            logger.info("All features have acceptable VIF values.")
            break
        
        # Remove feature with highest VIF
        worst_feature = vif_data.loc[vif_data["VIF"].idxmax(), "Variable"]
        logger.info(f"Removing {worst_feature} due to high multicollinearity (VIF: {max_vif:.2f})")
        
        X_clean = X_clean.drop(columns=[worst_feature])
        removed_features.append(worst_feature)
        
        # If no features left, break
        if X_clean.shape[1] == 0:
            logger.warning("No features left after VIF filtering.")
            break
    
    return X_clean, removed_features

def calculate_vif(X):
    """
    Calculate Variance Inflation Factor for features.
    Includes proper handling for non-numeric data and error cases.
    """
    # Ensure X is numeric
    X_numeric = X.copy()
    for col in X_numeric.columns:
        X_numeric[col] = pd.to_numeric(X_numeric[col], errors='coerce')
    
    # Fill any NaNs with column means
    for col in X_numeric.columns:
        if X_numeric[col].isna().any():
            X_numeric[col] = X_numeric[col].fillna(X_numeric[col].mean())
    
    # Create empty dataframe to store VIF values
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X_numeric.columns
    
    # Calculate VIF for each feature
    vif_values = []
    for i in range(X_numeric.shape[1]):
        try:
            # Handle potential division by zero
            with np.errstate(divide='ignore'):
                vif = variance_inflation_factor(X_numeric.values, i)
                
                # Cap excessively large values
                if np.isinf(vif) or vif > 1e6:
                    vif = 1e6
                    
            vif_values.append(vif)
        except Exception as e:
            # If calculation fails, set to a high value
            logger.warning(f"VIF calculation failed for column {X_numeric.columns[i]}: {e}")
            vif_values.append(1e6)
    
    vif_data["VIF"] = vif_values
    
    # Sort by descending VIF
    vif_data = vif_data.sort_values("VIF", ascending=False)
    
    return vif_data

def backward_elimination(X, y, significance_level=0.05):
    """
    Run backward elimination to remove insignificant variables.
    
    Args:
        X: Feature matrix
        y: Target variable
        significance_level: Threshold p-value for significance
        
    Returns:
        Tuple of (final model, significant features)
    """
    # Check if enough samples for reliable regression
    if X.shape[0] < 5 * X.shape[1]:
        logger.warning(f"Insufficient samples-to-features ratio: {X.shape[0]} samples, {X.shape[1]} features")
        logger.warning("Results may be unreliable. Consider collecting more data or reducing features.")
    
    # Start with all features
    X_working = X.copy()
    eliminated_features = []
    
    # Run elimination process
    max_iter = 100  # Safety limit
    for i in range(max_iter):
        # Run regression
        model = run_regression(X_working, y)
        
        # Get p-values (excluding constant)
        p_values = model.pvalues.copy()
        if 'const' in p_values:
            p_values = p_values.drop('const')
        
        # Check if any features left
        if len(p_values) == 0:
            logger.warning("No features left to eliminate.")
            break
        
        # Find highest p-value
        max_p_value = p_values.max()
        
        # If all p-values are acceptable, break
        if max_p_value <= significance_level:
            logger.info("All remaining variables are significant.")
            break
        
        # Remove feature with highest p-value
        worst_feature = p_values.idxmax()
        logger.info(f"Eliminating {worst_feature} (p-value: {max_p_value:.4f})")
        
        X_working = X_working.drop(columns=[worst_feature])
        eliminated_features.append((worst_feature, max_p_value))
        
        # If no features left, break
        if X_working.shape[1] == 0:
            logger.warning("No significant features found.")
            return None, None, eliminated_features
    
    # Final model with remaining features
    final_model = run_regression(X_working, y)
    significant_features = X_working.columns.tolist()
    
    return final_model, significant_features, eliminated_features

def handle_missing_values(X, method='mean'):
    """
    Handle missing values in the feature matrix.
    Also ensures all data is numeric to prevent statsmodels errors.
    
    Args:
        X: Feature matrix
        method: 'mean' or 'drop'
        
    Returns:
        Cleaned feature matrix
    """
    # Convert all columns to numeric, coercing errors to NaN
    for col in X.columns:
        try:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        except Exception as e:
            logger.warning(f"Column {col} conversion error: {e}")
    
    # Check for columns with all NaN values
    nan_cols = X.columns[X.isna().all()]
    if not nan_cols.empty:
        logger.warning(f"Removing columns with all missing values: {list(nan_cols)}")
        X = X.drop(columns=nan_cols)
    
    # Handle remaining missing values
    if method == 'mean':
        # Impute missing values with column means
        for col in X.columns:
            missing_count = X[col].isna().sum()
            if missing_count > 0:
                mean_val = X[col].mean()
                X[col] = X[col].fillna(mean_val)
                logger.info(f"Imputed {missing_count} missing values in {col} with mean: {mean_val:.4f}")
    else:
        # Drop rows with any missing values
        missing_count = X.isna().any(axis=1).sum()
        if missing_count > 0:
            logger.info(f"Dropping {missing_count} rows with missing values")
            X = X.dropna()
    
    # Final check for any remaining non-numeric data
    for col in X.columns:
        if not np.issubdtype(X[col].dtype, np.number):
            logger.warning(f"Column {col} is still non-numeric after processing. Converting to float.")
            X[col] = X[col].astype(float)
    
    return X

def plot_actual_vs_predicted(y, y_pred, model_name, results_dir):
    """Plot actual vs predicted values."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y, y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'{model_name}: Actual vs Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{model_name}_actual_vs_predicted.png"))
    plt.close()

def plot_residuals(model, model_name, results_dir):
    """Plot residuals to check for patterns."""
    residuals = model.resid
    fitted_values = model.fittedvalues
    
    plt.figure(figsize=(10, 6))
    plt.scatter(fitted_values, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title(f'{model_name}: Residuals vs Fitted Values')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{model_name}_residuals.png"))
    plt.close()

def plot_residuals_over_time(df, residuals, date_col, model_name, results_dir):
    """
    Plot residuals over time to check for seasonal patterns.
    
    Args:
        df: Original data with dates
        residuals: Model residuals
        date_col: Name of date column
        model_name: Name of model for plot title
        results_dir: Directory to save plot
    """
    # Create DataFrame with residuals and dates
    if len(df) != len(residuals):
        logger.warning("Cannot plot residuals over time: length mismatch")
        return
    
    residuals_df = pd.DataFrame({
        'date': pd.to_datetime(df[date_col]),
        'residuals': residuals
    })
    
    # Sort by date
    residuals_df = residuals_df.sort_values('date')
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(residuals_df['date'], residuals_df['residuals'], marker='o', linestyle='-', alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(f'{model_name}: Residuals Over Time')
    plt.xlabel('Date')
    plt.ylabel('Residuals')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{model_name}_residuals_time.png"))
    plt.close()
    
    # Also generate seasonal boxplot of residuals
    plt.figure(figsize=(10, 6))
    residuals_df['month'] = residuals_df['date'].dt.month
    monthly_boxplot = plt.boxplot([residuals_df.loc[residuals_df['month'] == m, 'residuals'] 
                                   for m in range(1, 13)], labels=range(1, 13))
    plt.title(f'{model_name}: Monthly Distribution of Residuals')
    plt.xlabel('Month')
    plt.ylabel('Residuals')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{model_name}_residuals_monthly.png"))
    plt.close()

def save_model_results(model, X, y, model_name, predictions, results_dir, df_dates=None, eliminated_features=None):
    """
    Save comprehensive model results.
    
    Args:
        model: Fitted statsmodels OLS model
        X: Feature matrix for final model
        y: Target variable
        model_name: Name of model
        predictions: Model predictions
        results_dir: Directory to save results
        df_dates: Series of dates for time plot
        eliminated_features: List of eliminated features with p-values
        
    Returns:
        Dictionary with model metrics
    """
    # Calculate metrics
    rmse = np.sqrt(np.mean((y - predictions) ** 2))
    r_squared = model.rsquared
    adjusted_r_squared = model.rsquared_adj
    aic = model.aic
    bic = model.bic
    
    logger.info(f"\nModel Metrics for {model_name}:")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"R-squared: {r_squared:.4f}")
    logger.info(f"Adjusted R-squared: {adjusted_r_squared:.4f}")
    logger.info(f"AIC: {aic:.4f}")
    logger.info(f"BIC: {bic:.4f}")
    
    # Save model summary to text file
    summary_path = os.path.join(results_dir, f"{model_name}_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(str(model.summary()))
        f.write(f"\n\nModel Metrics:\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"R-squared: {r_squared:.4f}\n")
        f.write(f"Adjusted R-squared: {adjusted_r_squared:.4f}\n")
        f.write(f"AIC: {aic:.4f}\n")
        f.write(f"BIC: {bic:.4f}\n")
        
        # Add information about eliminated features
        if eliminated_features:
            f.write("\nEliminated Features (by backward elimination):\n")
            for feature, p_value in eliminated_features:
                f.write(f"{feature}: p-value = {p_value:.4f}\n")
    
    # Save results dataframe
    results_df = pd.DataFrame({
        'Actual': y,
        'Predicted': predictions,
        'Residual': y - predictions
    })
    
    # Add date column if available
    if df_dates is not None and len(df_dates) == len(results_df):
        results_df['date'] = df_dates
    
    results_path = os.path.join(results_dir, f"{model_name}_results.csv")
    results_df.to_csv(results_path, index=False)
    
    # Save model coefficients
    coef_df = pd.DataFrame({
        'Variable': ["constant"] + list(X.columns),
        'Coefficient': model.params,
        'P-value': model.pvalues,
        'Standard Error': model.bse
    })
    
    coef_path = os.path.join(results_dir, f"{model_name}_coefficients.csv")
    coef_df.to_csv(coef_path, index=False)
    
    # Generate regression equation
    equation = f"{model_name} = {model.params['const']:.4f}"
    for feature in X.columns:
        coef = model.params[feature]
        sign = "+" if coef >= 0 else ""
        equation += f" {sign} {coef:.4f} × {feature}"
    
    # Save regression equation to file
    equation_path = os.path.join(results_dir, f"{model_name}_equation.txt")
    with open(equation_path, 'w') as f:
        f.write(equation)
    
    # Save VIF for final model
    try:
        vif_data = calculate_vif(X)
        vif_path = os.path.join(results_dir, f"{model_name}_vif.csv")
        vif_data.to_csv(vif_path, index=False)
    except Exception as e:
        logger.warning(f"Could not calculate VIF for {model_name}: {e}")
    
    return {
        'model': model,
        'rmse': rmse,
        'r_squared': r_squared,
        'adj_r_squared': adjusted_r_squared,
        'aic': aic,
        'bic': bic,
        'num_vars': len(X.columns)
    }

def run_seasonal_analysis(df, target_var, date_col='date', results_dir=None):
    """
    Run regression analysis with three different seasonality representations.
    
    Args:
        df: DataFrame with the data
        target_var: Target variable for regression
        date_col: Date column name (default: 'date')
        results_dir: Directory to save results (default: None - creates a timestamped dir)
    
    Returns:
        Dictionary with results for each model type
    """
    # Create results directory if not provided
    if results_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join("results", f"regression_{timestamp}")
    
    os.makedirs(results_dir, exist_ok=True)
    logger.info(f"Results will be saved to: {results_dir}")
    
    # Save a copy of the original data
    orig_data_path = os.path.join(results_dir, "original_data.csv")
    df.to_csv(orig_data_path, index=False)
    
    # Clean data - remove rows with missing target values
    df_clean = df.dropna(subset=[target_var]).copy()
    
    # Make sure date column is datetime
    df_clean.loc[:, date_col] = pd.to_datetime(df_clean[date_col])
    
    # Sort by date
    df_clean = df_clean.sort_values(date_col).reset_index(drop=True)
    
    # Extract target variable
    y = df_clean[target_var].copy()
    
    # Prepare base features (non-seasonal)
    df_base = prepare_base_features(df_clean, target_var, date_col)
    logger.info(f"Base feature set has {df_base.shape[1]} variables")
    
    # Create datasets with different seasonality representations
    seasonal_models = {
        'no_seasonality': {
            'data': df_base.copy(),
            'description': 'Base features without any seasonality'
        },
        'month_dummies': {
            'data': add_month_dummies(df_base, date_col),
            'description': '11 monthly dummy variables (January is reference)'
        },
        'quarter_dummies': {
            'data': add_quarter_dummies(df_base, date_col),
            'description': '3 quarterly dummy variables (Q1 is reference)'
        },
        'trigonometric': {
            'data': add_seasonal_components(df_base, date_col),
            'description': 'Sine and cosine functions for annual and semi-annual cycles'
        }
    }
    
    # Run analysis for each seasonal model
    results = {}
    
    for model_type, model_info in seasonal_models.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Running analysis with {model_type}")
        logger.info(f"Description: {model_info['description']}")
        
        # Get model data
        df_model = model_info['data'].copy()
        
        # Handle missing values
        X = handle_missing_values(df_model, method='mean')
        
        # Check for and remove multicollinearity
        logger.info(f"Checking multicollinearity among {X.shape[1]} features")
        X_clean, removed_multicollinear = check_multicollinearity(X, vif_threshold=10)
        logger.info(f"Removed {len(removed_multicollinear)} features due to multicollinearity")
        logger.info(f"Continuing with {X_clean.shape[1]} features")
        
        # Run backward elimination
        elimination_result = backward_elimination(X_clean, y)
        
        if elimination_result is None or elimination_result[0] is None:
            logger.warning(f"Could not find a significant model for {model_type}")
            continue
        
        final_model, significant_features, eliminated_features = elimination_result
        
        # Use only significant features for final model
        X_final = X_clean[significant_features]
        logger.info(f"Final model has {len(significant_features)} features")
        
        # Calculate predictions
        X_final_with_const = sm.add_constant(X_final)
        predictions = final_model.predict(X_final_with_const)
        
        # Create model name
        model_name = f"{target_var}_{model_type}"
        
        # Generate plots
        plot_actual_vs_predicted(y, predictions, model_name, results_dir)
        plot_residuals(final_model, model_name, results_dir)
        plot_residuals_over_time(df_clean, final_model.resid, date_col, model_name, results_dir)
        
        # Save model results
        results[model_type] = save_model_results(
            final_model, X_final, y, model_name, predictions, results_dir, 
            df_dates=df_clean[date_col], eliminated_features=eliminated_features
        )
        
        # Save information about multicollinearity
        if removed_multicollinear:
            multicollinear_path = os.path.join(results_dir, f"{model_name}_multicollinear.txt")
            with open(multicollinear_path, 'w') as f:
                f.write("Features removed due to multicollinearity:\n")
                for feature in removed_multicollinear:
                    f.write(f"{feature}\n")
    
    # Compare models
    compare_models(results, target_var, results_dir)
    
    return results

def compare_models(results, target_var, results_dir):
    """Compare model results and create summary table."""
    if not results:
        logger.warning("No models to compare")
        return
    
    # Create comparison table
    comparison = []
    for model_type, result in results.items():
        comparison.append({
            'Model Type': model_type,
            'RMSE': result['rmse'],
            'R-squared': result['r_squared'],
            'Adj R-squared': result['adj_r_squared'],
            'AIC': result['aic'],
            'BIC': result['bic'],
            'Num Variables': result['num_vars']
        })
    
    comparison_df = pd.DataFrame(comparison)
    
    # Sort by AIC (lower is better)
    comparison_df = comparison_df.sort_values('AIC')
    
    # Save comparison table
    comparison_path = os.path.join(results_dir, f"{target_var}_model_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)
    
    # Print comparison
    logger.info("\nModel Comparison:")
    logger.info(comparison_df.to_string())
    
    # Create comparison bar chart for key metrics
    metrics = ['RMSE', 'R-squared', 'Adj R-squared']
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        bars = plt.bar(comparison_df['Model Type'], comparison_df[metric])
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom')
        
        plt.title(f'Comparison of {metric} across Models')
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"{target_var}_{metric}_comparison.png"))
        plt.close()
    
    # Print best model according to different criteria
    best_rmse = comparison_df.loc[comparison_df['RMSE'].idxmin(), 'Model Type']
    best_r2 = comparison_df.loc[comparison_df['R-squared'].idxmax(), 'Model Type']
    best_adj_r2 = comparison_df.loc[comparison_df['Adj R-squared'].idxmax(), 'Model Type']
    best_aic = comparison_df.loc[comparison_df['AIC'].idxmin(), 'Model Type']
    best_bic = comparison_df.loc[comparison_df['BIC'].idxmin(), 'Model Type']
    
    logger.info("\nBest Models by Criteria:")
    logger.info(f"Best by RMSE: {best_rmse}")
    logger.info(f"Best by R-squared: {best_r2}")
    logger.info(f"Best by Adjusted R-squared: {best_adj_r2}")
    logger.info(f"Best by AIC: {best_aic}")
    logger.info(f"Best by BIC: {best_bic}")
    
    # Generate report with model comparison and seasonal pattern analysis
    generate_comparison_report(results, comparison_df, target_var, results_dir)

def generate_comparison_report(results, comparison_df, target_var, results_dir):
    """
    Generate a comprehensive HTML report comparing different seasonal models.
    
    Args:
        results: Dictionary with model results
        comparison_df: DataFrame with model metrics
        target_var: Target variable name
        results_dir: Directory to save report
    """
    # Create HTML report
    report_path = os.path.join(results_dir, f"{target_var}_seasonal_analysis_report.html")
    
    html_content = f"""<!DOCTYPE html>
    <html>
    <head>
        <title>Seasonal Analysis Report - {target_var}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .highlight {{ background-color: #e8f4f8; font-weight: bold; }}
            .container {{ display: flex; flex-wrap: wrap; }}
            .chart {{ width: 45%; margin: 10px; border: 1px solid #ddd; padding: 10px; }}
            .full-width {{ width: 95%; }}
            .warning {{ color: #cc0000; }}
            .insight {{ background-color: #fff8e1; padding: 10px; border-left: 4px solid #ffc107; }}
        </style>
    </head>
    <body>
        <h1>Seasonal Analysis Report</h1>
        <p>Target Variable: <strong>{target_var}</strong></p>
        <p>Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Model Comparison</h2>
        <table>
            <tr>
                <th>Model Type</th>
                <th>RMSE</th>
                <th>R-squared</th>
                <th>Adj R-squared</th>
                <th>AIC</th>
                <th>BIC</th>
                <th>Variables</th>
            </tr>
    """
    
    # Add rows for each model, highlighting the best values
    best_rmse = comparison_df['RMSE'].min()
    best_r2 = comparison_df['R-squared'].max()
    best_adj_r2 = comparison_df['Adj R-squared'].max()
    best_aic = comparison_df['AIC'].min()
    best_bic = comparison_df['BIC'].min()
    
    for _, row in comparison_df.iterrows():
        html_content += "<tr>"
        html_content += f"<td>{row['Model Type']}</td>"
        
        # Highlight best values
        rmse_class = 'highlight' if row['RMSE'] == best_rmse else ''
        r2_class = 'highlight' if row['R-squared'] == best_r2 else ''
        adj_r2_class = 'highlight' if row['Adj R-squared'] == best_adj_r2 else ''
        aic_class = 'highlight' if row['AIC'] == best_aic else ''
        bic_class = 'highlight' if row['BIC'] == best_bic else ''
        
        html_content += f"<td class='{rmse_class}'>{row['RMSE']:.4f}</td>"
        html_content += f"<td class='{r2_class}'>{row['R-squared']:.4f}</td>"
        html_content += f"<td class='{adj_r2_class}'>{row['Adj R-squared']:.4f}</td>"
        html_content += f"<td class='{aic_class}'>{row['AIC']:.2f}</td>"
        html_content += f"<td class='{bic_class}'>{row['BIC']:.2f}</td>"
        html_content += f"<td>{row['Num Variables']}</td>"
        html_content += "</tr>"
    
    html_content += """
        </table>
        
        <div class="insight">
            <h3>Key Insights</h3>
            <ul>
    """
    
    # Add insights about best models
    best_model_type = comparison_df.loc[comparison_df['AIC'].idxmin(), 'Model Type']
    html_content += f"<li>The <strong>{best_model_type}</strong> model performs best according to AIC, which balances goodness of fit with model complexity.</li>"
    
    # Compare seasonal vs non-seasonal
    if 'no_seasonality' in comparison_df['Model Type'].values:
        no_season_metrics = comparison_df[comparison_df['Model Type'] == 'no_seasonality'].iloc[0]
        best_seasonal_model = comparison_df[comparison_df['Model Type'] != 'no_seasonality'].sort_values('AIC').iloc[0]
        
        aic_diff = no_season_metrics['AIC'] - best_seasonal_model['AIC']
        if aic_diff > 10:
            html_content += f"<li>Adding seasonality significantly improves model performance (AIC difference: {aic_diff:.2f}).</li>"
        elif aic_diff > 2:
            html_content += f"<li>Adding seasonality moderately improves model performance (AIC difference: {aic_diff:.2f}).</li>"
        elif aic_diff >= -2:
            html_content += f"<li>Seasonality has minimal impact on model performance (AIC difference: {aic_diff:.2f}).</li>"
        else:
            html_content += f"<li>The no-seasonality model outperforms seasonal models, suggesting seasonality may not be relevant or is captured by other variables.</li>"
    
    # Compare different seasonal approaches
    if 'month_dummies' in comparison_df['Model Type'].values and 'trigonometric' in comparison_df['Model Type'].values:
        month_metrics = comparison_df[comparison_df['Model Type'] == 'month_dummies'].iloc[0]
        trig_metrics = comparison_df[comparison_df['Model Type'] == 'trigonometric'].iloc[0]
        
        if month_metrics['AIC'] < trig_metrics['AIC'] - 4:
            html_content += "<li>Monthly dummy variables capture seasonality better than the trigonometric approach, suggesting non-sinusoidal or irregular seasonal patterns.</li>"
        elif trig_metrics['AIC'] < month_metrics['AIC'] - 4:
            html_content += "<li>The trigonometric approach captures seasonality better than monthly dummies, suggesting smooth, regular seasonal patterns.</li>"
        else:
            html_content += "<li>Monthly dummies and trigonometric approaches perform similarly, suggesting both capture the seasonal pattern adequately.</li>"
    
    # Check parsimony
    if len(comparison_df) > 1:
        most_vars = comparison_df['Num Variables'].max()
        least_vars = comparison_df['Num Variables'].min()
        
        if most_vars > 2 * least_vars:
            html_content += f"<li>Models vary significantly in complexity, from {least_vars} to {most_vars} variables.</li>"
            
            # Find parsimonious model
            parsimonious_models = comparison_df[comparison_df['Num Variables'] == least_vars]
            most_complex_models = comparison_df[comparison_df['Num Variables'] == most_vars]
            
            if parsimonious_models['AIC'].min() < most_complex_models['AIC'].min() + 4:
                most_parsimonious = parsimonious_models.sort_values('AIC').iloc[0]['Model Type']
                html_content += f"<li>The <strong>{most_parsimonious}</strong> model achieves strong performance with minimal complexity.</li>"
    
    html_content += """
            </ul>
        </div>
        
        <h2>Model Details</h2>
    """
    
    # Add details for each model
    for model_type, result in results.items():
        model = result['model']
        html_content += f"<h3>{model_type}</h3>"
        
        # Add significant variables and their coefficients
        html_content += "<table>"
        html_content += "<tr><th>Variable</th><th>Coefficient</th><th>P-value</th></tr>"
        
        # Sort by absolute coefficient value
        params = pd.Series(model.params)
        pvalues = pd.Series(model.pvalues)
        variables = pd.DataFrame({'coefficient': params, 'pvalue': pvalues})
        variables = variables.sort_values(by='coefficient', key=abs, ascending=False)
        
        # Remove constant for sorting but add it back at the top
        if 'const' in variables.index:
            const_coef = variables.loc['const', 'coefficient']
            const_pval = variables.loc['const', 'pvalue']
            variables = variables.drop('const')
            
            html_content += f"<tr><td>Constant</td><td>{const_coef:.4f}</td><td>{const_pval:.4f}</td></tr>"
        
        # Add sorted variables
        for var, row in variables.iterrows():
            html_content += f"<tr><td>{var}</td><td>{row['coefficient']:.4f}</td><td>{row['pvalue']:.4f}</td></tr>"
        
        html_content += "</table>"
        
        # Add regression equation
        html_content += "<h4>Regression Equation</h4>"
        equation = f"{target_var} = {model.params['const']:.4f}"
        for feature in model.params.index:
            if feature != 'const':
                coef = model.params[feature]
                sign = "+" if coef >= 0 else ""
                equation += f" {sign} {coef:.4f} × {feature}"
        
        html_content += f"<pre>{equation}</pre>"
    
    # Add residual plots section - just reference the images
    html_content += """
        <h2>Diagnostic Plots</h2>
        <p>See the individual model folders for the following diagnostic plots:</p>
        <ul>
            <li>Actual vs Predicted</li>
            <li>Residuals vs Fitted Values</li>
            <li>Residuals Over Time</li>
            <li>Monthly Distribution of Residuals</li>
        </ul>
        
        <h2>Conclusion</h2>
    """
    
    # Add conclusion
    if len(comparison_df) > 0:
        best_model = comparison_df.sort_values('AIC').iloc[0]['Model Type']
        html_content += f"<p>Based on AIC, the <strong>{best_model}</strong> model provides the best balance of fit and parsimony for predicting {target_var}.</p>"
    
    html_content += """
    </body>
    </html>
    """
    
    # Write HTML file
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Report generated: {report_path}")

def main():
    """Main function for seasonal analysis."""
    logger.info("Economic Data Seasonal Analysis")
    logger.info("==============================")
    
    # Get available files
    files = get_available_files()
    if not files:
        logger.error("No CSV files available. Please add data files to the exports folder.")
        return
    
    # Select file
    selected_file = select_file(files)
    if not selected_file:
        return
    
    file_path = os.path.join("exports", selected_file)
    logger.info(f"\nLoading {file_path}...")
    
    # Load data
    df = pd.read_csv(file_path)
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"seasonal_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Get numeric columns
    numeric_columns = get_numeric_columns(df)
    
    # Select target variable
    target = select_target_variable(numeric_columns)
    logger.info(f"\nSelected target variable: {target}")
    
    # Run seasonal analysis
    run_seasonal_analysis(df, target, results_dir=results_dir)
    
    logger.info(f"\nAll results saved to: {results_dir}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"An error occurred: {e}")