"""
Main module for time series analysis and forecasting pipeline.
Orchestrates data loading, feature engineering, dimension reduction, 
and modeling to fulfill project objectives.
"""
import os
import pandas as pd
import numpy as np
import argparse
import logging
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict, Union, Optional, Tuple

# Import project modules
from data.processor import DataLoader
from data.transformations import TimeSeriesTransformer
from data.feature_engineering import engineer_features
from data.dimension_reduction import reduce_dimensions, plot_pca_variance
from models.modeling import train_and_evaluate, analyze_residuals, plot_residuals, plot_actual_vs_predicted

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ts_forecast.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Time Series Forecasting Pipeline')
    
    # Data arguments
    parser.add_argument('--data-file', type=str, required=True, 
                        help='Path to the input data CSV file')
    parser.add_argument('--target-col', type=str, required=True,
                        help='Name of the target column to forecast')
    parser.add_argument('--date-col', type=str, default='date',
                        help='Name of the date column')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save results')
                        
    # Feature engineering arguments
    parser.add_argument('--fourier-periods', type=str, default='12,4',
                        help='Comma-separated list of Fourier periods')
    parser.add_argument('--fourier-harmonics', type=int, default=2,
                        help='Number of Fourier harmonics')
    parser.add_argument('--max-lag', type=int, default=3,  # Reduced from 6 to 3
                        help='Maximum lag for lagged features')
    parser.add_argument('--use-log', action='store_true',
                        help='Apply log transformation to features')
    parser.add_argument('--respect-existing-features', action='store_true', default=True,
                        help='Respect existing date features in the dataset')
                        
    # Dimension reduction arguments
    parser.add_argument('--use-pca', action='store_true',
                        help='Use PCA for dimension reduction')
    parser.add_argument('--pca-variance', type=float, default=0.95,
                        help='PCA cumulative variance threshold')
    parser.add_argument('--feature-selection', type=str, default='mutual_info',
                        choices=['mutual_info', 'f_regression', 'lasso'],
                        help='Feature selection method')
    parser.add_argument('--top-n-features', type=int, default=20,  # Increased from 10 to 20
                        help='Number of top features to select')
                        
    # Modeling arguments
    parser.add_argument('--model-type', type=str, default='randomforest',  # Changed default from linear to randomforest
                        choices=['linear', 'ridge', 'lasso', 'elasticnet', 'randomforest', 'gbm'],
                        help='Type of model to train')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of data to use for testing')
    parser.add_argument('--cv-splits', type=int, default=5,
                        help='Number of splits for time series cross-validation')
    parser.add_argument('--no-plots', action='store_true',
                        help='Disable plotting of results')
                        
    return parser.parse_args()

def load_and_prepare_data(args):
    """Load and prepare data for modeling."""
    logger.info(f"Loading data from {args.data_file}")
    
    # Initialize data loader
    data_loader = DataLoader(date_col=args.date_col)
    
    # Load data
    df = data_loader.load_data(args.data_file)
    
    # Check if target column exists
    if args.target_col not in df.columns:
        logger.error(f"Target column '{args.target_col}' not found in data")
        raise ValueError(f"Target column '{args.target_col}' not found in data")
    
    # Detect data frequency
    frequency = data_loader.detect_frequency(df)
    logger.info(f"Detected data frequency: {frequency}")
    
    # Handle missing values
    if frequency in ['monthly', 'quarterly']:
        fill_groups = ['year', 'month' if frequency == 'monthly' else 'quarter']
        df = data_loader.handle_missing_values(df, method='mean', fill_groups=fill_groups)
    else:
        df = data_loader.handle_missing_values(df, method='mean')
    
    # Add time variables if not already present
    if 'year' not in df.columns or 'month' not in df.columns:
        df = data_loader.add_time_variables(df)
    
    return df, frequency

def run_pipeline(args):
    """Run the complete analysis pipeline."""
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and prepare data
    df, frequency = load_and_prepare_data(args)
    
    # Log basic data info
    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Target column: {args.target_col}")
    
    # Save prepared data
    prepared_data_path = os.path.join(args.output_dir, 'prepared_data.csv')
    df.to_csv(prepared_data_path, index=False)
    logger.info(f"Saved prepared data to {prepared_data_path}")
    
    # Engineer features
    logger.info("Engineering features...")
    
    # Parse Fourier periods
    fourier_periods = [int(p) for p in args.fourier_periods.split(',')]
    
    # Engineer features
    df_features = engineer_features(
        df, 
        target_col=args.target_col,
        date_col=args.date_col,
        create_lags=True,
        create_fourier=True,
        create_rolling=True,
        log_transform=args.use_log,
        max_lag=args.max_lag
    )
    
    # Save feature-engineered data
    features_path = os.path.join(args.output_dir, 'engineered_features.csv')
    df_features.to_csv(features_path, index=False)
    logger.info(f"Saved engineered features to {features_path}")
    logger.info(f"Feature-engineered data shape: {df_features.shape}")
    
    # Reduce dimensions
    logger.info("Reducing dimensions...")
    
    # Get all numeric columns except date and target
    all_features = df_features.select_dtypes(include=['number']).columns.tolist()
    if args.date_col in all_features:
        all_features.remove(args.date_col)
    if args.target_col in all_features:
        all_features.remove(args.target_col)
    
    # Drop rows with NaN values (from lagging)
    df_features = df_features.dropna()
    logger.info(f"Data shape after dropping NaNs: {df_features.shape}")
    
    # Reduce dimensions
    df_reduced, dim_reduction_metadata = reduce_dimensions(
        df_features,
        target_col=args.target_col,
        feature_cols=all_features,
        date_col=args.date_col,
        use_pca=args.use_pca,
        pca_variance_threshold=args.pca_variance,
        feature_selection_method=args.feature_selection,
        top_n_features=args.top_n_features
    )
    
    # Save reduced data
    reduced_path = os.path.join(args.output_dir, 'reduced_data.csv')
    df_reduced.to_csv(reduced_path, index=False)
    logger.info(f"Saved reduced data to {reduced_path}")
    logger.info(f"Reduced data shape: {df_reduced.shape}")
    
    # Plot PCA variance if PCA was used
    if args.use_pca and 'pca_applied' in dim_reduction_metadata and dim_reduction_metadata['pca_applied']:
        if not args.no_plots and 'pca' in locals():
            plot_pca_variance(dim_reduction_metadata.get('pca'))
    
    # Train and evaluate model
    logger.info(f"Training {args.model_type} model...")
    
    # Get selected features
    selected_features = dim_reduction_metadata.get('selected_features', [])
    
    # If PCA was applied, use PCA components
    if args.use_pca and 'pca_applied' in dim_reduction_metadata and dim_reduction_metadata['pca_applied']:
        if 'pca_components' in dim_reduction_metadata:
            selected_features = dim_reduction_metadata['pca_components']
    
    # If no features were selected, use all available numeric features
    if not selected_features:
        selected_features = [col for col in df_reduced.columns 
                           if col != args.target_col and col != args.date_col]
    
    logger.info(f"Selected features: {selected_features}")
    
    # Set model parameters based on model type
    model_params = {}
    if args.model_type == 'ridge':
        model_params = {'alpha': 1.0}
    elif args.model_type == 'lasso':
        model_params = {'alpha': 0.01}
    elif args.model_type == 'elasticnet':
        model_params = {'alpha': 0.01, 'l1_ratio': 0.5}
    elif args.model_type == 'randomforest':
        model_params = {'n_estimators': 100, 'max_depth': 10}
    elif args.model_type == 'gbm':
        model_params = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3}
    
    # Train and evaluate model
    model_results = train_and_evaluate(
        df_reduced,
        target_col=args.target_col,
        feature_cols=selected_features,
        date_col=args.date_col,
        model_type=args.model_type,
        params=model_params,
        test_size=args.test_size,
        use_cv=True,
        n_cv_splits=args.cv_splits,
        plot_results=not args.no_plots
    )
    
    # Log evaluation results
    if 'test_evaluation' in model_results:
        eval_results = model_results['test_evaluation']
        logger.info(f"Test RMSE: {eval_results['rmse']:.4f}")
        logger.info(f"Test MAE: {eval_results['mae']:.4f}")
        logger.info(f"Test MAPE: {eval_results['mape']:.4f}%")
        logger.info(f"Test R²: {eval_results['r2']:.4f}")
    
    # Log residual analysis results
    if 'residual_analysis' in model_results:
        residual_analysis = model_results['residual_analysis']
        logger.info(f"Residual statistics: {residual_analysis['stats']}")
        
        if 'durbin_watson' in residual_analysis:
            logger.info(f"Durbin-Watson statistic: {residual_analysis['durbin_watson']:.4f}")
            logger.info(f"Durbin-Watson interpretation: {residual_analysis['durbin_watson_interpretation']}")
        
        if 'ljung_box' in residual_analysis:
            lb_test = residual_analysis['ljung_box']
            logger.info(f"Ljung-Box test: statistic={lb_test['statistic']:.4f}, p-value={lb_test['p_value']:.4f}")
    
    # Save model results summary
    summary = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_file': args.data_file,
        'target_column': args.target_col,
        'model_type': args.model_type,
        'feature_count': len(selected_features),
        'test_rmse': model_results.get('test_evaluation', {}).get('rmse'),
        'test_mae': model_results.get('test_evaluation', {}).get('mae'),
        'test_mape': model_results.get('test_evaluation', {}).get('mape'),
        'test_r2': model_results.get('test_evaluation', {}).get('r2'),
        'residual_autocorrelation': model_results.get('residual_analysis', {}).get('durbin_watson_interpretation'),
        'selected_features': selected_features
    }
    
    # Save summary to file
    summary_path = os.path.join(args.output_dir, 'model_summary.txt')
    with open(summary_path, 'w') as f:
        for key, value in summary.items():
            if key == 'selected_features':
                f.write(f"{key}:\n")
                for feature in value:
                    f.write(f"  - {feature}\n")
            else:
                f.write(f"{key}: {value}\n")
    
    logger.info(f"Saved model summary to {summary_path}")
    logger.info("Pipeline completed successfully")
    
    return model_results, df_reduced

if __name__ == "__main__":
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Run pipeline
        model_results, df_reduced = run_pipeline(args)
        
        logger.info("Process completed successfully")
    except Exception as e:
        logger.exception(f"Error in pipeline: {str(e)}")
        raise