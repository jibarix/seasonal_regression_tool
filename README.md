# Time Series Analysis and Forecasting Pipeline

A modular and efficient pipeline for time series analysis, feature engineering, dimension reduction, and forecasting. This project addresses common challenges in time series forecasting by implementing a comprehensive approach that includes data preprocessing, feature engineering with Fourier terms, dimensionality reduction via PCA, and robust model evaluation.

## Project Structure

```
project/
├── data/
│   ├── processor.py           - Data loading and preprocessing
│   ├── transformations.py     - Time series transformations
│   ├── feature_engineering.py - Feature creation with Fourier terms
│   └── dimension_reduction.py - PCA and feature selection
├── models/
│   ├── modeling.py            - Model training and evaluation
│   └── visualization.py       - Time series visualization
├── main.py                    - Pipeline orchestration
├── data_input_sample.csv      - Example data file
├── implementation_data.md     - Dataset documentation
├── .gitignore                 - Git ignore configuration
└── README.md                  - Project documentation
```

## Key Features

- **Data Preprocessing**: Handles missing values, frequency detection, and time variable creation
- **Feature Engineering**: 
  - Fourier terms for modeling seasonality
  - Lagged variables for capturing temporal dependencies
  - Log transformations with proper handling of edge cases
  - Rolling window features and interaction terms
- **Dimension Reduction**:
  - PCA with automatic variance threshold selection
  - Multiple feature selection methods (mutual information, F-regression, Lasso)
  - PCA loading analysis to trace back component importance
- **Modeling**:
  - Multiple regression models (linear, ridge, lasso, elasticnet, random forest, gradient boosting)
  - Time series cross-validation
  - Comprehensive model evaluation (RMSE, MAE, MAPE, R²)
  - Residual analysis with autocorrelation testing (Ljung-Box, Durbin-Watson)
- **Visualization**:
  - Time series plots with proper date formatting
  - Seasonal decomposition visualization
  - Feature importance and PCA component visualization
  - Prediction and error analysis plots

## Dataset Information

The project works with a comprehensive economic indicators dataset that includes:
- Monthly and quarterly economic indicators
- Automotive industry metrics (sales, manufacturing, inventories)
- Financial indicators (federal funds rate, equity risk premium)
- Labor statistics
- Price indices
- GDP and forecasts

Detailed information about the dataset is available in `implementation_data.md`.

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- statsmodels
- matplotlib
- seaborn

## Usage

The pipeline can be run from the command line with various options:

```bash
python main.py --data-file data_input_sample.csv --target-col auto_sales --use-pca --model-type randomforest
```

### Command Line Arguments

#### Data Arguments:
- `--data-file`: Path to the input data CSV file (required)
- `--target-col`: Name of the target column to forecast (required)
- `--date-col`: Name of the date column (default: 'date')
- `--output-dir`: Directory to save results (default: 'results')

#### Feature Engineering Arguments:
- `--fourier-periods`: Comma-separated list of Fourier periods (default: '12,4')
- `--fourier-harmonics`: Number of Fourier harmonics (default: 2)
- `--max-lag`: Maximum lag for lagged features (default: 6)
- `--use-log`: Apply log transformation to features (flag)

#### Dimension Reduction Arguments:
- `--use-pca`: Use PCA for dimension reduction (flag)
- `--pca-variance`: PCA cumulative variance threshold (default: 0.95)
- `--feature-selection`: Feature selection method ('mutual_info', 'f_regression', 'lasso') (default: 'mutual_info')
- `--top-n-features`: Number of top features to select (default: 10)

#### Modeling Arguments:
- `--model-type`: Type of model to train ('linear', 'ridge', 'lasso', 'elasticnet', 'randomforest', 'gbm') (default: 'linear')
- `--test-size`: Proportion of data to use for testing (default: 0.2)
- `--cv-splits`: Number of splits for time series cross-validation (default: 5)
- `--no-plots`: Disable plotting of results (flag)

## Output

The pipeline produces several outputs in the specified output directory:

1. **prepared_data.csv**: Data after preprocessing
2. **engineered_features.csv**: Data after feature engineering
3. **reduced_data.csv**: Data after dimension reduction
4. **model_summary.txt**: Summary of model performance and parameters
5. Various plots showing model predictions, residual analysis, and feature importance

## Example

```python
# Basic example of using the pipeline components directly
from data.processor import DataLoader
from data.feature_engineering import engineer_features
from data.dimension_reduction import reduce_dimensions
from models.modeling import train_and_evaluate

# Load and prepare data
loader = DataLoader()
df = loader.load_data('data_input_sample.csv')

# Engineer features
df_features = engineer_features(df, target_col='auto_sales', create_fourier=True)

# Reduce dimensions
df_reduced, metadata = reduce_dimensions(df_features, target_col='auto_sales', use_pca=True)

# Train and evaluate model
results = train_and_evaluate(df_reduced, 
                            target_col='auto_sales',
                            feature_cols=metadata['selected_features'],
                            model_type='randomforest')

# Print evaluation metrics
print(f"RMSE: {results['test_evaluation']['rmse']:.4f}")
print(f"MAE: {results['test_evaluation']['mae']:.4f}")
print(f"R²: {results['test_evaluation']['r2']:.4f}")
```

## Project Objectives

This pipeline was designed to address the following key objectives found in the project roadmap:

1. Select and export data based on target variable availability
2. Limit data based on target data available time period
3. Handle missing data points with appropriate imputation methods
4. Check for anomalies and handle different units/scaling
5. Apply Fourier terms for seasonality modeling
6. Transform data with log transformations, handling edge cases
7. Apply lagged variables based on domain knowledge
8. Perform dimensionality reduction via PCA with proper standardization
9. Select optimal features from PCA outputs
10. Run regression models with cross-validation
11. Evaluate models with multiple metrics
12. Check residuals for patterns and autocorrelation
13. Examine PCA loadings to understand key drivers

## License

This project is available for educational and research purposes. Please contact the authors for commercial use permissions.