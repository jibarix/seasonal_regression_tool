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
├── pipeline_options.md        - Detailed CLI options documentation
├── .gitignore                 - Git ignore configuration
└── README.md                  - Project documentation
```

## Key Features

- **Intelligent Data Preprocessing**:
  - Automatic frequency detection (monthly, quarterly, annual)
  - Sophisticated missing value handling with group-based imputation (by year/month/quarter)
  - Automatic trimming to common date ranges where data is available
  - Graceful handling of different data formats and time variables

- **Feature Engineering**: 
  - Fourier terms for modeling multiple seasonal patterns simultaneously
  - Smart lagged variable creation focused on target and key indicators
  - Selective log transformations with automatic offset adjustment for negative values
  - Context-aware rolling window features for relevant metrics
  - Preservation of existing date features to respect domain-specific encodings

- **Dimension Reduction**:
  - Automatic correlation filtering to remove highly correlated features (>0.95)
  - PCA with adaptive variance threshold selection
  - PCA loading analysis to trace component importance back to original features
  - Multiple feature selection methods (mutual information, F-regression, Lasso)
  - Robust fallback mechanisms when feature selection yields insufficient features

- **Modeling**:
  - Multiple regression models with appropriate parameter configurations
  - Time series cross-validation that respects temporal order
  - Comprehensive model evaluation (RMSE, MAE, MAPE, R²)
  - Residual analysis with autocorrelation testing (Ljung-Box, Durbin-Watson)
  - Interpretable model outputs with feature importance analysis

- **Visualization**:
  - Time series plots with proper date formatting
  - Seasonal decomposition visualization
  - Correlation analysis heatmaps
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
- scipy

Install requirements with:

```bash
pip install -r requirements.txt
```

## Usage

The pipeline can be run from the command line with various options:

```bash
python main.py --data-file data_input_sample.csv --target-col auto_sales --use-pca --model-type randomforest
```

### Command Line Arguments Overview

#### Data Arguments:
- `--data-file`: Path to the input data CSV file (required)
- `--target-col`: Name of the target column to forecast (required)
- `--date-col`: Name of the date column (default: 'date')
- `--output-dir`: Directory to save results (default: 'results')

#### Feature Engineering Arguments:
- `--fourier-periods`: Comma-separated list of seasonal periods (default: '12,4')
- `--fourier-harmonics`: Number of Fourier harmonics (default: 2)
- `--max-lag`: Maximum lag for lagged features (default: 6)
- `--use-log`: Apply log transformation to appropriate features (flag)
- `--respect-existing-features`: Preserve existing date features (default: True)

#### Dimension Reduction Arguments:
- `--use-pca`: Use PCA for dimension reduction (flag)
- `--pca-variance`: PCA cumulative variance threshold (default: 0.95)
- `--feature-selection`: Feature selection method (default: 'mutual_info')
- `--top-n-features`: Number of top features to select (default: 10)

#### Modeling Arguments:
- `--model-type`: Type of model to train (default: 'randomforest')
- `--test-size`: Proportion of data to use for testing (default: 0.2)
- `--cv-splits`: Number of splits for time series cross-validation (default: 5)
- `--no-plots`: Disable plotting of results (flag)

For a detailed explanation of all options, see `pipeline_options.md`.

## Output

The pipeline produces several outputs in a timestamped directory within the specified output directory:

1. **prepared_data.csv**: Data after preprocessing with missing values handled
2. **engineered_features.csv**: Data with all engineered features added
3. **reduced_data.csv**: Data after dimension reduction and feature selection
4. **model_summary.txt**: Detailed summary of model performance and parameters
5. **Visualization plots**: 
   - Actual vs. predicted values
   - Residual analysis with autocorrelation tests
   - Feature importance or model coefficients
   - PCA variance explained (when applicable)

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
df_features = engineer_features(df, target_col='auto_sales', 
                               create_fourier=True,
                               log_transform=True)

# Reduce dimensions
df_reduced, metadata = reduce_dimensions(df_features, 
                                        target_col='auto_sales', 
                                        use_pca=True)

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

## Advanced Usage Examples

### Forecasting with Complex Seasonality
```bash
python main.py --data-file data_input_sample.csv --target-col auto_sales --fourier-periods 12,4,6 --fourier-harmonics 3 --max-lag 12
```
This configuration models three different seasonal patterns (annual, quarterly, and biannual) with complex shapes (3 harmonics) and captures a full year of lagged dependencies.

### High-Accuracy Nonlinear Forecasting
```bash
python main.py --data-file data_input_sample.csv --target-col imports_value --model-type gbm --use-log --max-lag 6 --feature-selection mutual_info --top-n-features 15
```
This setup uses Gradient Boosting with log-transformed features and mutual information feature selection to capture complex nonlinear patterns in import values.

### Highly Interpretable Model with PCA
```bash
python main.py --data-file data_input_sample.csv --target-col federal_funds_rate_rate --use-pca --pca-variance 0.9 --model-type elasticnet --feature-selection lasso --no-plots
```
This configuration creates an interpretable ElasticNet model on PCA-transformed features, with additional feature selection through Lasso to identify the most important components affecting federal funds rates.

## Error Handling and Robustness

The pipeline implements several safeguards to ensure robustness:

- **Graceful feature handling**: Automatically detects and adapts to existing features
- **Fallback mechanisms**: Implements multiple levels of fallbacks for feature selection and transformations
- **Missing value strategies**: Uses context-aware imputation with multiple methods
- **Correlation filtering**: Automatically removes redundant highly-correlated features
- **Edge case handling**: Properly handles negative values in transformations, zeros in MAPE calculations, etc.

## Practical Considerations

1. **For large datasets**: Consider using `--use-pca` and limiting the features with `--top-n-features` to improve performance.

2. **For noisy data**: Increase regularization by using `ridge`, `lasso`, or `elasticnet` models.

3. **For capturing long-term dependencies**: Increase `--max-lag` to include more historical values.

4. **For complex seasonal patterns**: Increase `--fourier-harmonics` and include all relevant seasonal periods in `--fourier-periods`.

5. **For highly interpretable results**: Use `linear` or `lasso` models without PCA and examine the coefficients.

## License

This project is available for educational and research purposes. Please contact the authors for commercial use permissions.