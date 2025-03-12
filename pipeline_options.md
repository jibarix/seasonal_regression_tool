# Time Series Forecasting Pipeline Options

This document provides detailed explanations of the command-line options available in the time series forecasting pipeline. Understanding these options will help you make informed decisions when configuring the pipeline for your specific forecasting needs.

## Data Options

### `--data-file` (Required)
Specifies the path to the CSV file containing your time series data. The file should contain at least a date column and one or more numeric columns for analysis.

**Example**: `--data-file data_input_sample.csv`

**Technical details**: The pipeline automatically handles various date formats and performs necessary conversions. It also validates data integrity and provides informative messages about the dataset characteristics.

### `--target-col` (Required)
Defines which column in your dataset should be forecasted. This is the dependent variable in your model.

**Example**: `--target-col auto_sales`

**Technical details**: The pipeline automatically performs validation checks on the target column, such as checking for sufficient non-missing values and appropriate data types. Transformations applied to features are not automatically applied to the target variable to preserve its original scale.

### `--date-col`
Specifies the column that contains date information. This column is used for time-based splitting, seasonal feature generation, visualization, and automatic frequency detection.

**Default**: `date`

**Example**: `--date-col time_stamp`

**Technical details**: The date column is converted to pandas datetime format and is used to derive time-based features. The pipeline intelligently examines the date patterns to detect the data frequency (monthly, quarterly, annual), which then influences how features are generated and how missing values are handled.

### `--output-dir`
Defines where results, visualizations, and model artifacts will be saved. The pipeline creates a timestamped directory within this location for each run, ensuring that multiple runs don't overwrite previous results.

**Default**: `results`

**Example**: `--output-dir my_forecasts`

**Technical details**: The timestamped directory includes a log file with detailed information about the pipeline run, making it easier to track and diagnose issues. If the directory doesn't exist, it will be created automatically.

## Feature Engineering Options

### `--fourier-periods`
Specifies the seasonal periods to model using Fourier terms. These are provided as comma-separated integers representing the number of time units in each seasonal cycle.

For time series with multiple seasonal patterns, you can specify multiple periods. For example, in monthly data:
- `12` captures annual seasonality (12 months per year)
- `4` captures quarterly patterns
- `3` would capture seasonal patterns that repeat every 3 months

**Default**: `12,4` (annual and quarterly seasonality for monthly data)

**Example**: `--fourier-periods 12,4,3`

**Technical details**: Fourier terms use sine and cosine functions to model cyclical patterns. They offer a flexible way to model seasonality without requiring dummy variables for each time period. The implementation automatically determines if the data already contains specific seasonal encodings (like quarter dummies) and adjusts the Fourier terms accordingly to avoid redundancy. When quarterly dummies are detected, the pipeline will only generate Fourier terms for non-quarterly seasonal patterns.

### `--fourier-harmonics`
Controls the complexity of seasonal patterns modeled by Fourier terms. Higher values capture more complex seasonal patterns but increase the risk of overfitting.

- `1`: Captures only the fundamental frequency (simple seasonal pattern)
- `2`: Adds the second harmonic, allowing for bimodal patterns
- `3`: Adds the third harmonic, allowing for more complex patterns
- `4`: Adds the fourth harmonic, for highly complex seasonal patterns

**Default**: `2`

**Example**: `--fourier-harmonics 3`

**Technical details**: Each harmonic adds one pair of sine and cosine terms. With a period of 12 and 2 harmonics, the pipeline generates 4 Fourier terms (sin_12_1, cos_12_1, sin_12_2, cos_12_2). These terms collectively model the seasonal pattern's shape. The implementation handles edge cases such as insufficient data for higher harmonics and automatically adjusts the number of harmonics based on available data points if necessary.

### `--max-lag`
Defines the maximum number of lagged values to create for the target variable and key indicators. Lagged values capture temporal dependencies in the data.

For example, with `--max-lag 3`:
- If the target is `auto_sales`, the pipeline creates features representing auto_sales values from 1, 2, and 3 time periods ago.

Setting appropriate lags helps the model learn from recent history. The appropriate value depends on the temporal dependencies in your data.

**Default**: `3`

**Example**: `--max-lag 6`

**Technical details**: The pipeline intelligently selects which variables to create lags for, focusing on the target variable and key indicators identified through column name analysis (variables containing terms like 'sales', 'rate', 'price', 'index', etc.). This selective approach maintains a manageable feature space while capturing the most relevant temporal relationships. The implementation handles missing values that result from lagging by using appropriate imputation strategies.

### `--use-log`
Applies logarithmic transformation to appropriate numeric features. This is useful for data with exponential growth patterns or features with skewed distributions.

Log transformation helps stabilize variance and can make multiplicative relationships more linear. The implementation includes automatic handling of zeros and negative values.

**Default**: Disabled

**Example**: `--use-log` (flag to enable)

**Technical details**: When enabled, the pipeline selectively applies log transformation to variables that typically benefit from it, particularly those related to volume, sales, prices, and production. A dynamic offset is automatically calculated for each variable to handle zeros or negative values. For a column with minimum value -5, the offset would automatically be adjusted to at least 6 to ensure all values become positive before taking the logarithm. The target variable is not log-transformed by default to maintain interpretability, but the model can still capture relationships with log-transformed predictors.

### `--respect-existing-features`
Instructs the pipeline to preserve and use existing date-derived features in the dataset instead of regenerating them. This is useful when your dataset already contains specially engineered temporal features.

**Default**: `True`

**Example**: `--respect-existing-features` (flag to enable)

**Technical details**: When enabled, the pipeline checks for existing time-related columns (year, month, quarter, quarter dummies, etc.) and uses them rather than creating new ones. This allows for preserving domain-specific seasonal encodings that might be present in the dataset. The implementation dynamically adjusts other feature engineering processes based on which features already exist. For example, if quarter dummies (Q1, Q2, Q3, Q4) are detected, the pipeline will only generate Fourier terms for non-quarterly seasonality.

## Dimension Reduction Options

### `--use-pca`
Enables Principal Component Analysis (PCA) for dimension reduction. PCA transforms possibly correlated features into linearly uncorrelated components, reducing dimensionality while preserving variance.

This is particularly useful when dealing with many features or highly correlated variables.

**Default**: Disabled

**Example**: `--use-pca` (flag to enable)

**Technical details**: Before applying PCA, the pipeline automatically filters out highly correlated features (correlation > 0.95) to avoid multicollinearity issues. The implementation standardizes the data before applying PCA to ensure all features contribute equally regardless of their scale. The PCA process includes a sophisticated analysis of component loadings, which traces each principal component back to the original features to maintain interpretability. If feature selection after PCA yields insufficient features, the pipeline includes fallback mechanisms to ensure a viable model can still be trained.

### `--pca-variance`
Defines the proportion of variance that should be preserved when selecting principal components. Higher values retain more information but use more components.

- `0.8`: Preserves 80% of variance (more aggressive dimension reduction)
- `0.95`: Preserves 95% of variance (balanced approach)
- `0.99`: Preserves 99% of variance (conservative reduction)

**Default**: `0.95`

**Example**: `--pca-variance 0.9`

**Technical details**: The pipeline first calculates the full PCA decomposition and then analyzes the cumulative explained variance to determine how many components to keep. It selects the minimum number of components needed to exceed the specified threshold. The implementation includes detailed logging of the variance explained by each component, which is useful for diagnosing issues with feature representation. Additionally, a variance plot is generated to visualize the explained variance distribution across components.

### `--feature-selection`
Specifies the method used to select the most relevant features for modeling. Different methods capture different aspects of the relationship between features and the target.

- `mutual_info`: Uses mutual information to capture both linear and non-linear relationships
- `f_regression`: Uses F-test to identify linear relationships
- `lasso`: Uses Lasso regression to select features while promoting sparsity

**Default**: `mutual_info`

**Example**: `--feature-selection f_regression`

**Technical details**: Feature selection helps simplify the model, reduce noise, and improve interpretability. The implementation handles missing values appropriately for each method and includes fallback mechanisms if a method fails to select enough features. For Lasso-based selection, the implementation adjusts the regularization strength as needed to ensure a suitable number of features are selected. If all feature selection methods fail, the pipeline falls back to using original numeric features to ensure robustness.

### `--top-n-features`
Defines the maximum number of features to select after the feature selection process. This limits model complexity and focuses on the most important predictors.

**Default**: `20`

**Example**: `--top-n-features 10`

**Technical details**: The pipeline ranks features by importance according to the selected feature selection method and keeps only the top N. If PCA is used, this parameter determines how many principal components to retain in the final model. If feature selection yields fewer than the specified number, all selected features are used. The implementation includes appropriate handling of edge cases such as when feature selection methods return insufficient features.

## Modeling Options

### `--model-type`
Specifies the machine learning algorithm to use for forecasting. Each model has different strengths and characteristics.

- `linear`: Standard linear regression - fast, interpretable, but captures only linear relationships
- `ridge`: Linear regression with L2 regularization - reduces overfitting by penalizing large coefficients
- `lasso`: Linear regression with L1 regularization - performs feature selection during training
- `elasticnet`: Combines L1 and L2 regularization - balances feature selection and coefficient stability
- `randomforest`: Ensemble of decision trees - captures non-linear relationships and feature interactions
- `gbm`: Gradient Boosting Machine - powerful sequential ensemble method

**Default**: `randomforest`

**Example**: `--model-type gbm`

**Technical details**: Model selection should align with your data characteristics and interpretation needs. The implementation includes pre-configured parameter settings for each model type that work well across a range of forecasting problems. For regularized models (ridge, lasso, elasticnet), the pipeline automatically handles standardization of features. For tree-based models (randomforest, gbm), the pipeline extracts and visualizes feature importance. The implementation includes appropriate error handling for each model type and detailed logging of model parameters and performance metrics.

### `--test-size`
Determines what proportion of data to reserve for the final model evaluation. This data is not used during model training.

- `0.1`: 10% of data used for testing (90% for training)
- `0.2`: 20% of data used for testing (80% for training)
- `0.3`: 30% of data used for testing (70% for training)

**Default**: `0.2`

**Example**: `--test-size 0.15`

**Technical details**: Since this is time series data, the split is always performed chronologically, with the most recent data in the test set. This mimics the real-world forecasting scenario where you predict future values based on historical data. The implementation ensures that the training set contains enough data points to properly capture seasonal patterns based on the data frequency. If the test size would result in too few training samples, a warning is logged and the test size is automatically adjusted.

### `--cv-splits`
Defines the number of folds to use in time series cross-validation. Each fold creates a different train/validation split point for model evaluation.

More splits provide a more robust estimate of model performance but increase computation time.

**Default**: `5`

**Example**: `--cv-splits 3`

**Technical details**: Unlike standard cross-validation, time series cross-validation respects the temporal order of observations. Each validation set consists of a continuous time period that follows its respective training set. The implementation uses the `TimeSeriesSplit` from scikit-learn with appropriate configuration for time series data. The pipeline ensures that each training fold has enough data to properly capture seasonal patterns and applies consistent preprocessing across folds.

### `--no-plots`
Disables the generation of diagnostic plots and visualizations. This can be useful for batch processing or when running on systems without display capabilities.

**Default**: Disabled (plots are generated)

**Example**: `--no-plots` (flag to enable)

**Technical details**: When plots are enabled, the pipeline generates visualizations for actual vs. predicted values, residual analysis (including histogram, QQ plot, and autocorrelation plot), feature importance or coefficients, and, if applicable, PCA variance explained. Even when plots are disabled, the pipeline still calculates all diagnostic statistics and includes them in the model summary. Plots are automatically saved to the output directory with appropriate naming for easy reference.

## Advanced Usage Scenarios

### Handling Multiple Seasonal Patterns

For data with multiple seasonal patterns (e.g., daily data with weekly and annual patterns), you can specify multiple periods in the Fourier terms:

```bash
python main.py --data-file daily_data.csv --target-col sales --fourier-periods 7,365 --fourier-harmonics 3
```

This captures both weekly seasonality (period=7) and annual seasonality (period=365). The harmonic parameter controls how complex each seasonal pattern can be, with higher values allowing more complex shapes.

### Dealing with High Cardinality Features

When your dataset contains many categorical variables with high cardinality, consider this approach:

```bash
python main.py --data-file data.csv --target-col target --use-pca --pca-variance 0.9 --top-n-features 15
```

This configuration applies PCA to reduce dimensionality while preserving 90% of the variance, then selects the top 15 most informative components or features. This is particularly useful when one-hot encoding has created a large feature space.

### Forecasting Highly Volatile Series

For volatile series with irregular patterns, consider using tree-based models with more lags:

```bash
python main.py --data-file volatile_data.csv --target-col price --model-type gbm --max-lag 12 --no-pca
```

This uses gradient boosting with a longer lag structure to capture complex temporal dependencies, without applying PCA to preserve the original feature space which may contain important signals.

### Interpretable Models for Business Insights

When explanation is more important than raw prediction accuracy:

```bash
python main.py --data-file business_data.csv --target-col revenue --model-type lasso --feature-selection f_regression --top-n-features 10
```

This creates a sparse linear model with only the most statistically significant predictors, making it easier to explain relationships to stakeholders.

## Error Handling and Edge Cases

The pipeline implements several safeguards to ensure robustness:

1. **Missing value detection and imputation**: Intelligently handles missing values using context-aware strategies (group-based imputations for time series data).

2. **Data validation**: Checks for sufficient data points, valid date ranges, and appropriate data types before processing.

3. **Feature correlation filtering**: Automatically detects and removes highly correlated features that could cause instability.

4. **Frequency detection**: Automatically detects data frequency (monthly, quarterly, annual) and adjusts feature engineering accordingly.

5. **Fallback mechanisms**: If feature selection or dimension reduction fails to yield sufficient features, the pipeline falls back to simpler approaches to ensure a model can still be trained.

6. **Log transformation safeguards**: Automatically adjusts offsets for log transformations to handle zeros and negative values.

7. **MAPE calculation protection**: Handles zero values in the denominator when calculating Mean Absolute Percentage Error.

8. **Residual analysis**: Performs advanced diagnostic tests for autocorrelation (Ljung-Box and Durbin-Watson) with appropriate interpretation.

## Performance Considerations

- **For large datasets** (>100,000 rows): Consider using `--no-plots` to reduce memory usage and processing time.

- **For high-dimensional data** (many features): Use correlation filtering and PCA to reduce dimensionality before modeling.

- **For computationally intensive models** (like GBM with many features): Reduce `--cv-splits` to 3 to decrease training time.

- **For memory-constrained environments**: Reduce `--top-n-features` and avoid using tree-based models with very deep trees.

Understanding these options allows you to tailor the forecasting pipeline to your specific data characteristics, computational resources, and business requirements.