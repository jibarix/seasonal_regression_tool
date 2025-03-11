# Time Series Forecasting Pipeline Options

This document provides detailed explanations of the command-line options available in the time series forecasting pipeline. Understanding these options will help you make informed decisions when configuring the pipeline for your specific forecasting needs.

## Data Options

### `--data-file` (Required)
Specifies the path to the CSV file containing your time series data. The file should contain at least a date column and one or more numeric columns for analysis.

**Example**: `--data-file data_input_sample.csv`

### `--target-col` (Required)
Defines which column in your dataset should be forecasted. This is the dependent variable in your model.

**Example**: `--target-col auto_sales`

### `--date-col`
Specifies the column that contains date information. This column is used for time-based splitting, seasonal feature generation, and visualization.

**Default**: `date`

**Example**: `--date-col time_stamp`

### `--output-dir`
Defines where results, visualizations, and model artifacts will be saved. The pipeline creates a timestamped directory within this location for each run.

**Default**: `results`

**Example**: `--output-dir my_forecasts`

## Feature Engineering Options

### `--fourier-periods`
Specifies the seasonal periods to model using Fourier terms. These are provided as comma-separated integers representing the number of time units in each seasonal cycle.

For time series with multiple seasonal patterns, you can specify multiple periods. For example, in monthly data:
- `12` captures annual seasonality (12 months per year)
- `4` captures quarterly patterns
- `3` would capture seasonal patterns that repeat every 3 months

**Default**: `12,4` (annual and quarterly seasonality for monthly data)

**Example**: `--fourier-periods 12,4,3`

**Technical details**: Fourier terms use sine and cosine functions to model cyclical patterns. They offer a flexible way to model seasonality without requiring dummy variables for each time period. The transformation converts a time index into smooth cyclical features.

### `--fourier-harmonics`
Controls the complexity of seasonal patterns modeled by Fourier terms. Higher values capture more complex seasonal patterns but increase the risk of overfitting.

- `1`: Captures only the fundamental frequency (simple seasonal pattern)
- `2`: Adds the second harmonic, allowing for bimodal patterns
- `3`: Adds the third harmonic, allowing for more complex patterns
- `4`: Adds the fourth harmonic, for highly complex seasonal patterns

**Default**: `2`

**Example**: `--fourier-harmonics 3`

**Technical details**: Each harmonic adds one pair of sine and cosine terms. With a period of 12 and 2 harmonics, the pipeline generates 4 Fourier terms (sin_12_1, cos_12_1, sin_12_2, cos_12_2). These terms collectively model the seasonal pattern's shape.

### `--max-lag`
Defines the maximum number of lagged values to create for the target variable and key indicators. Lagged values capture temporal dependencies in the data.

For example, with `--max-lag 3`:
- If the target is `auto_sales`, the pipeline creates features representing auto_sales values from 1, 2, and 3 time periods ago.

Setting appropriate lags helps the model learn from recent history. The appropriate value depends on the temporal dependencies in your data.

**Default**: `3`

**Example**: `--max-lag 6`

**Technical details**: Lags are essential for time series forecasting as they encode the time-based relationships. Instead of using autoregressive modeling directly, lags transform the time series problem into a supervised learning problem that works with traditional machine learning algorithms.

### `--use-log`
Applies logarithmic transformation to appropriate numeric features. This is useful for data with exponential growth patterns or features with skewed distributions.

Log transformation helps stabilize variance and can make multiplicative relationships more linear. The implementation includes automatic handling of zeros and negative values.

**Default**: Disabled

**Example**: `--use-log` (flag to enable)

**Technical details**: When enabled, the pipeline applies log transformation to variables that typically benefit from it, particularly those related to volume, sales, prices, and production. A small offset is automatically added to handle zeros or negative values.

### `--respect-existing-features`
Instructs the pipeline to preserve and use existing date-derived features in the dataset instead of regenerating them. This is useful when your dataset already contains specially engineered temporal features.

**Default**: `True`

**Example**: `--respect-existing-features` (flag to enable)

**Technical details**: When enabled, the pipeline checks for existing time-related columns (year, month, quarter, etc.) and uses them rather than creating new ones. This allows for preserving domain-specific seasonal encodings that might be present in the dataset.

## Dimension Reduction Options

### `--use-pca`
Enables Principal Component Analysis (PCA) for dimension reduction. PCA transforms possibly correlated features into linearly uncorrelated components, reducing dimensionality while preserving variance.

This is particularly useful when dealing with many features or highly correlated variables.

**Default**: Disabled

**Example**: `--use-pca` (flag to enable)

**Technical details**: PCA finds the directions (principal components) that maximize the variance in the feature space. The implementation automatically standardizes the data before applying PCA to ensure all features contribute equally regardless of their scale.

### `--pca-variance`
Defines the proportion of variance that should be preserved when selecting principal components. Higher values retain more information but use more components.

- `0.8`: Preserves 80% of variance (more aggressive dimension reduction)
- `0.95`: Preserves 95% of variance (balanced approach)
- `0.99`: Preserves 99% of variance (conservative reduction)

**Default**: `0.95`

**Example**: `--pca-variance 0.9`

**Technical details**: The pipeline automatically determines how many components to keep based on the cumulative explained variance ratio. It selects the minimum number of components needed to exceed the specified threshold.

### `--feature-selection`
Specifies the method used to select the most relevant features for modeling. Different methods capture different aspects of the relationship between features and the target.

- `mutual_info`: Uses mutual information to capture both linear and non-linear relationships
- `f_regression`: Uses F-test to identify linear relationships
- `lasso`: Uses Lasso regression to select features while promoting sparsity

**Default**: `mutual_info`

**Example**: `--feature-selection f_regression`

**Technical details**: Feature selection helps simplify the model, reduce noise, and improve interpretability. The implementation handles missing values and properly scales features when required by the selection method.

### `--top-n-features`
Defines the maximum number of features to select after the feature selection process. This limits model complexity and focuses on the most important predictors.

**Default**: `20`

**Example**: `--top-n-features 10`

**Technical details**: The pipeline ranks features by importance according to the selected feature selection method and keeps only the top N. If PCA is used, this parameter determines how many principal components to retain in the final model.

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

**Technical details**: Model selection should align with your data characteristics and interpretation needs. Linear models offer better interpretability, while tree-based models often provide higher accuracy for complex relationships.

### `--test-size`
Determines what proportion of data to reserve for the final model evaluation. This data is not used during model training.

- `0.1`: 10% of data used for testing (90% for training)
- `0.2`: 20% of data used for testing (80% for training)
- `0.3`: 30% of data used for testing (70% for training)

**Default**: `0.2`

**Example**: `--test-size 0.15`

**Technical details**: Since this is time series data, the split is always performed chronologically, with the most recent data in the test set. This mimics the real-world forecasting scenario where you predict future values based on historical data.

### `--cv-splits`
Defines the number of folds to use in time series cross-validation. Each fold creates a different train/validation split point for model evaluation.

More splits provide a more robust estimate of model performance but increase computation time.

**Default**: `5`

**Example**: `--cv-splits 3`

**Technical details**: Unlike standard cross-validation, time series cross-validation respects the temporal order of observations. Each validation set consists of a continuous time period that follows its respective training set. The implementation uses the `TimeSeriesSplit` from scikit-learn.

### `--no-plots`
Disables the generation of diagnostic plots and visualizations. This can be useful for batch processing or when running on systems without display capabilities.

**Default**: Disabled (plots are generated)

**Example**: `--no-plots` (flag to enable)

**Technical details**: When plots are enabled, the pipeline generates visualizations for actual vs. predicted values, residual analysis, feature importance, and, if applicable, PCA variance explained. These are saved in the output directory.

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

## Practical Considerations

1. **For large datasets**: Consider using `--use-pca` and limiting the features with `--top-n-features` to improve performance.

2. **For noisy data**: Increase regularization by using `ridge`, `lasso`, or `elasticnet` models.

3. **For capturing long-term dependencies**: Increase `--max-lag` to include more historical values.

4. **For complex seasonal patterns**: Increase `--fourier-harmonics` and include all relevant seasonal periods in `--fourier-periods`.

5. **For highly interpretable results**: Use `linear` or `lasso` models without PCA and examine the coefficients.

Understanding these options allows you to tailor the forecasting pipeline to your specific data characteristics and business requirements.