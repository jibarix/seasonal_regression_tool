"""
Standalone script to run Prophet directly on the input data.
This bypasses the pipeline's feature engineering and dimension reduction steps.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'results/prophet_direct_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    data_file = 'data_input_sample.csv'
    df = pd.read_csv(data_file)
    logger.info(f"Loaded data with shape: {df.shape}")
    
    # Set target and date columns
    target_col = 'auto_sales_sales'
    date_col = 'date'
    
    # Check if columns exist
    if target_col not in df.columns:
        logger.error(f"Target column '{target_col}' not found")
        return
    if date_col not in df.columns:
        logger.error(f"Date column '{date_col}' not found")
        return
    
    # Handle missing values in target column
    logger.info(f"Missing values in target: {df[target_col].isna().sum()}")
    df[target_col] = df[target_col].fillna(df[target_col].mean())
    
    # Prepare data for Prophet
    prophet_df = pd.DataFrame()
    prophet_df['ds'] = pd.to_datetime(df[date_col])
    prophet_df['y'] = df[target_col]
    
    # Add regressors
    regressors = ['used_car_retail_sales_sales', 'auto_manufacturing_orders_orders', 'unemployment_rate_rate']
    for reg in regressors:
        if reg in df.columns:
            logger.info(f"Adding regressor: {reg}")
            prophet_df[reg] = df[reg]
            if prophet_df[reg].isna().any():
                logger.info(f"Filling {prophet_df[reg].isna().sum()} missing values in {reg}")
                prophet_df[reg] = prophet_df[reg].fillna(prophet_df[reg].mean())
    
    # Log data info
    logger.info(f"Prophet dataframe shape: {prophet_df.shape}")
    logger.info(f"Prophet columns: {prophet_df.columns.tolist()}")
    
    # Split data
    train_size = int(len(prophet_df) * 0.8)
    train_df = prophet_df.iloc[:train_size]
    test_df = prophet_df.iloc[train_size:]
    logger.info(f"Training size: {len(train_df)}, Test size: {len(test_df)}")
    
    # Create and fit model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.01,  # Reduced from default 0.05
        seasonality_prior_scale=5.0    # Reduced from default 10.0
    )
 
    # Add quarterly seasonality explicitly
    model.add_seasonality(
        name='quarterly',
        period=91.25,  # ~365/4
        fourier_order=3
    )
    
    # Add regressor if it exists in the dataframe
    for reg in regressors:
        if reg in prophet_df.columns:
            model.add_regressor(reg)
    
    logger.info("Fitting model...")
    model.fit(train_df)
    
    # Make predictions
    future = model.make_future_dataframe(periods=len(test_df), freq='MS')
    
    # Add regressor values to future dataframe
    for reg in regressors:
        if reg in prophet_df.columns:
            # Get mean value (for filling missing values)
            reg_mean = prophet_df[reg].mean()
            
            # For historical data, use actual values
            known_values = prophet_df[reg].values
            
            # For future points, use the mean
            future_periods = len(future) - len(prophet_df)
            future_values = np.full(future_periods, reg_mean)
            
            # Combine and set to future dataframe
            all_values = np.concatenate([known_values, future_values])
            future[reg] = pd.Series(all_values, index=future.index)
            
            # Ensure no NaN values
            if future[reg].isna().any():
                logger.warning(f"Found {future[reg].isna().sum()} NaN values in regressor '{reg}', filling with mean")
                future[reg] = future[reg].fillna(reg_mean)
    
    # Log future dataframe info
    logger.info(f"Future dataframe shape: {future.shape}")
    logger.info(f"Future columns: {future.columns.tolist()}")
    
    forecast = model.predict(future)
    
    # Evaluate predictions
    y_true = test_df['y'].values
    y_pred = forecast.iloc[train_size:]['yhat'].values
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    
    logger.info(f"Results:")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"MAPE: {mape:.4f}%")
    logger.info(f"R²: {r2:.4f}")
    
    # Save results
    with open(f"{output_dir}/results.txt", "w") as f:
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"MAPE: {mape:.4f}%\n")
        f.write(f"R²: {r2:.4f}\n")
    
    # Plot forecast
    plt.figure(figsize=(12, 6))
    plt.plot(prophet_df['ds'], prophet_df['y'], 'k.', label='Actual')
    plt.plot(forecast['ds'], forecast['yhat'], 'b-', label='Forecast')
    plt.fill_between(
        forecast['ds'],
        forecast['yhat_lower'],
        forecast['yhat_upper'],
        color='blue', alpha=0.2,
        label='Uncertainty Interval'
    )
    plt.axvline(x=prophet_df['ds'].iloc[train_size-1], color='r', linestyle='--', label='Train/Test Split')
    plt.legend()
    plt.title(f"Prophet Forecast for {target_col}")
    plt.xlabel("Date")
    plt.ylabel(target_col)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/forecast.png", dpi=300, bbox_inches='tight')
    
    # Plot components
    fig = model.plot_components(forecast)
    fig.savefig(f"{output_dir}/components.png", dpi=300, bbox_inches='tight')
    
    logger.info(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()