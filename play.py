import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet

def main():
    # Load data
    df = pd.read_csv('data_input_sample.csv')
    
    print(f"Data shape: {df.shape}")
    print(f"Available columns: {df.columns.tolist()}")
    
    # Check for target column
    target_col = 'auto_sales_sales'
    if target_col not in df.columns:
        print(f"Target column {target_col} not found!")
        return
        
    # Check data availability in target column
    non_null_count = df[target_col].notna().sum()
    print(f"Non-null values in {target_col}: {non_null_count}/{len(df)}")
    
    # Prepare data for Prophet
    prophet_df = pd.DataFrame({
        'ds': pd.to_datetime(df['date']),
        'y': df[target_col]
    })
    
    # Handle missing values in y
    prophet_df['y'] = prophet_df['y'].fillna(method='ffill')
    
    # Split into train/test
    train_size = int(len(prophet_df) * 0.8)
    train_df = prophet_df[:train_size]
    test_df = prophet_df[train_size:]
    
    print(f"Training data: {len(train_df)} rows")
    print(f"Test data: {len(test_df)} rows")
    
    # Train basic Prophet model (no regressors for simplicity)
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0
    )
    
    model.fit(train_df)
    
    # Make future dataframe including test period
    future = model.make_future_dataframe(periods=len(test_df), freq='MS')  # 'MS' = month start
    
    # Make predictions
    forecast = model.predict(future)
    
    # Evaluate on test set
    y_true = test_df['y'].values
    y_pred = forecast.iloc[train_size:]['yhat'].values
    
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test MAPE: {mape:.4f}%")
    
    # Plot the forecast
    plt.figure(figsize=(12, 6))
    plt.plot(prophet_df['ds'], prophet_df['y'], 'ko', markersize=2, label='Actual')
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
    plt.title(f'Prophet Forecast for {target_col}')
    plt.xlabel('Date')
    plt.ylabel(target_col)
    plt.savefig('prophet_forecast.png', dpi=300, bbox_inches='tight')
    
    # Plot components
    fig = model.plot_components(forecast)
    fig.savefig('prophet_components.png', dpi=300, bbox_inches='tight')
    
    print("Forecasting complete! Check the output images.")

if __name__ == "__main__":
    main()