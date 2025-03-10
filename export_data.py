"""
Export economic data to CSV using the DataConnector class.
Final version with correct handling of quarterly data.
"""
import pandas as pd
import argparse
import logging
from typing import Dict, Optional, List, Tuple

from db_connector import DataConnector
from db_config import DATASETS  # Direct import for clarity

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('data_export')

def add_time_variables(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """
    Add time-related variables (quarters, months, etc.) to a DataFrame.
    
    Args:
        df: DataFrame containing date column
        date_col: Name of the date column
        
    Returns:
        DataFrame with additional time variables
    """
    if df.empty:
        return df
    
    # Ensure date column is datetime
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Add year
    df['year'] = df[date_col].dt.year
    
    # Add month (1-12)
    df['month'] = df[date_col].dt.month
    
    # Add month name
    df['month_name'] = df[date_col].dt.strftime('%b')
    
    # Add month end date (the actual period the data represents)
    df['month_end'] = df[date_col] + pd.offsets.MonthEnd(0)
    
    # Add quarter number (1-4)
    df['quarter'] = df[date_col].dt.quarter
    
    # Add quarter end date
    df['quarter_end'] = df[date_col] + pd.offsets.QuarterEnd(0)
    
    # Add dummy variables for quarters
    for q in range(1, 5):
        df[f'Q{q}'] = (df['quarter'] == q).astype(int)
    
    # Add period label
    df['period'] = df[date_col].dt.strftime('%b %Y')
    
    # Ensure all time columns are present and not None/NaN
    time_cols = ['year', 'month', 'month_name', 'month_end', 'quarter', 'quarter_end', 'period', 
                'Q1', 'Q2', 'Q3', 'Q4']
    
    for col in time_cols:
        if col in df.columns and df[col].isna().any():
            logger.warning(f"Column {col} contains NaN values. Attempting to fix.")
            if col in ['year', 'month', 'quarter']:
                # These should be numeric
                df[col] = df[col].fillna(0).astype(int)
            elif col in ['Q1', 'Q2', 'Q3', 'Q4']:
                # These should be binary
                df[col] = df[col].fillna(0).astype(int)
            elif col in ['month_name', 'period']:
                # These should be strings
                df[col] = df[col].fillna('')
            elif col in ['month_end', 'quarter_end']:
                # These should be dates
                df[col] = df[col].fillna(pd.NaT)
    
    return df

def create_date_range_df(start_date: pd.Timestamp, end_date: pd.Timestamp, 
                          freq: str = 'MS') -> pd.DataFrame:
    """
    Create a DataFrame with a complete date range and time variables.
    
    Args:
        start_date: The start date
        end_date: The end date
        freq: Frequency ('MS' for month start)
        
    Returns:
        DataFrame with date range and time variables
    """
    # Create date range for standard months
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
    
    # Create base DataFrame
    df = pd.DataFrame({'date': date_range})
    
    # Add time variables
    df = add_time_variables(df)
    
    return df

def detect_frequency(df: pd.DataFrame, date_col: str = 'date') -> str:
    """
    Detect the frequency of a dataset based on date differences.
    
    Args:
        df: DataFrame with date column
        date_col: Name of the date column
        
    Returns:
        'monthly', 'quarterly', or 'unknown'
    """
    if df.empty or len(df) < 2:
        return 'unknown'
    
    # Calculate date differences
    date_diffs = pd.Series(pd.to_datetime(df[date_col]).sort_values().diff().dt.days.dropna())
    if date_diffs.empty:
        return 'unknown'
    
    # Get median difference to avoid outliers
    median_diff = date_diffs.median()
    
    # Classify based on median difference
    if 28 <= median_diff <= 31:
        return 'monthly'
    elif 80 <= median_diff <= 95:
        return 'quarterly'
    else:
        return 'unknown'

def map_quarterly_date_to_month(date):
    """
    Maps a FRED quarterly date to the month that should have the data.
    
    In FRED data in our database:
    - 2024-04-01 represents Q1 2024 (data for Jan-Mar)
    - 2024-07-01 represents Q2 2024 (data for Apr-Jun)
    - 2024-10-01 represents Q3 2024 (data for Jul-Sep)
    - 2025-01-01 represents Q4 2024 (data for Oct-Dec)
    
    We want to map each quarterly data point to all three months in the quarter it represents.
    
    Args:
        date: pandas Timestamp with the FRED quarterly date
        
    Returns:
        list of three pandas Timestamps for the three months in the quarter
    """
    month = date.month
    year = date.year
    
    if month == 4:  # Apr 1 -> Q1 (Jan, Feb, Mar)
        return [
            pd.Timestamp(year, 1, 1),  # Jan 1
            pd.Timestamp(year, 2, 1),  # Feb 1
            pd.Timestamp(year, 3, 1)   # Mar 1
        ]
    elif month == 7:  # Jul 1 -> Q2 (Apr, May, Jun)
        return [
            pd.Timestamp(year, 4, 1),  # Apr 1
            pd.Timestamp(year, 5, 1),  # May 1
            pd.Timestamp(year, 6, 1)   # Jun 1
        ]
    elif month == 10:  # Oct 1 -> Q3 (Jul, Aug, Sep)
        return [
            pd.Timestamp(year, 7, 1),  # Jul 1
            pd.Timestamp(year, 8, 1),  # Aug 1
            pd.Timestamp(year, 9, 1)   # Sep 1
        ]
    elif month == 1:  # Jan 1 -> Q4 of previous year (Oct, Nov, Dec)
        prev_year = year - 1
        return [
            pd.Timestamp(prev_year, 10, 1),  # Oct 1
            pd.Timestamp(prev_year, 11, 1),  # Nov 1
            pd.Timestamp(prev_year, 12, 1)   # Dec 1
        ]
    else:
        logger.warning(f"Unexpected quarterly date: {date}")
        return []

def expand_quarterly_data(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """
    Expands quarterly data to monthly data.
    
    For each quarterly data point, it creates three monthly rows (one for each month in the quarter)
    with the same data values.
    
    Args:
        df: DataFrame with quarterly data
        date_col: Name of the date column
        
    Returns:
        DataFrame with expanded monthly data
    """
    if df.empty:
        return df
    
    # Create expanded dataframe
    expanded_rows = []
    
    for _, row in df.iterrows():
        quarterly_date = row[date_col]
        monthly_dates = map_quarterly_date_to_month(quarterly_date)
        
        for monthly_date in monthly_dates:
            # Create a copy of the row with the new date
            new_row = row.copy()
            new_row[date_col] = monthly_date
            expanded_rows.append(new_row)
    
    # Create new dataframe with expanded rows
    expanded_df = pd.DataFrame(expanded_rows)
    
    # Sort by date
    expanded_df = expanded_df.sort_values(date_col).reset_index(drop=True)
    
    return expanded_df

def prepare_datasets(datasets: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    """
    Prepare datasets for merging by adding time variables and detecting frequencies.
    
    Args:
        datasets: Dictionary of dataset name to DataFrame
        
    Returns:
        Tuple of processed datasets and frequency mapping
    """
    processed_data = {}
    frequency_mapping = {}
    
    for name, df in datasets.items():
        if df.empty:
            logger.warning(f"Dataset '{name}' is empty. Skipping.")
            continue
        
        # Get dataset config
        dataset_config = DATASETS[name]
        date_col = dataset_config['date_col']
        
        # Get frequency from config first if available
        frequency = dataset_config.get('frequency', None)
        
        # If not in config, detect frequency
        if not frequency:
            frequency = detect_frequency(df, date_col)
            logger.info(f"Detected {frequency} frequency for dataset '{name}'")
        
        # Standardize all date values to datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        # For quarterly data, expand to monthly
        if frequency == 'quarterly':
            logger.info(f"Expanding quarterly data for {name} to monthly")
            df = expand_quarterly_data(df, date_col)
            
        # Add time variables
        df = add_time_variables(df, date_col)
        
        # Add dataset name to column names for value columns
        value_cols = dataset_config.get('value_cols', [dataset_config.get('value_col', 'value')])
        df = df.rename(columns={col: f"{name}_{col}" for col in value_cols})
        
        frequency_mapping[name] = frequency
        processed_data[name] = df
    
    return processed_data, frequency_mapping

def merge_datasets(processed_data: Dict[str, pd.DataFrame], 
                  frequency_mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Merge all datasets into a single DataFrame with complete date coverage.
    
    Args:
        processed_data: Dictionary of processed datasets
        frequency_mapping: Dictionary mapping dataset names to frequencies
        
    Returns:
        Merged DataFrame
    """
    if not processed_data:
        return pd.DataFrame()
    
    # Find global min and max dates across all datasets
    all_dates = []
    for df in processed_data.values():
        if not df.empty and 'date' in df.columns:
            all_dates.extend(df['date'].tolist())
    
    if not all_dates:
        logger.error("No valid dates found in any dataset")
        return pd.DataFrame()
    
    min_date = pd.Timestamp(min(all_dates))
    max_date = pd.Timestamp(max(all_dates))
    
    logger.info(f"Creating date range from {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
    
    # Create base dataframe with full date range at monthly frequency
    base_df = create_date_range_df(min_date, max_date, freq='MS')
    
    # Merge all datasets (all are now in monthly format)
    for name, df in processed_data.items():
        value_cols = [c for c in df.columns if c.startswith(f"{name}_")]
        if value_cols:
            base_df = pd.merge(base_df, df[['date'] + value_cols], on='date', how='left')
            logger.info(f"Merged {len(value_cols)} columns from '{name}'")
    
    return base_df

def export_datasets(connector: DataConnector, output_path: str, 
                    start_date: Optional[str] = None, 
                    end_date: Optional[str] = None) -> None:
    """
    Export all datasets to a single combined CSV file.
    
    Args:
        connector: DataConnector instance
        output_path: Base path for output files
        start_date: Optional start date filter
        end_date: Optional end date filter
    """
    # Fetch all datasets
    logger.info(f"Fetching all datasets from {start_date or 'beginning'} to {end_date or 'latest'}")
    datasets = connector.fetch_all_datasets(start_date, end_date)
    
    if not datasets:
        logger.error("No datasets found.")
        return
    
    # Prepare datasets for merging
    processed_data, frequency_mapping = prepare_datasets(datasets)
    
    # Merge all datasets
    combined_df = merge_datasets(processed_data, frequency_mapping)
    
    if combined_df.empty:
        logger.error("Failed to create combined dataset")
        return
    
    # Sort by date
    combined_df = combined_df.sort_values('date')
    
    # Create output directory if it doesn't exist
    import os
    output_dir = os.path.join(os.getcwd(), "exports")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving output to directory: {output_dir}")
    
    # Export to CSV with timestamp in filename
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_path}_{timestamp}.csv"
    full_output_path = os.path.join(output_dir, filename)
    
    try:
        combined_df.to_csv(full_output_path, index=False)
        logger.info(f"Exported combined dataset to {full_output_path} ({len(combined_df)} records, {len(combined_df.columns)} columns)")
        
        # Log number of non-NaN values per dataset to help with debugging
        for name in frequency_mapping.keys():
            cols = [col for col in combined_df.columns if col.startswith(f"{name}_")]
            if cols:
                non_nan_count = combined_df[cols].count().mean()
                logger.info(f"Dataset '{name}': {non_nan_count:.0f}/{len(combined_df)} non-NaN values ({non_nan_count/len(combined_df)*100:.1f}%)")
    except Exception as e:
        logger.error(f"Error exporting CSV file: {e}")

def main():
    parser = argparse.ArgumentParser(description='Export economic data to CSV')
    parser.add_argument('--output', '-o', default='economic_data', help='Output CSV file base name (without extension)')
    parser.add_argument('--output-dir', '-dir', default='exports', help='Directory to save the exported CSV file')
    parser.add_argument('--start-date', '-s', help='Start date filter (YYYY-MM-DD)')
    parser.add_argument('--end-date', '-e', help='End date filter (YYYY-MM-DD)')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize connector
        connector = DataConnector()
        
        # Export data - create the output directory if specified
        if args.output_dir:
            import os
            output_dir = os.path.join(os.getcwd(), args.output_dir)
            os.makedirs(output_dir, exist_ok=True)
            
        export_datasets(connector, args.output, args.start_date, args.end_date)
        
    except Exception as e:
        logger.error(f"Error exporting data: {e}", exc_info=True)

if __name__ == '__main__':
    main()