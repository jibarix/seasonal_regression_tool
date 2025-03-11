"""
Data module for loading and processing time series data with seasonal components.
Provides functionality to handle different data frequencies and formats.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Union, Optional, Tuple
from pathlib import Path

# Setup logging
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Class for loading and preprocessing time series data from CSV files.
    Handles data of different frequencies (monthly, quarterly) and formats.
    """
    
    def __init__(self, date_col: str = 'date'):
        """
        Initialize the DataLoader with configuration.
        
        Args:
            date_col: Name of the date column in the data
        """
        self.date_col = date_col
        
    def load_data(self, file_path: Union[str, Path], 
                  start_date: Optional[str] = None, 
                  end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from a CSV file with optional date filtering.
        
        Args:
            file_path: Path to the CSV file
            start_date: Optional start date for filtering (YYYY-MM-DD)
            end_date: Optional end date for filtering (YYYY-MM-DD)
            
        Returns:
            DataFrame with the loaded data
        """
        logger.info(f"Loading data from {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            
            # Ensure date column is datetime
            if self.date_col in df.columns:
                df[self.date_col] = pd.to_datetime(df[self.date_col])
                
                # Apply date filters if provided
                if start_date:
                    start_date = pd.to_datetime(start_date)
                    df = df[df[self.date_col] >= start_date]
                    
                if end_date:
                    end_date = pd.to_datetime(end_date)
                    df = df[df[self.date_col] <= end_date]
                
                # Sort by date
                df = df.sort_values(self.date_col)
            else:
                logger.warning(f"Date column '{self.date_col}' not found in the data")
            
            logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
            
    def detect_frequency(self, df: pd.DataFrame) -> str:
        """
        Automatically detect the frequency of the time series data.
        
        Args:
            df: DataFrame with datetime index or column
            
        Returns:
            Detected frequency ('monthly', 'quarterly', 'annual', or 'unknown')
        """
        if df.empty or len(df) < 2:
            return 'unknown'
        
        # Get date column as series
        if self.date_col in df.columns:
            dates = pd.to_datetime(df[self.date_col]).sort_values()
        elif isinstance(df.index, pd.DatetimeIndex):
            dates = df.index.sort_values()
        else:
            logger.warning("No datetime column or index found for frequency detection")
            return 'unknown'
        
        # Calculate the difference between consecutive dates
        date_diffs = dates.diff().dropna()
        
        if date_diffs.empty:
            return 'unknown'
        
        # Use median to avoid outliers
        median_days = date_diffs.dt.days.median()
        
        # Classification based on median days difference
        if 25 <= median_days <= 32:
            return 'monthly'
        elif 85 <= median_days <= 95:
            return 'quarterly'
        elif 350 <= median_days <= 380:
            return 'annual'
        else:
            # Try to infer from pattern of months
            if len(dates) >= 4:
                months = dates.dt.month.tolist()
                month_diffs = np.diff(months)
                
                # Check if the pattern is consistent with quarterly data
                # (differences of 3 months)
                if all(diff in [3, -9] for diff in month_diffs):
                    return 'quarterly'
                # Check if all dates are the same month (annual data)
                elif len(set(months)) == 1:
                    return 'annual'
                    
            logger.warning(f"Could not clearly determine frequency. Median days: {median_days}")
            return 'unknown'
            
    def align_to_monthly(self, df: pd.DataFrame, frequency: str) -> pd.DataFrame:
        """
        Align data to monthly frequency, distributing values for lower frequencies.
        
        Args:
            df: DataFrame to align
            frequency: Current frequency of the data ('quarterly', 'annual')
            
        Returns:
            DataFrame aligned to monthly frequency
        """
        if frequency == 'monthly':
            return df.copy()
            
        if self.date_col not in df.columns:
            logger.error(f"Date column '{self.date_col}' not found")
            return df.copy()
            
        logger.info(f"Aligning {frequency} data to monthly frequency")
        
        # Create a copy to avoid modifying the original
        aligned_df = df.copy()
        aligned_df[self.date_col] = pd.to_datetime(aligned_df[self.date_col])
        
        if frequency == 'quarterly':
            # For quarterly data, map to the three months it represents
            result_rows = []
            
            for _, row in aligned_df.iterrows():
                quarter_date = row[self.date_col]
                month = quarter_date.month
                year = quarter_date.year
                
                # Determine which quarter this date represents
                # and map to corresponding three months
                if month == 1:  # Q4 of previous year
                    months = [(year-1, 10), (year-1, 11), (year-1, 12)]
                elif month == 4:  # Q1
                    months = [(year, 1), (year, 2), (year, 3)]
                elif month == 7:  # Q2
                    months = [(year, 4), (year, 5), (year, 6)]
                elif month == 10:  # Q3
                    months = [(year, 7), (year, 8), (year, 9)]
                else:
                    logger.warning(f"Unexpected month in quarterly data: {month}")
                    continue
                
                # Create a row for each month
                for month_year in months:
                    new_row = row.copy()
                    new_row[self.date_col] = pd.Timestamp(year=month_year[0], month=month_year[1], day=1)
                    result_rows.append(new_row)
            
            if result_rows:
                result_df = pd.DataFrame(result_rows)
                # Sort by date and reset index
                result_df = result_df.sort_values(self.date_col).reset_index(drop=True)
                return result_df
            else:
                logger.warning("No rows generated during quarterly to monthly conversion")
                return aligned_df
                
        elif frequency == 'annual':
            # For annual data, distribute to all 12 months
            result_rows = []
            
            for _, row in aligned_df.iterrows():
                annual_date = row[self.date_col]
                year = annual_date.year
                
                # Create a row for each month in the year
                for month in range(1, 13):
                    new_row = row.copy()
                    new_row[self.date_col] = pd.Timestamp(year=year, month=month, day=1)
                    result_rows.append(new_row)
            
            if result_rows:
                result_df = pd.DataFrame(result_rows)
                # Sort by date and reset index
                result_df = result_df.sort_values(self.date_col).reset_index(drop=True)
                return result_df
            else:
                logger.warning("No rows generated during annual to monthly conversion")
                return aligned_df
        else:
            logger.warning(f"Unsupported frequency for alignment: {frequency}")
            return aligned_df
            
    def merge_datasets(self, datasets: Dict[str, pd.DataFrame], 
                       align_frequency: str = 'monthly') -> pd.DataFrame:
        """
        Merge multiple datasets with different frequencies into a single DataFrame.
        
        Args:
            datasets: Dictionary of dataset name to DataFrame
            align_frequency: Target frequency for alignment
            
        Returns:
            Merged DataFrame with consistent frequency
        """
        if not datasets:
            logger.warning("No datasets provided for merging")
            return pd.DataFrame()
            
        logger.info(f"Merging {len(datasets)} datasets to {align_frequency} frequency")
        
        # Detect frequency and align each dataset
        aligned_datasets = {}
        
        for name, df in datasets.items():
            if df.empty:
                logger.warning(f"Dataset '{name}' is empty, skipping")
                continue
                
            # Detect frequency
            frequency = self.detect_frequency(df)
            logger.info(f"Dataset '{name}' has {frequency} frequency")
            
            # Align to target frequency if needed
            if frequency != align_frequency and frequency != 'unknown':
                if align_frequency == 'monthly':
                    aligned_df = self.align_to_monthly(df, frequency)
                    aligned_datasets[name] = aligned_df
                else:
                    logger.warning(f"Alignment to {align_frequency} not yet implemented")
                    aligned_datasets[name] = df
            else:
                aligned_datasets[name] = df
        
        # Find global date range
        all_dates = []
        for df in aligned_datasets.values():
            if self.date_col in df.columns:
                all_dates.extend(pd.to_datetime(df[self.date_col]))
        
        if not all_dates:
            logger.error("No valid dates found across datasets")
            return pd.DataFrame()
            
        min_date = min(all_dates)
        max_date = max(all_dates)
        
        # Create base DataFrame with complete date range
        if align_frequency == 'monthly':
            date_range = pd.date_range(start=min_date, end=max_date, freq='MS')
        elif align_frequency == 'quarterly':
            date_range = pd.date_range(start=min_date, end=max_date, freq='QS')
        else:
            date_range = pd.date_range(start=min_date, end=max_date, freq='MS')
            
        base_df = pd.DataFrame({self.date_col: date_range})
        
        # Add time variables
        base_df = self.add_time_variables(base_df)
        
        # Merge all datasets
        for name, df in aligned_datasets.items():
            if self.date_col not in df.columns:
                logger.warning(f"Date column missing in dataset '{name}', skipping")
                continue
                
            # Rename value columns to include dataset name
            # Find numeric columns (excluding date and time variables)
            time_vars = ['year', 'month', 'quarter', 'month_name', 'period', 
                        'month_end', 'quarter_end', 'Q1', 'Q2', 'Q3', 'Q4']
            
            value_cols = [col for col in df.columns 
                          if col != self.date_col and col not in time_vars]
            
            # Only keep value columns for merge
            merge_cols = [self.date_col] + value_cols
            
            # Merge with base DataFrame
            base_df = pd.merge(base_df, df[merge_cols], on=self.date_col, how='left')
            
        return base_df
    
    def add_time_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-related variables to a DataFrame for time series analysis,
        only if they don't already exist in the data.
        
        Args:
            df: DataFrame containing date column
            
        Returns:
            DataFrame with additional time variables if needed
        """
        if df.empty:
            return df
            
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Ensure date column is datetime
        result[self.date_col] = pd.to_datetime(result[self.date_col])
        
        # Only add year if it doesn't exist
        if 'year' not in result.columns:
            result['year'] = result[self.date_col].dt.year
        
        # Only add month if it doesn't exist
        if 'month' not in result.columns:
            result['month'] = result[self.date_col].dt.month
        
        # Only add month_name if it doesn't exist
        if 'month_name' not in result.columns:
            result['month_name'] = result[self.date_col].dt.strftime('%b')
        
        # Only add month_end if it doesn't exist
        if 'month_end' not in result.columns:
            result['month_end'] = result[self.date_col] + pd.offsets.MonthEnd(0)
        
        # Only add quarter if it doesn't exist
        if 'quarter' not in result.columns:
            result['quarter'] = result[self.date_col].dt.quarter
        
        # Only add quarter_end if it doesn't exist
        if 'quarter_end' not in result.columns:
            result['quarter_end'] = result[self.date_col] + pd.offsets.QuarterEnd(0)
        
        # Add dummy variables for quarters only if they don't exist
        quarter_cols_exist = all(f'Q{q}' in result.columns for q in range(1, 5))
        if not quarter_cols_exist:
            for q in range(1, 5):
                result[f'Q{q}'] = (result['quarter'] == q).astype(int)
        
        # Only add period if it doesn't exist
        if 'period' not in result.columns:
            result['period'] = result[self.date_col].dt.strftime('%b %Y')
        
        return result
    
    def trim_to_common_date_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Trim dataset to the common date range where data is available across columns.
        
        Args:
            df: DataFrame with date column and possibly missing values
            
        Returns:
            DataFrame trimmed to the common date range
        """
        # Input validation
        if df is None:
            logger.error("Cannot trim date range: DataFrame is None")
            return df
            
        if df.empty:
            logger.warning("Cannot trim date range: DataFrame is empty")
            return df.copy()
            
        if self.date_col not in df.columns:
            logger.warning(f"Cannot trim date range: date column '{self.date_col}' not found")
            return df.copy()
            
        # Ensure date column is datetime
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        
        # Get numeric columns (excluding date-related columns)
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # Skip if no numeric columns found
        if not numeric_cols:
            logger.warning("No numeric columns found for trimming")
            return df.copy()
            
        date_related_cols = ['year', 'month', 'quarter', 'month_end', 'quarter_end', 
                        'Q1', 'Q2', 'Q3', 'Q4', 'day_of_year', 'day_of_month']
        numeric_cols = [col for col in numeric_cols if col not in date_related_cols]
        
        # Skip if all numeric columns are date-related
        if not numeric_cols:
            logger.warning("No non-date-related numeric columns found for trimming")
            return df.copy()
        
        # Find the first valid date for each column
        first_valid_dates = {}
        last_valid_dates = {}
        
        for col in numeric_cols:
            # Skip columns that are all NaN
            if df[col].isna().all():
                continue
                
            # Find first non-NaN value - use try/except to handle any issues
            try:
                first_valid_idx = df[col].first_valid_index()
                last_valid_idx = df[col].last_valid_index()
                
                if first_valid_idx is not None and last_valid_idx is not None:
                    first_valid_dates[col] = df.loc[first_valid_idx, self.date_col]
                    last_valid_dates[col] = df.loc[last_valid_idx, self.date_col]
            except Exception as e:
                logger.warning(f"Error finding valid dates for column {col}: {e}")
                continue
        
        if not first_valid_dates or not last_valid_dates:
            logger.warning("No valid data ranges found in any column")
            return df.copy()
            
        try:
            # Determine the latest start date and earliest end date
            latest_start = max(first_valid_dates.values())
            earliest_end = min(last_valid_dates.values())
            
            # Check if the range makes sense
            if latest_start > earliest_end:
                logger.warning(f"Invalid date range: {latest_start} > {earliest_end}")
                return df.copy()
            
            # Trim the DataFrame to this range
            trimmed_df = df[(df[self.date_col] >= latest_start) & (df[self.date_col] <= earliest_end)].copy()
            
            logger.info(f"Trimmed data from {df[self.date_col].min()} - {df[self.date_col].max()} to {latest_start} - {earliest_end}")
            logger.info(f"Rows reduced from {len(df)} to {len(trimmed_df)}")
            
            # Ensure we're not returning an empty dataframe
            if len(trimmed_df) == 0:
                logger.warning("Trimming resulted in empty DataFrame. Returning original data.")
                return df.copy()
                
            return trimmed_df
            
        except Exception as e:
            logger.error(f"Error during date trimming: {e}")
            return df.copy()
    
    def handle_missing_values(self, df: pd.DataFrame, method: str = 'mean',
                             fill_groups: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Handle missing values in the DataFrame using the specified method.
        
        Args:
            df: DataFrame with missing values
            method: Method to handle missing values ('mean', 'median', 'forward', 'backward', 'drop')
            fill_groups: Optional list of columns to group by when filling values
            
        Returns:
            DataFrame with missing values handled
        """
        if df.empty:
            return df
            
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Get numeric columns
        numeric_cols = result.select_dtypes(include=['number']).columns.tolist()
        
        # Remove date column if present
        if self.date_col in numeric_cols:
            numeric_cols.remove(self.date_col)
            
        # Handle missing values based on method
        if method == 'drop':
            # Drop rows with missing values
            orig_len = len(result)
            result = result.dropna(subset=numeric_cols)
            dropped = orig_len - len(result)
            logger.info(f"Dropped {dropped} rows with missing values")
        
        elif method in ['mean', 'median']:
            # Fill with mean or median
            for col in numeric_cols:
                missing = result[col].isna().sum()
                if missing > 0:
                    if fill_groups and all(g in result.columns for g in fill_groups):
                        # Fill by group (e.g., by month or quarter)
                        grouped = result.groupby(fill_groups)
                        
                        if method == 'mean':
                            result[col] = result[col].fillna(grouped[col].transform('mean'))
                        else:  # median
                            result[col] = result[col].fillna(grouped[col].transform('median'))
                            
                        # Fill any remaining NaNs with overall mean/median
                        remaining = result[col].isna().sum()
                        if remaining > 0:
                            if method == 'mean':
                                result[col] = result[col].fillna(result[col].mean())
                            else:  # median
                                result[col] = result[col].fillna(result[col].median())
                                
                            logger.info(f"Filled {missing} missing values in '{col}' ({remaining} outside groups)")
                    else:
                        # Fill with overall mean/median
                        if method == 'mean':
                            result[col] = result[col].fillna(result[col].mean())
                        else:  # median
                            result[col] = result[col].fillna(result[col].median())
                            
                        logger.info(f"Filled {missing} missing values in '{col}'")
        
        elif method == 'forward':
            # Forward fill
            for col in numeric_cols:
                missing = result[col].isna().sum()
                if missing > 0:
                    result[col] = result[col].fillna(method='ffill')
                    remaining = result[col].isna().sum()
                    
                    # Handle any remaining NaNs at the beginning
                    if remaining > 0:
                        result[col] = result[col].fillna(method='bfill')
                        
                    logger.info(f"Forward filled {missing} missing values in '{col}'")
        
        elif method == 'backward':
            # Backward fill
            for col in numeric_cols:
                missing = result[col].isna().sum()
                if missing > 0:
                    result[col] = result[col].fillna(method='bfill')
                    remaining = result[col].isna().sum()
                    
                    # Handle any remaining NaNs at the end
                    if remaining > 0:
                        result[col] = result[col].fillna(method='ffill')
                        
                    logger.info(f"Backward filled {missing} missing values in '{col}'")
        
        else:
            logger.warning(f"Unknown missing value handling method: {method}")
        
        return result
    
    def split_data(self, df: pd.DataFrame, 
                  test_size: float = 0.2,
                  validation_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split time series data into training, validation, and test sets.
        
        Args:
            df: DataFrame to split
            test_size: Proportion of data to use for testing
            validation_size: Proportion of data to use for validation
            
        Returns:
            Tuple of (train_df, validation_df, test_df)
        """
        if df.empty:
            return df.copy(), df.copy(), df.copy()
            
        # Ensure data is sorted by date
        if self.date_col in df.columns:
            df = df.sort_values(self.date_col).reset_index(drop=True)
            
        n = len(df)
        test_idx = int(n * (1 - test_size))
        validation_idx = int(n * (1 - test_size - validation_size))
        
        train_df = df.iloc[:validation_idx].copy()
        validation_df = df.iloc[validation_idx:test_idx].copy()
        test_df = df.iloc[test_idx:].copy()
        
        logger.info(f"Split data into train ({len(train_df)} rows), "
                    f"validation ({len(validation_df)} rows), "
                    f"and test ({len(test_df)} rows)")
        
        return train_df, validation_df, test_df


# Helper functions

def list_available_datasets(directory: Union[str, Path] = 'exports') -> List[str]:
    """
    List all CSV files in the specified directory.
    
    Args:
        directory: Directory to search for CSV files
        
    Returns:
        List of CSV file paths
    """
    directory = Path(directory)
    
    if not directory.exists():
        logger.warning(f"Directory '{directory}' does not exist")
        return []
        
    csv_files = list(directory.glob('*.csv'))
    return [str(file) for file in csv_files]