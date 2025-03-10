# Economic Indicators Dataset Analysis

## Data Overview

The repository contains a data pipeline that collects and processes economic indicators from multiple sources, with a focus on Puerto Rico's economy. The data includes monthly and quarterly economic indicators spanning various sectors.

### Data Sources

1. **Economic Development Bank (EDB) of Puerto Rico**
   - Monthly and quarterly economic indicators for Puerto Rico
   - Data format: Excel files published on EDB website
   - URL: https://www.bde.pr.gov/BDE/PREDDOCS/
   - Organization: Fiscal year format (July-June)
   - Update frequency: Monthly for most indicators

2. **Federal Reserve Economic Data (FRED)**
   - U.S. national economic indicators via API
   - Base URL: https://api.stlouisfed.org/fred/series/observations
   - Requires API key stored as environment variable
   - Update frequency: Varies by indicator

3. **NYU Stern School of Business**
   - Equity Risk Premium data
   - URL: https://pages.stern.nyu.edu/~adamodar/pc/implprem/ERPbymonth.xlsx
   - Update frequency: Monthly

## Dataset Structure

### Complete Dataset Catalog

| Dataset | Description | Frequency | Units | Source | File Name/Series ID |
|---------|-------------|-----------|-------|--------|---------------------|
| auto_sales | Automobile and light truck sales | Monthly | Units | EDB Puerto Rico | I_AUTO.XLS (Sheet: AS01) |
| bankruptcies | Bankruptcy filings | Monthly | Count | EDB Puerto Rico | I_BANKRUPT.XLS (Sheet: BAN01) |
| cement_production | Cement production | Monthly | 94lb. bags (000s) | EDB Puerto Rico | I_CEMENT.XLS (Sheet: CD01) |
| electricity_consumption | Electric energy consumption | Monthly | mm kWh | EDB Puerto Rico | I_ENERGY.XLS (Sheet: EEC01) |
| gas_price | Gasoline average retail price | Monthly | Dollars per gallon | EDB Puerto Rico | I_GAS.XLS (Sheet: GAS01) |
| gas_consumption | Gasoline consumption | Monthly | Million gallons | EDB Puerto Rico | I_GAS.XLS (Sheet: GAS02) |
| labor_participation | Labor force participation rate | Monthly | Percentage | EDB Puerto Rico | I_LABOR.XLS (Sheet: LF03) |
| unemployment_rate | Unemployment rate | Monthly | Percentage | EDB Puerto Rico | I_LABOR.XLS (Sheet: LF08) |
| employment_rate | Employment rate | Monthly | Percentage | EDB Puerto Rico | I_LABOR.XLS (Sheet: LF09) |
| unemployment_claims | Unemployment insurance initial file claims | Monthly | 000s | EDB Puerto Rico | I_LABOR.XLS (Sheet: LF10) |
| trade_employment | Payroll employment in trade sector | Monthly | 000s | EDB Puerto Rico | I_PAYROLL.XLS (Sheet: PE05) |
| consumer_price_index | Consumer price index (Dec. 2006=100) | Monthly | Index | EDB Puerto Rico | I_PRICE.XLS (Sheet: CPI01) |
| transportation_price_index | Transportation price index (Dec. 2006=100) | Monthly | Index | EDB Puerto Rico | I_PRICE.XLS (Sheet: CPI05) |
| retail_sales | Total retail store sales | Monthly | Million $ | EDB Puerto Rico | I_RETAIL.XLS (Sheet: RS01) |
| imports | External trade imports | Monthly | Million $ | EDB Puerto Rico | I_TRADE.XLS (Sheet: ET05) |
| federal_funds_rate | Federal funds effective rate | Monthly | Percentage | FRED | DFF |
| auto_manufacturing_orders | Manufacturers' new orders for motor vehicles and parts | Monthly | Millions $ | FRED | AMVPNO |
| used_car_retail_sales | Retail sales from used car dealers | Monthly | Millions $ | FRED | MRTSSM44112USN |
| domestic_auto_inventories | Domestic auto inventories | Monthly | Thousands of units | FRED | AUINSA |
| domestic_auto_production | Domestic auto production | Monthly | Thousands of units | FRED | DAUPSA |
| liquidity_credit_facilities | Liquidity and credit facilities loans | Monthly | Value | FRED | WLCFLL |
| semiconductor_manufacturing_units | Industrial production of semiconductor components | Monthly | Index | FRED | IPG3344S |
| aluminum_new_orders | Manufacturers' new orders for aluminum and nonferrous metal products | Monthly | Value | FRED | AANMNO |
| real_gdp | Real Gross Domestic Product | Quarterly | Billions of Chained 2017 Dollars, Seasonally Adjusted Annual Rate | FRED | GDPC1 |
| gdp_now_forecast | GDPNow forecast from Federal Reserve Bank of Atlanta | Quarterly | Percent Change at Annual Rate, Seasonally Adjusted Annual Rate | FRED | GDPNOW |
| tbond_rate | Treasury bond rate | Monthly | Percentage | NYU Stern | ERPbymonth.xlsx (Sheet: Historical ERP) |
| erp_sustainable | Equity risk premium (sustainable payout) | Monthly | Percentage | NYU Stern | ERPbymonth.xlsx (Sheet: Historical ERP) |
| erp_t12m | Equity risk premium (T12M) | Monthly | Percentage | NYU Stern | ERPbymonth.xlsx (Sheet: Historical ERP) |

## Data Collection Details

| Process Step | Description | Implementation | Notes |
|--------------|-------------|----------------|-------|
| Fetching | Download of raw data from source | `download_excel()`, FRED API calls | Handles HTTP requests with error checking |
| Extraction | Parsing raw data into structured format | `extract_data()`, specific sheet and cell references | Uses pandas for processing |
| Transformation | Converting to standard schema | `process_data()` | Handles fiscal year conversion, data type conversion |
| Loading | Storing in database | `insert_data()`, `smart_update()` | Uses Supabase PostgreSQL database |
| Revision Tracking | Detecting and recording changes | `smart_update()` in data_tracker.py | Compares new values with existing records |
| Export | Creating analysis-ready datasets | export_data.py | Combines all datasets with date alignment |

## Database Schema

| Table | Primary Key | Date Column | Value Columns | Related Tables | Creation Script |
|-------|-------------|-------------|---------------|----------------|----------------|
| auto_sales | id (SERIAL) | date (DATE) | sales (INTEGER) | data_revisions | config.py (MONTHLY_TABLE_SQL_TEMPLATE) |
| bankruptcies | id (SERIAL) | date (DATE) | filings (INTEGER) | data_revisions | config.py (MONTHLY_TABLE_SQL_TEMPLATE) |
| cement_production | id (SERIAL) | date (DATE) | production (DECIMAL(12,2)) | data_revisions | config.py (MONTHLY_TABLE_SQL_TEMPLATE) |
| electricity_consumption | id (SERIAL) | date (DATE) | consumption (DECIMAL(12,2)) | data_revisions | config.py (MONTHLY_TABLE_SQL_TEMPLATE) |
| gas_price | id (SERIAL) | date (DATE) | price (DECIMAL(12,2)) | data_revisions | config.py (MONTHLY_TABLE_SQL_TEMPLATE) |
| gas_consumption | id (SERIAL) | date (DATE) | consumption (DECIMAL(12,2)) | data_revisions | config.py (MONTHLY_TABLE_SQL_TEMPLATE) |
| labor_participation | id (SERIAL) | date (DATE) | rate (DECIMAL(6,2)) | data_revisions | config.py (PERCENT_TABLE_SQL_TEMPLATE) |
| unemployment_rate | id (SERIAL) | date (DATE) | rate (DECIMAL(6,2)) | data_revisions | config.py (PERCENT_TABLE_SQL_TEMPLATE) |
| employment_rate | id (SERIAL) | date (DATE) | rate (DECIMAL(6,2)) | data_revisions | config.py (PERCENT_TABLE_SQL_TEMPLATE) |
| unemployment_claims | id (SERIAL) | date (DATE) | claims (INTEGER) | data_revisions | config.py (MONTHLY_TABLE_SQL_TEMPLATE) |
| trade_employment | id (SERIAL) | date (DATE) | employment (DECIMAL(12,2)) | data_revisions | config.py (MONTHLY_TABLE_SQL_TEMPLATE) |
| consumer_price_index | id (SERIAL) | date (DATE) | index (DECIMAL(12,2)) | data_revisions | config.py (MONTHLY_TABLE_SQL_TEMPLATE) |
| transportation_price_index | id (SERIAL) | date (DATE) | index (DECIMAL(12,2)) | data_revisions | config.py (MONTHLY_TABLE_SQL_TEMPLATE) |
| retail_sales | id (SERIAL) | date (DATE) | sales (DECIMAL(12,2)) | data_revisions | config.py (MONTHLY_TABLE_SQL_TEMPLATE) |
| imports | id (SERIAL) | date (DATE) | value (DECIMAL(12,2)) | data_revisions | config.py (MONTHLY_TABLE_SQL_TEMPLATE) |
| federal_funds_rate | id (SERIAL) | date (DATE) | rate (DECIMAL(12,3)) | data_revisions | fred_config.py (FRED_TABLE_SQL_TEMPLATE) |
| auto_manufacturing_orders | id (SERIAL) | date (DATE) | orders (DECIMAL(12,2)) | data_revisions | fred_config.py (FRED_TABLE_SQL_TEMPLATE) |
| used_car_retail_sales | id (SERIAL) | date (DATE) | sales (DECIMAL(12,2)) | data_revisions | fred_config.py (FRED_TABLE_SQL_TEMPLATE) |
| domestic_auto_inventories | id (SERIAL) | date (DATE) | inventories (DECIMAL(12,3)) | data_revisions | fred_config.py (FRED_TABLE_SQL_TEMPLATE) |
| domestic_auto_production | id (SERIAL) | date (DATE) | production (DECIMAL(12,1)) | data_revisions | fred_config.py (FRED_TABLE_SQL_TEMPLATE) |
| liquidity_credit_facilities | id (SERIAL) | date (DATE) | facilities (DECIMAL(12,1)) | data_revisions | fred_config.py (FRED_TABLE_SQL_TEMPLATE) |
| semiconductor_manufacturing_units | id (SERIAL) | date (DATE) | units (DECIMAL(12,4)) | data_revisions | fred_config.py (FRED_TABLE_SQL_TEMPLATE) |
| aluminum_new_orders | id (SERIAL) | date (DATE) | orders (DECIMAL(12,1)) | data_revisions | fred_config.py (FRED_TABLE_SQL_TEMPLATE) |
| real_gdp | id (SERIAL) | date (DATE) | value (DECIMAL(12,2)) | data_revisions | fred_config.py (FRED_TABLE_SQL_TEMPLATE) |
| gdp_now_forecast | id (SERIAL) | date (DATE) | forecast (DECIMAL(12,4)) | data_revisions | fred_config.py (FRED_TABLE_SQL_TEMPLATE) |
| equity_risk_premium | id (SERIAL) | date (DATE) | tbond_rate, erp_sustainable, erp_t12m (DECIMAL(6,4)) | data_revisions | nyu_config.py (NYU_STERN_TABLE_SQL) |
| data_revisions | id (SERIAL) | data_date (DATE) | dataset, value_field, old_value, new_value, revision_date | All above tables | data_tracker.py (initialize_revision_table) |
| scraper_metadata | dataset (VARCHAR) | last_run (TIMESTAMP) | N/A | All above tables | models.py (initialize_tables) |

## Important Note on GDP Indicators

The dataset includes two different GDP-related indicators that should not be confused:

1. **Real GDP (GDPC1)**: This dataset contains the absolute value of Real Gross Domestic Product in Billions of Chained 2017 Dollars, Seasonally Adjusted Annual Rate. It represents the actual size of the economy in real terms (adjusted for inflation). In the database, this is stored in the `value` column of the `real_gdp` table.

2. **GDP Now (GDPNOW)**: This dataset contains the GDPNow forecast from the Federal Reserve Bank of Atlanta, expressed as a Percent Change at Annual Rate, Seasonally Adjusted Annual Rate. It represents the forecasted growth rate of the economy. In the database, this is stored in the `forecast` column of the `gdp_now_forecast` table.

These two indicators measure very different aspects of economic performance and cannot be directly compared. Real GDP measures the absolute size of the economy, while GDP Now measures the expected rate of change. For time series analysis and forecasting, it's essential to use these indicators correctly according to their units and what they represent.

## Data Processing Tools

| Tool | Purpose | Key Features | Input/Output |
|------|---------|--------------|--------------|
| main.py | Data collection orchestration | Runs all scrapers, handles errors | Inputs: API keys, URLs; Outputs: Database updates |
| view_data.py | Data exploration | View latest data, dataset summaries, revisions | Inputs: Dataset name; Outputs: Console display, optional plots |
| export_data.py | Create combined datasets | Merges all datasets with date alignment | Inputs: Optional date range; Outputs: CSV file |
| forecast.py | Basic linear regression forecasting | Variable selection, diagnostics | Inputs: CSV data, target variable; Outputs: Model, plots, reports |
| prophet_forecast.py | Advanced time series forecasting | Seasonal decomposition, component analysis | Inputs: CSV data, target variable; Outputs: Forecasts, plots |
| regression_forecast.py | Seasonal pattern analysis | Multiple seasonality representations | Inputs: CSV data, target variable; Outputs: Comparative analysis |

## Data Quality Controls

| Process | Implementation | Purpose | Notes |
|---------|----------------|---------|-------|
| Revision Tracking | data_tracker.smart_update() | Detect and record data changes | Maintains audit trail of all data modifications |
| Float Precision Tolerance | 0.001 threshold | Avoid false positives for revisions | Small rounding differences ignored |
| Schema Validation | Supabase types, pandas conversion | Ensure data type consistency | Automatic numeric conversions with error handling |
| Error Recovery | Exception handling, logging | Prevent pipeline failures | Detailed logging for troubleshooting |
| Data Verification | view_data.py | Manual inspection capabilities | Interactive CLI for data exploration |
| Multicollinearity Detection | VIF analysis | Improve model reliability | Automatic detection in forecasting tools |

This comprehensive overview provides a detailed picture of all datasets, their sources, structure, and the tools used to process them. The consistent level of detail across all tables helps understand the complete data pipeline from collection to analysis. Special attention should be paid to the GDP indicators to ensure proper interpretation based on their different units and meanings.