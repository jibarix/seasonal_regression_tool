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
| auto_sales_sales | Automobile and light truck sales | Monthly | Units | EDB Puerto Rico | I_AUTO.XLS (Sheet: AS01) |
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

## Important Note on GDP Indicators

The dataset includes two different GDP-related indicators that should not be confused:

1. **Real GDP (GDPC1)**: This dataset contains the absolute value of Real Gross Domestic Product in Billions of Chained 2017 Dollars, Seasonally Adjusted Annual Rate. It represents the actual size of the economy in real terms (adjusted for inflation). In the database, this is stored in the `value` column of the `real_gdp` table.

2. **GDP Now (GDPNOW)**: This dataset contains the GDPNow forecast from the Federal Reserve Bank of Atlanta, expressed as a Percent Change at Annual Rate, Seasonally Adjusted Annual Rate. It represents the forecasted growth rate of the economy. In the database, this is stored in the `forecast` column of the `gdp_now_forecast` table.

These two indicators measure very different aspects of economic performance and cannot be directly compared. Real GDP measures the absolute size of the economy, while GDP Now measures the expected rate of change. For time series analysis and forecasting, it's essential to use these indicators correctly according to their units and what they represent.

## Data Characteristics

### Time Range
- The dataset spans from January 2014 to March 2025
- Most indicators have consistent monthly observations, with some gaps during economic disruptions

### Data Quality
- Some indicators have seasonal patterns that must be accounted for in modeling
- Economic shocks (like the COVID-19 pandemic in 2020) create outliers in many series
- Some indicators have missing values, particularly in earlier periods
- The data exhibits varying degrees of stationarity across different series

### Potential Use Cases
- Forecasting economic indicators for Puerto Rico
- Analyzing relationships between U.S. national and Puerto Rico regional economic factors
- Studying the impact of external shocks on various economic sectors
- Building composite indicators for economic activity

### Preprocessing Considerations
- Seasonal adjustment may be necessary for many indicators
- Transformation (such as log transformations) are recommended for some indicators with exponential patterns
- Outlier detection and treatment should be considered, especially around known economic disruptions
- Indicators have different scales and units, requiring normalization for certain types of analysis