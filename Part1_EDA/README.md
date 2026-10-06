# Part 1: Data Ingestion & Exploratory Data Analysis (EDA)

## Dataset Summary
- **Source:** Telco Customer Churn (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)
- **Total Records:** 7,043 rows
- **Features:** 20 predictor columns + 1 target (`Churn`)

## Cleaning & Preprocessing Steps
1. **Missing Value Handling:** The `TotalCharges` column contained blank string spaces `" "` for customers with `tenure = 0`. These were converted to numeric NaN values and imputed using `MonthlyCharges`.
2. **Feature Removal:** Removed `customerID` as it is a non-informative identifier.
3. **Data Types:** Converted `TotalCharges` from object to float.

## Key Insights from EDA
1. **Target Class Imbalance:** Approximately 26.5% of customers in the dataset have churned (`Churn = Yes`), while 73.5% remained.
2. **Tenure Effect:** Customers with shorter tenure are significantly more likely to churn.
3. **Monthly Charges:** Higher monthly charges positively correlate with higher churn rates.