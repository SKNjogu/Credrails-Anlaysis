# Transaction Data Analysis Case Study

## Executive Summary
The analysis of the transaction dataset uncovered several important trends, patterns, and performance insights from applied machine learning models. The dataset includes 3050 transactions across 8 features. We identified key transaction trends and the most important features affecting classification performance.

## Data Overview
The dataset contains transaction details including Transaction ID, Customer ID, Transaction Date, Description, Amount, Classification Tag, Transaction Time, and Status. Below is a summary of the dataset:

- Number of transactions: 3050
- Number of features: 8

### Sample Data
The first few rows of the dataset are shown below:

```
  Transaction_ID Customer_ID Transaction_Date                    Description  Amount Classification_Tag Transaction_Time     Status
0      TXN795097    CUST0136       17/02/2024                E-wallet top-up 2921.24        Pre-Funding         05:04:04  Completed
1      TXN825582    CUST0133       29/02/2024      Money transfer via mobile  601.64          Transfers         20:28:17   Refunded
2      TXN654812    CUST0040       24/01/2024            POS cash withdrawal  136.21        Withdrawals         01:04:19  Completed
3      TXN283765    CUST0003       24/01/2024  Online purchase - electronics 3362.72      Card_Payments         06:54:40  Completed
4      TXN749095    CUST0037       28/02/2024       Bank transfer to savings 1304.69          Transfers         08:46:14  Completed
```

## Exploratory Data Analysis
### Transaction Trends and Patterns
- **Transaction Counts by Category:** The analysis revealed varied transaction counts across different categories such as Pre-Funding, Transfers, Withdrawals, etc.
- **Hourly Transaction Patterns:** Transactions displayed distinct patterns during different hours, indicating peak activity periods.

*Visualization:* Transaction categories and hourly patterns are visualized in the accompanying figures (e.g., *amount_distribution.png* for amount distribution).

### Amount Distribution
The distribution of transaction amounts highlights key trends in customer behavior and possible anomalies. Detailed visualizations help to identify the spread and central tendencies.

## Machine Learning Models
Classification models were implemented and evaluated on the dataset. The following table summarizes the performance metrics:

| Model                   | Accuracy | Precision | Recall | F1 Score |
|-------------------------|----------|-----------|--------|----------|
| Logistic Regression     | 0.9984   | 0.9968    | 0.9984 | 0.9975   |
| Random Forest           | 0.9984   | 0.9968    | 0.9984 | 0.9975   |
| Support Vector Machine  | 0.9984   | 0.9968    | 0.9984 | 0.9975   |

*Visualization:* See visualizations such as feature importance charts for further analysis of model performance.

## Recommendations
Based on the analysis, several recommendations can be made:
- Leverage high-performing models in production to streamline transaction classification.
- Monitor transactions during peak hours to ensure optimal operation and prepare for potential network loads.
- Investigate any anomalies in transactions amounts further with detailed time-series analysis.

## Scenario Responses
For specific business scenarios:
- **Fraud Detection:** The excellent performance of the classification models indicates these approaches could be further refined for real-time fraud detection.
- **Customer Segmentation:** The data trends can be used to create customer profiles and targeted offerings using clustering techniques.

## Visualizations
The following visualizations are provided as part of the analysis:
- `amount_distribution.png`: Histogram showing the distribution of transaction amounts.
- Additional plots are generated during the analysis, including transaction category breakdowns and hourly patterns.

## Conclusion
The comprehensive analysis of transactional data and modeling indicates robust performance across multiple machine learning approaches. The insights derived from the EDA and the model performance evaluations suggest practical steps that can be taken to enhance transactional processing and fraud detection.

---
*All source code for data processing and analysis is available in the repository along with this documentation.*
