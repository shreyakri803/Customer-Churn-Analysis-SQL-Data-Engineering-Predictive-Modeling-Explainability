-- Training view
DROP VIEW IF EXISTS vw_ChurnData;
CREATE VIEW vw_ChurnData AS
SELECT *
FROM prod_Churn
WHERE Customer_Status IN ('Churned','Stayed');

-- Joined view
DROP VIEW IF EXISTS vw_JoinData;
CREATE VIEW vw_JoinData AS
SELECT *
FROM prod_Churn
WHERE Customer_Status = 'Joined';

-- (These two can stay as plain SELECTs — they don't create objects)
-- Portfolio churn rate
SELECT COUNT(CASE WHEN Customer_Status='Churned' THEN 1 END)*1.0/COUNT(*) AS churn_rate
FROM prod_Churn;

-- Feature engineering sample
SELECT
  Customer_ID, Age, Tenure_in_Months, Monthly_Charge, Total_Revenue,
  CASE WHEN Tenure_in_Months<=6 THEN 'New'
       WHEN Tenure_in_Months BETWEEN 7 AND 24 THEN 'Intermediate'
       ELSE 'Long-Term' END AS Tenure_Bucket,
  CASE WHEN Monthly_Charge<30 THEN 'Low'
       WHEN Monthly_Charge BETWEEN 30 AND 70 THEN 'Medium'
       ELSE 'High' END AS Billing_Segment,
  CASE WHEN Tenure_in_Months>0 THEN Total_Revenue/NULLIF(Tenure_in_Months,0) END AS Avg_Revenue_Per_Month
FROM prod_Churn;
