# Customer Churn Analysis: SQL Data Engineering + Predictive Modeling + Explainability
### Author: **Shreya Kumari**

This project delivers an end-to-end **Customer Churn Analysis solution** combining  
**SQL-based data engineering**, **machine learning**, and **model explainability** in a clean, production-style pipeline.  


## 📌 Executive Summary

This solution enables organizations to:

- Process raw customer data using an **SQL-driven data preparation pipeline**
- Train a churn prediction model using a **configurable ML pipeline**
- Perform batch predictions (CSV or SQL-based)
- Generate **SHAP explainability** for business stakeholders
- Connect results to a **Power BI dashboard** for insights and decision-making

The model delivers:

✅ **AUC:** 0.8854  
✅ **F1 Score:** 0.7033  
✅ **Threshold:** 0.50  

---

# 🏗️ Solution Architecture

```

Customer_Data.csv → SQLite (churn.db) → SQL Views
↓                     ↓
Feature Pipeline   ───►  vw_ChurnData  (for model training)
└──► vw_JoinData   (for scoring new customers)

Config.yaml → ML Pipeline (Sklearn: Imputation + OneHotEncoding + Logistic Regression)

Train → Model.pkl
Predict → Predictions.csv / SQL table (predictions)
Explain → SHAP global importance (reports/)

```

---

# 🗂️ Project Structure

```

.
├─ config.yaml
├─ SQLQueries.sql
├─ churn.db
├─ Customer_Data.csv
├─ Predictions.csv
├─ artifacts/
│  └─ model.pkl
├─ reports/
│  └─ global_importance.png
└─ src/
├─ config.py
├─ data_sql.py
├─ features.py
├─ logging_utils.py
├─ train.py
├─ predict.py
└─ explain.py

````

---

# 🧰 Tech Stack

| Layer | Tools |
|------|-------|
| Data Storage | SQLite (via SQLQueries.sql) |
| Data Processing | Python, Pandas, SQL |
| ML Framework | Scikit-learn |
| Explainability | SHAP |
| Visualization | Power BI |
| Pipeline Orchestration | CLI (`cli.py`) |

---

# 🗃️ SQL Data Engineering

SQL transformations are stored in **`SQLQueries.sql`** and automatically applied during training.

### ✅ Training View
```sql
DROP VIEW IF EXISTS vw_ChurnData;
CREATE VIEW vw_ChurnData AS
SELECT * FROM prod_Churn
WHERE Customer_Status IN ('Churned','Stayed');
````

### ✅ New Customer View

```sql
DROP VIEW IF EXISTS vw_JoinData;
CREATE VIEW vw_JoinData AS
SELECT * FROM prod_Churn
WHERE Customer_Status = 'Joined';
```

### ✅ Portfolio Churn Rate

```sql
SELECT
COUNT(CASE WHEN Customer_Status='Churned' THEN 1 END)*1.0/COUNT(*) AS churn_rate
FROM prod_Churn;
```

### ✅ Feature Engineering (SQL)

```sql
CASE WHEN Tenure_in_Months<=6 THEN 'New'
     WHEN Tenure_in_Months BETWEEN 7 AND 24 THEN 'Intermediate'
     ELSE 'Long-Term' END AS Tenure_Bucket
```

SQL views are materialized inside **`churn.db`** when training runs.

---

# ⚙️ Configure the Pipeline

All parameters are set in **`config.yaml`**:

```yaml
use_sql: true
db_path: "churn.db"
sql:
  table_name: "prod_Churn"
  view_churn: "vw_ChurnData"
  view_joined: "vw_JoinData"
---


# 🚀 How to Run (CLI Commands)

### ✅ 1. Train the Model (SQL + CSV auto-handled)

```bash
python cli.py train
```

### ✅ 2. Predict All Customers

```bash
python cli.py predict
```

### ✅ 3. Predict Only Joined Customers (from SQL View)

```bash
python cli.py predict --joined
```

### ✅ 4. Predict + Save Back into SQL Table

```bash
python cli.py predict --joined --sql-save
```

### ✅ 5. Explain Model (SHAP)

```bash
python cli.py explain
```

Produces:

```
reports/global_importance.png
---


# 📂 **Data Description**

This project uses a structured customer dataset representing a telecommunications subscription business. The dataset captures key demographic, billing, service usage, and contract-level information required to model churn behavior.

The data is loaded from **Customer_Data.csv** and ingested into a **SQLite database (churn.db)** as the base table **prod_Churn**, where SQL transformations and views are applied.

---

## 🧾 **Dataset Overview**

| Column Name                                         | Description                                        |
| --------------------------------------------------- | -------------------------------------------------- |
| **Customer_ID**                                     | Unique identifier for each customer                |
| **Gender**                                          | Male / Female                                      |
| **Age**                                             | Customer age                                       |
| **Married**                                         | Marital status (Yes/No)                            |
| **State**                                           | Customer’s home state                              |
| **Number_of_Referrals**                             | How many people the customer referred              |
| **Tenure_in_Months**                                | Months since customer joined                       |
| **Value_Deal**                                      | Promotional deal or offer assigned                 |
| **Phone_Service**                                   | Whether customer has active phone service          |
| **Multiple_Lines**                                  | Single vs multiple phone lines                     |
| **Internet_Service**                                | Yes/No — Whether customer has internet service     |
| **Internet_Type**                                   | DSL / Fiber Optic / Cable                          |
| **Online_Security**                                 | Whether customer has online security add-on        |
| **Online_Backup**                                   | Online backup service availability                 |
| **Device_Protection_Plan**                          | Device protection add-on                           |
| **Premium_Support**                                 | Premium tech support                               |
| **Streaming_TV, Streaming_Movies, Streaming_Music** | Streaming services subscribed                      |
| **Unlimited_Data**                                  | Whether customer has unlimited data plan           |
| **Contract**                                        | Month-to-Month / One Year / Two Year               |
| **Paperless_Billing**                               | Yes / No                                           |
| **Payment_Method**                                  | Credit Card / Bank Withdrawal / Mailed Check       |
| **Monthly_Charge**                                  | Amount billed monthly                              |
| **Total_Charges**                                   | Total billed charges during tenure                 |
| **Total_Refunds**                                   | Lifetime customer refunds                          |
| **Total_Extra_Data_Charges**                        | Additional data-related charges                    |
| **Total_Long_Distance_Charges**                     | Long-distance call charges                         |
| **Total_Revenue**                                   | Total revenue contributed by the customer          |
| **Customer_Status**                                 | *Stayed*, *Churned*, or *Joined*                   |
| **Churn_Category**                                  | Reason category for churn (if churned)             |
| **Churn_Reason**                                    | Detailed churn reason                              |
| **Customer_Status_Predicted**                       | (Only in predictions file) Model’s predicted label |

---

## 🧪 **Data Segmentation (SQL-Based)**

### ✅ Training Data

Using SQL view `vw_ChurnData`:

```sql
SELECT *
FROM prod_Churn
WHERE Customer_Status IN ('Churned', 'Stayed');
```

Only *Churned* and *Stayed* customers are used for model training.

---

### ✅ Scoring / Prediction Data

Using SQL view `vw_JoinData`:

```sql
SELECT *
FROM prod_Churn
WHERE Customer_Status = 'Joined';
```

The model predicts churn likelihood for newly joined customers.

---

## 📊 **Feature Engineering Highlights**

The pipeline uses **both SQL and Python** feature engineering:

### ✅ SQL Feature Engineering

* **Tenure Buckets** (New, Intermediate, Long-Term)
* **Billing Segmentation** (Low / Medium / High)
* **Avg_Revenue_Per_Month**

### ✅ Python Feature Engineering

* Automatic **numerical imputation** (median)
* Automatic **categorical imputation** (most frequent)
* **OneHotEncoding** for all categorical features
* Data leakage prevention via `ColumnTransformer`
* Scaled Logistic Regression decisioning

---

## ✅ **Data Volume**

* **Total records:** 6,418
* **Training data:** ~ 5,200 (Stayed + Churned)
* **Joined (new customers):** ~ 1,200
* **Features:** 30+ raw features + SQL-engineered features

---

## 🧠 **Why This Data Is Suitable for Churn Modeling**

* Covers demographic, billing, service usage, and contract behavior
* Provides detailed churn labels and churn reasons
* Includes both numerical and categorical richness
* Strongly aligned with real-world telecom churn datasets used in industry
* Allows end-to-end SQL + ML + BI pipeline demonstration

---

# 📊 Model Performance

| Metric        | Score      |
| ------------- | ---------- |
| **AUC**       | **0.8854** |
| **F1 Score**  | **0.7033** |
| **Threshold** | 0.50       |

> Note: Logistic Regression may raise a ConvergenceWarning due to large feature space; increasing `max_iter` resolves this.

---

# 🔍 Explainability (SHAP)

Model global drivers are visualized via:

```
reports/global_importance.png
```

This helps business teams understand which features influence churn risk the most.

---

🗺️ Data Flow Diagram

Below is a clean, consulting-style representation of your project’s full data flow.


               ┌───────────────────────────┐
               │      Raw Source Data      │
               │   Customer_Data.csv       |
               │                           |
               └──────────────┬────────────┘
                              
                              ▼
                              
                   ┌────────────────────┐
                   │   SQL Ingestion    │
                   │  (SQLite: churn.db)│
                   │  Table: prod_Churn |
                   │                    |
                   └───────────┬────────┘
                               
                               ▼
                 ┌─────────────────────────┐
                 │     SQL Transformations │
                 │    (SQLQueries.sql)     │
                 │  • vw_ChurnData         │
                 │  • vw_JoinData          | 
                 │                         |
                 └───────────┬─────────────┘
                             
                             ▼
             ┌───────────────────────────────────┐
             │      Data Preparation Layer       │
             │    (Python: Pandas + Sklearn)     │
             │  • Null imputation                │
             │  • Categorical encoding           │
             │  • Feature engineering            │
             │  • Leakage-proof pipeline         |
             │                                   |
             └─────────────┬─────────────────────┘
                          
                           ▼
                 ┌─────────────────────┐
                 │  Model Training     │
                 │ Logistic Regression │
                 │ model.pkl           |
                 │                     |
                 └───────────┬────────┘
                             
                             ▼
               ┌───────────────────────────────┐
               │        Batch Prediction       │
               │ • Predictions.csv             │
               │ • SQL Output (predictions)    |
               │                               |
               └──────────────┬───────────────┘
                              
                              ▼
             ┌─────────────────────────────────┐
             │     Explainability Layer         │
             │    (SHAP: global_importance.png) |
             │                                  |
             └────────────────┬────────────────┘
                              
                              ▼
            ┌──────────────────────────────────┐
            │    Visualization & Reporting      │
            │      (Power BI Dashboard)         |
            │                                   |
            └──────────────────────────────────┘

```

---

# 📊 **EDA Summary (Written for Professional Reports)**

A high-level exploratory analysis of the dataset reveals the following patterns and business insights:

---

## ✅ **1. Customer Status Distribution**

* Majority of customers are **‘Stayed’**, indicating a moderately stable portfolio.
* **Churned customers** form a smaller yet significant segment requiring strategic retention focus.
* **‘Joined’ customers** represent the onboarding pipeline and are crucial for future churn prediction.

---

## ✅ **2. Tenure Insights**

* Customers with **< 6 months tenure** show higher churn propensity, reflecting early dissatisfaction.
* Customers with **long-term contracts (>24 months)** exhibit the lowest churn, indicating strong loyalty.

---

## ✅ **3. Contract Type Behavior**

* **Month-to-Month** contracts show the highest churn rates due to:

  * No lock-in
  * Higher perceived volatility
  * Greater sensitivity to price and service issues
* **One-year and two-year** contracts demonstrate stronger retention.

---

## ✅ **4. Service Usage Patterns**

* Users with **Fiber Optic** connections show slightly higher churn than DSL or Cable.
  Possible reasons:

  * Competing fiber providers
  * High expectations for speed and reliability
* Lack of add-on services (premium support, device protection, streaming) correlates with increased churn.

---

## ✅ **5. Financial Behavior**

* Customers with **higher monthly charges** and **frequent refunds** show elevated churn likelihood.
* Customers with **automatic bank withdrawal** tend to stay longer compared to those paying via mailed checks or credit cards.

---

## ✅ **6. Demographics**

* Churn distribution is **not strongly skewed by gender**, but shows slight variance across **states**, likely due to service availability and network reliability.

---

### ✅ Summary

EDA indicates that churn is driven by a combination of **contract flexibility**, **billing experiences**, **tenure**, and **service satisfaction**—all of which are effectively captured in the model pipeline.

---

# 🔍 **Feature Importance Explanation (SHAP Interpretation)**

SHAP was used to quantify the impact of each feature on churn probability.

Below is a professional interpretation of the output:

### ✅ **Top Global Drivers of Churn (based on SHAP values)**

1. **Contract Type**

   * Month-to-Month customers exhibit the highest churn risk.
   * Longer contracts significantly reduce churn probability.

2. **Tenure_in_Months**

   * Lower tenure strongly increases churn likelihood—early churn is common.

3. **Monthly_Charge**

   * Higher monthly charges correlate with more dissatisfaction and churn.

4. **Total_Revenue**

   * High lifetime revenue customers tend to be more loyal.

5. **Online_Security / Online_Backup / Premium Support**

   * Lack of essential add-ons increases churn propensity.

6. **Internet_Type (DSL vs Fiber)**

   * Fiber customers show slightly higher volatility due to competition or expectations.

7. **Payment_Method**

   * Bank withdrawal users are more stable; mailed-check customers churn more.

---

### ✅ How Business Teams Use SHAP Insights

SHAP converts model predictions into **human-readable reasons**:

* Identify *why* a customer is at churn risk
* Prioritize *actionable levers* like contract upgrades, targeted offers, or improved service bundles
* Justify decisions to leadership with transparent evidence

This aligns the ML model with **business decision-making**, increasing trust and adoption.

---

# 💼 **Business Recommendations (Consulting-Grade)**

Based on data analysis, model outputs, and SHAP insights, the following strategic recommendations are made:

---

## ✅ **1. Improve Early Lifecycle Experience**

Customers in their **first 6 months** show elevated churn risk.
**Recommended actions:**

* Launch onboarding support calls
* Provide first-month service assurance
* Offer early-bird discounts for contract upgrades

---

## ✅ **2. Contract Migration Strategy**

Since Month-to-Month customers churn the most:

* Incentivize migration to **annual or two-year plans**
* Offer bundled discounts for early contract locking
* Introduce loyalty points for tenure milestones

---

## ✅ **3. Optimize Pricing & Billing Experience**

High charges and refunds correlate with churn.

**Actions:**

* Offer personalized pricing adjustments for premium users
* Reduce bill shock by introducing spending alerts
* Improve refund process transparency

---

## ✅ **4. Service Bundle Enhancement**

Customers without add-ons are more likely to churn.

**Strategy:**

* Introduce “starter bundles” combining security + backup
* Offer seasonal promotions for add-on adoption
* Highlight the benefits during onboarding and renewal cycles

---

## ✅ **5. Improve Network-Related Reliability**

Fiber Optic users churn slightly more.

**Actions:**

* Launch proactive outage notifications
* Implement quality-of-service credits
* Partner with regional ISPs for faster issue resolution

---

## ✅ **6. Build Retention Dashboard Using Power BI**

Leverage `Predictions.csv` in Power BI to:

* Monitor churn risk by region, age, contract, and service usage
* Prioritize outreach to high-risk segments
* Track retention KPIs (customer lifetime value, churn %, recovery rate)

---

## ✅ **7. Targeted Retention Campaigns**

* Identify "high value + high risk" customers
* Provide personalized offers before renewal
* Integrate predictive scores with CRM systems (Salesforce, HubSpot)

---


The integration of SQL, machine learning, and business intelligence ensures a reliable, interpretable, and scalable framework for proactive churn reduction and customer value growth.
