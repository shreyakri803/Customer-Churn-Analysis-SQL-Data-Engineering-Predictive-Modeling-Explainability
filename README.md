# **Customer Churn Analytics: SQL-Driven Data Pipeline + Predictive Modeling + Explainability**

### **Author: Shreya Kumari**

This project showcases a complete, production-style **customer churn analytics pipeline**, combining:

* SQL-based data preparation
* A modular machine learning workflow
* Automated batch predictions
* SHAP-powered explainability
* Power BI reporting for business insights

The goal is to demonstrate how a telecom-style customer dataset can be transformed into a scalable churn prediction system used by analytics teams.

---

# ⭐ **Executive Summary**

This solution helps businesses:

* Prepare raw customer data into analysis-ready tables using **SQL views**
* Train an ML model through a clean, configurable Python pipeline
* Score both historical and newly joined customers
* Generate interpretable insights using **SHAP**
* Visualize key churn metrics through a **Power BI dashboard**

**Model Performance:**

* **AUC:** 0.8854
* **F1-Score:** 0.7033
* **Decision Threshold:** 0.50

---

# 🏗️ **End-to-End Architecture**

```
Customer_Data.csv
        ↓
  SQLite Database (churn.db)
        ↓
 SQL Views (vw_ChurnData, vw_JoinData)
        ↓
 Python Feature Pipeline (Impute → Encode → Transform)
        ↓
 Logistic Regression Model (model.pkl)
        ↓
 Batch Predictions → Predictions.csv / SQL output
        ↓
 SHAP Explainability (global_importance.png)
        ↓
 Power BI Dashboard
```

---

# 📁 **Project Structure**

```
├── config.yaml
├── SQLQueries.sql
├── churn.db
├── Customer_Data.csv
├── Predictions.csv
├── artifacts/
│   └── model.pkl
├── reports/
│   └── global_importance.png
└── src/
    ├── config.py
    ├── data_sql.py
    ├── features.py
    ├── logging_utils.py
    ├── train.py
    ├── predict.py
    └── explain.py
```

---

# 🧰 **Tech Stack**

**Data Layer:** SQLite, SQL
**Processing:** Python (Pandas, SQL), Scikit-learn
**Explainability:** SHAP
**Visualization:** Power BI
**Orchestration:** CLI-based commands

---

# 🗃️ **SQL Data Engineering Layer**

All SQL transformations are housed in **`SQLQueries.sql`** and auto-executed during training.

### Training View

```sql
CREATE VIEW vw_ChurnData AS
SELECT *
FROM prod_Churn
WHERE Customer_Status IN ('Churned','Stayed');
```

### New Customers View

```sql
CREATE VIEW vw_JoinData AS
SELECT *
FROM prod_Churn
WHERE Customer_Status = 'Joined';
```

### Portfolio Churn Metric

```sql
SELECT
COUNT(CASE WHEN Customer_Status='Churned' THEN 1 END)*1.0/COUNT(*) AS churn_rate
FROM prod_Churn;
```

### SQL Feature Engineering (Example)

```sql
CASE 
    WHEN Tenure_in_Months<=6 THEN 'New'
    WHEN Tenure_in_Months BETWEEN 7 AND 24 THEN 'Intermediate'
    ELSE 'Long-Term'
END AS Tenure_Bucket
```

---

# 🔧 **Pipeline Configuration**

All key settings exist in **`config.yaml`**, including:

* database paths
* SQL view names
* ML pipeline parameters
* feature transformation settings

---

# 🚀 **How to Run the Project**

### 1️⃣ Train the Model

```
python cli.py train
```

### 2️⃣ Predict All Customers

```
python cli.py predict
```

### 3️⃣ Predict New Joiners Only

```
python cli.py predict --joined
```

### 4️⃣ Save Predictions into SQL

```
python cli.py predict --joined --sql-save
```

### 5️⃣ Generate Explainability

```
python cli.py explain
```

---

# 📂 **Dataset Description**

The dataset reflects a telecom subscription business and includes demographics, billing behavior, service usage, and contract information.

Data is first loaded from **Customer_Data.csv**, then ingested into **churn.db** as `prod_Churn`.

### Highlights:

* **6,418** total customers
* 30+ raw features
* Includes churn labels, churn reasons, service usage, financial attributes
* Rich mix of categorical and numerical variables

The dataset is realistic for churn modeling and aligns with industry benchmarks.

---

# 📊 **EDA Summary**

A high-level exploratory analysis reveals:

### ✅ Tenure Behavior

* Early-tenure customers (0–6 months) show highest churn risk.
* Long-tenure customers are significantly more stable.

### ✅ Contract Influence

* **Month-to-Month** customers churn the most.
* Annual and multi-year contracts show better retention.

### ✅ Service Adoption

* Customers without online security, backup, or premium support churn more often.

### ✅ Financial Patterns

* High monthly charges and refund-related interactions correlate with churn.

### ✅ Demographics

* Minor gender differences; churn varies more by state/service availability.

---

# 🔍 **Model Interpretability (SHAP)**

SHAP values help clarify **why** the model predicts churn.

### Top Drivers:

1. Contract Type
2. Tenure
3. Monthly Charge
4. Total Revenue
5. Add-on Services
6. Internet Type
7. Payment Method

Business teams can directly use these insights for targeted retention strategies.

---

# 📈 **Model Performance**

| Metric        | Score  |
| ------------- | ------ |
| **AUC**       | 0.8854 |
| **F1 Score**  | 0.7033 |
| **Threshold** | 0.50   |

Performance is strong for a linear model on high-dimensional encoded features.

---

# 🧭 **Business Recommendations**

### 1️⃣ Strengthen Early Lifecycle Interventions

Improve onboarding, early support, and experience for new customers.

### 2️⃣ Contract Upgrade Strategy

Encourage Month-to-Month users to move to longer commitments.

### 3️⃣ Pricing & Billing Optimization

Reduce bill shock, enhance transparency, and improve refund processes.

### 4️⃣ Promote Service Bundles

Introduce attractive add-on bundles to increase stickiness.

### 5️⃣ Network Reliability Enhancements

Proactively address outages and communicate more transparently.

### 6️⃣ Power BI Monitoring

Use risk dashboards to prioritize outreach and measure churn KPIs.

### 7️⃣ Targeted Retention Offers

Apply personalized recovery strategies to high-risk, high-value customers.

---

# ✅ **Final Note**

This project demonstrates a complete, integrated churn analytics workflow—starting from SQL engineering and moving all the way through ML, explainability, and business intelligence. The pipeline is modular, scalable, and aligned with real-world analytics team practices.

---
