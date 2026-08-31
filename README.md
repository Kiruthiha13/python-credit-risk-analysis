# Credit Risk Analysis – Probability of Default & FICO Score Bucketing

## Project Overview

This project focuses on credit risk analysis using borrower-level loan data. The objective is to develop techniques for estimating the **Probability of Default (PD)** and to transform numerical FICO scores into categorical credit ratings that can be used in machine learning models.

The project consists of two tasks:

- **Task 1:** Build a Probability of Default (PD) model and calculate Expected Loss.
- **Task 2:** Quantize FICO scores into categorical ratings using Dynamic Programming and Log-Likelihood optimization.

## Task 1 – Probability of Default & Expected Loss

### Objective

The objective is to build a model that estimates the **Probability of Default (PD)** for a borrower.
The estimated PD can then be used to calculate the **Expected Loss (EL)** on a loan.

### Methodology

#### 1. Feature Engineering

Two financial ratios were created:

**Debt-to-Income Ratio (DTI)**

DTI = {Total Debt Outstanding}/{Income}

**Payment-to-Income Ratio (PTI)**

PTI = {Loan Amount Outstanding}/{Income}

These ratios provide additional information about the borrower's debt burden relative to their income.

#### 2. Logistic Regression

A Logistic Regression model was used to estimate the probability that a borrower will default.

Features used:

- `credit_lines_outstanding`
- `debt_to_income`
- `payment_to_income`
- `years_employed`
- `fico_score`

The target variable is:

- `default` – 1 if the borrower defaulted
- `default` – 0 otherwise

The model outputs a probability between 0 and 1, representing the estimated **Probability of Default (PD)**.

#### 3. Expected Loss

Expected Loss was calculated using:

Expected Loss = PD * LGD * EAD


where:

- **PD** = Probability of Default
- **LGD** = Loss Given Default
- **Exposure** = Exposure at Default

The recovery rate is assumed to be **10%**.

Therefore:

LGD = 1 - Recovery Rate

LGD = 1 - 0.10 = 0.90

So:

Expected Loss = PD * 0.90 * Loan Amount

The model includes a function that takes borrower/loan characteristics as inputs and returns the estimated PD and Expected Loss.

### Model Evaluation

The Logistic Regression model was evaluated using:

- Classification predictions
- Error rate
- ROC-AUC

ROC-AUC measures the model's ability to distinguish between borrowers who default and those who do not.

## Task 2 – FICO Score Bucketing and PD Analysis

### Objective

FICO scores are numerical values ranging from approximately 300 to 850. However, the target machine learning architecture requires **categorical input variables**.

The objective is therefore to transform continuous FICO scores into a fixed number of categorical ratings.

The requirement is that:

> A lower rating represents better credit quality.

Therefore:

- **Rating 1 → Higher FICO / lower risk**
- **Rating 10 → Lower FICO / higher risk**

### Methodology

#### 1. FICO Score Aggregation

Borrowers were grouped by FICO score.

For each FICO score, the following were calculated:

- Number of borrowers
- Number of defaults

The dataset contains 10,000 borrower records and 374 unique FICO scores.

Aggregating by unique FICO scores significantly reduces the computational complexity of the optimization.

#### 2. Log-Likelihood Optimization

For each potential FICO bucket, the Probability of Default is calculated as:

\[
p_i = \frac{k_i}{n_i}
\]

where:

- \(n_i\) = number of borrowers in the bucket
- \(k_i\) = number of defaults in the bucket

The log-likelihood for a bucket is:

\[
LL_i =
k_i\log(p_i)
+
(n_i-k_i)\log(1-p_i)
\]

The objective is to find bucket boundaries that maximize the total log-likelihood.

#### 3. Dynamic Programming

Dynamic Programming was used to efficiently search for the optimal bucket boundaries.
This approach avoids manually selecting arbitrary FICO ranges.

#### 4. Rating Assignment

The resulting FICO boundaries are used to convert numerical FICO scores into categorical ratings.

The rating scale is structured such that:

| Rating | Credit Quality |
|--------|----------------|
| 1 | Best |
| 2 | Very Good |
| 3 | Good |
| ... | ... |
| 10 | Highest Risk |

The exact FICO boundaries are determined from the optimization process.

#### 5. PD by Rating

After assigning ratings, the observed PD is calculated for each rating.

The analysis includes:

- Number of customers
- Number of defaults
- Probability of Default (PD)

This allows the relationship between FICO-based ratings and default risk to be analyzed.

## Key Observation

Most ratings show a clear relationship between credit quality and default risk, with lower-quality ratings generally having higher observed PD.
Some buckets may contain relatively few borrowers. For example, a bucket with only a small number of observations can have a volatile observed PD. Such results should therefore be interpreted with caution.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Dynamic Programming
- Logistic Regression
- Statistical Analysis

## Repository Structure

```text
credit-risk-analysis/
│
├── data/
│
├── scripts/
│   ├── credit_risk_analysis.ipynb
│   └── FICO_score_bucketing_and_PD_analysis.ipynb
│
├── LICENSE
└── README.md
