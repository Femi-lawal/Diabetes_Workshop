# Diabetes Progression Modeling

This project contains a complete, clearly structured solution for modeling **one‑year diabetes progression** using the
Scikit‑Learn Diabetes dataset. It follows the assignment’s Parts 1–3, includes rigorous EDA, multiple model families,
evaluation via **R²**, **MAE**, **MAPE**, and a classification add‑on with **Logistic Regression** for a binarized
“screening” task (Accuracy/Precision/Recall/F1/ROC‑AUC/LogLoss), plus your requested talking points throughout.

---

## Files

- **`diabetes_models_lab.ipynb`** - self‑contained notebook.
  - Computes its own tables/figures per section so you can run cells independently.
  - Includes: EDA, train/val/test split, univariate BMI polynomials (0–5) with equation & param counts, multivariate
    polynomial/tree/kNN comparisons, logistic regression (binarized), conclusions & limitations.

---

## Quick Start

1. **Create & activate a virtual environment** (recommended):

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Launch Jupyter and open the notebook**:

Run cells **top to bottom** the first time. The notebook is designed so major sections recompute what they need, but a
linear run ensures reproducibility.

---

## What’s in the Notebook?

### Part 1 - Data → Problem → EDA → Cleaning → Split

- Load data via `sklearn.datasets.load_diabetes(as_frame=True)`
- EDA: summary statistics, histograms, scatter plots, correlation matrix, and concise insights
- Minimal cleaning (dtypes, duplicates)
- **Split:** 75% train, 10% validation, 15% test (two‑stage split)

### Part 2 - Univariate Polynomial Regression on **BMI**

- Degrees **0–5** (baseline mean model for deg 0)
- Train/validation **comparison table** with **R²/MAE/MAPE**
- Pick best degree (validation‑first), evaluate on **test**
- **Plot**: train/val/test points with best‑fit curve
- **Equation** (rounded to two decimals) and **example prediction**
- **Parameter counts** and explanation

### Part 3 — Multivariate Models (All Features)

- **Polynomial Regression**: two degrees (>1)
- **Decision Trees**: two `max_depth` values
- **k‑Nearest Neighbors (kNN)**: two k values **with scaling**
- **Logistic Regression** (classification): two C values, target **binarized at the training median**
- Each model family includes a **validation table** and **test‑set results** for the winner (R²/MAE/MAPE).  
  For Logistic Regression we report **Accuracy/Precision/Recall/F1/ROC‑AUC/LogLoss** on validation.

---

## Talking Points (embedded in the notebook)

- **Polynomial Regression:** Nonlinearity via polynomial features; **degree** ↔ flexibility; **overfitting** risk; dimensionality growth.
- **Decision Trees:** Recursive splits, **max_depth** controls bias/variance; interpretability vs overfitting.
- **kNN:** **Non‑parametric**, relies on distances; needs **feature scaling**; **curse of dimensionality**.
- **Logistic Regression:** **Sigmoid**, **log‑odds** linearity, **regularization (C)**; **class imbalance** considerations.
- **Performance Metrics (Classification):** Accuracy, **Precision/Recall**, F1, **ROC‑AUC**, LogLoss; when accuracy can mislead.

---

## Reproducibility Notes

- We set a **random seed (42)** for numpy/scikit‑learn where applicable.
- The notebook computes all tables inside each section so symbols like `poly_df`, `tree_df`, and `knn_df` are defined
  **before** being displayed.
- If you run cells out of order, re‑run the **“Part 1”** setup cell to (re)create `X_train`, `X_val`, and `X_test`.

---

## Troubleshooting

- **NameError (e.g., `poly_df is not defined`)**: Run the cell **above** the display cell in the same section, it builds the table.
- **Plot not showing**: Ensure you ran the section’s compute cell first (creates `best_model` and `xgrid`).
- **kNN results look odd**: Confirm the **scaling** cell ran (creates `StandardScaler` and scaled matrices).
- **Jupyter memory errors**: Restart the kernel and run top‑to‑bottom.

---

## License

For classroom and instructional use. If you reuse the notebook in a different context, please credit the original
assignment authors and note that the data is the Scikit‑Learn Diabetes toy dataset.
