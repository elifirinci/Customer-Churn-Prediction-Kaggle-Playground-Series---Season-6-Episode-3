# 📊 Customer Churn Prediction

### Kaggle Playground Series S6E3 – Advanced Ensemble Solution

This project focuses on predicting customer churn using a **competition-grade machine learning pipeline**.
The solution goes beyond standard models by leveraging **nested cross-validation, leakage-free encoding, and multi-level stacking ensembles**.

---

## 🚀 Project Overview

Customer churn prediction is a critical task for improving retention and maximizing revenue.

In this project, we:

* Perform **exploratory data analysis (EDA)**
* Apply **advanced feature engineering**
* Use **leakage-safe target encoding**
* Train multiple machine learning models
* Combine them using **stacking (meta-learning)**
* Evaluate performance with **robust cross-validation**

---
## 🏗️ Model Architecture
                ┌──────────────────────┐
                │     Raw Dataset      │
                └─────────┬────────────┘
                          │
                          ▼
                ┌──────────────────────┐
                │   Data Preprocessing │
                │  (Cleaning, EDA)     │
                └─────────┬────────────┘
                          │
                          ▼
                ┌──────────────────────┐
                │ Feature Engineering  │
                │ + Target Encoding    │
                │ (Inner CV - 5 folds) │
                └─────────┬────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────┐
        │        Base Models (Level-0)        │
        │------------------------------------│
        │ XGBoost   LightGBM   CatBoost      │
        │ RandomForest   KNN   SVM   MLP     │
        └─────────┬──────────────────────────┘
                  │
                  ▼
        ┌─────────────────────────────────────┐
        │   OOF Predictions (20-fold CV)      │
        │   → New Feature Space              │
        └─────────┬──────────────────────────┘
                  │
                  ▼
        ┌─────────────────────────────────────┐
        │      Meta Model (Level-1)           │
        │        Ridge Regression             │
        └─────────┬──────────────────────────┘
                  │
                  ▼
        ┌──────────────────────┐
        │   Final Prediction   │
        └──────────────────────┘
---
## 📂 Dataset

The dataset is provided by Kaggle:

👉 [https://www.kaggle.com/competitions/playground-series-s6e3/data](https://www.kaggle.com/competitions/playground-series-s6e3/data)

It includes structured customer data such as:

* Demographics
* Subscription details
* Service usage patterns

---

## 🧠 Modeling Approach

Unlike basic ML pipelines, this project uses a **multi-model ensemble architecture**.

### 🔹 Base Models

* XGBoost
* LightGBM
* CatBoost
* Random Forest
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)
* Neural Network (RealMLP - PyTorch)

### 🔹 Meta Model (Stacking)

* Ridge Regression

---

## 🔁 Stacking Strategy

The project implements a **stacking ensemble**:

1. Train base models using cross-validation
2. Generate **out-of-fold (OOF) predictions**
3. Use OOF predictions as new features
4. Train a **Ridge meta-model** on top

This approach improves generalization and reduces overfitting.

---

## 🏗️ Validation Strategy

A **Nested Cross-Validation** setup is used:

* **Outer CV (20 folds)** → Model evaluation
* **Inner CV (5 folds)** → Target encoding (prevents leakage)

This ensures highly reliable performance estimation.

---

## 🧪 Feature Engineering

* Feature transformations without data leakage
* Interaction features
* **Target Encoding with inner folds**
* Mixed encoding strategies depending on feature type

---

## 📈 Evaluation

Models are evaluated using:

* **ROC-AUC (primary metric)**
* Accuracy
* Precision / Recall
* F1 Score

The main benchmark metric for the competition is **ROC-AUC**.

---

## ⚙️ Project Structure

```
├── Predict Customer Churn.ipynb   # End-to-end ML pipeline
├── README.md                     # Project documentation
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the notebook:

```bash
jupyter notebook
```

Then open:

```
Predict Customer Churn.ipynb
```

Pipeline steps inside the notebook:

1. Data loading
2. EDA
3. Feature engineering
4. Encoding (target encoding with CV)
5. Model training
6. Stacking
7. Evaluation & prediction

---

## 📊 Results

The final solution uses a **stacked ensemble model**, which significantly outperforms individual models.

> Performance is optimized for **generalization**, not just leaderboard overfitting.

---

## ⚠️ Notes

* High fold count (**20-fold CV**) increases training time
* Neural network training benefits from **GPU acceleration**
* Pipeline is optimized for **performance over simplicity**

---

## 🧩 Future Improvements

* Hyperparameter optimization (Optuna)
* Feature selection techniques
* Model interpretability (e.g., SHAP)
* Refactoring into modular Python package
* Deployment (API / web app)

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repository and submit a pull request.

---

## 📜 License

This project is for educational purposes and complies with Kaggle competition rules.

---

## 🙌 Acknowledgements

* Kaggle Playground Series
* Scikit-learn
* XGBoost / LightGBM / CatBoost
* PyTorch

---

