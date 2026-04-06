# Medical-Insurance-Cost-Prediction

# 🏥 Medical Insurance Cost Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-2.3%2B-orange.svg)](https://mlflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📌 Project Overview

This project predicts medical insurance costs based on personal health and demographic information using Machine Learning. The model takes inputs like age, BMI, smoking status, number of children, and region to estimate annual insurance charges.

**Live Demo:** [Streamlit App](#) *(Add your deployed app link here)*

---

## 🎯 Problem Statement

Insurance companies need accurate cost predictions to set personalized premiums. This project analyzes historical insurance data to build a regression model that predicts medical charges for new customers based on their profile.

---

## 📊 Dataset

The dataset contains **1,337 records** (after cleaning) with the following features:

| Feature | Description |
|---------|-------------|
| `age` | Age of the primary beneficiary |
| `sex` | Gender (male/female) |
| `bmi` | Body Mass Index (kg/m²) |
| `children` | Number of dependents covered |
| `smoker` | Smoking status (yes/no) |
| `region` | Residential area (northeast, northwest, southeast, southwest) |
| `charges` | **Target** - Medical insurance cost |

**Source:** Medical Insurance Dataset

---

## 🔍 Exploratory Data Analysis (EDA)

### Key Findings:

- **Smoking** is the strongest predictor - smokers pay **3-4x more** than non-smokers
- **Obese smokers (BMI > 30)** have the highest insurance costs
- **Age** shows moderate positive correlation with charges
- **Gender** and **region** have minimal impact on costs
- **Number of children** has negligible effect

### Visualizations Included:
- Age vs. Charges scatter plot
- BMI vs. Charges scatter plot  
- Smoker vs. Charges box plot
- Region vs. Charges box plot

---

## ⚙️ Feature Engineering

Created additional features to improve model performance:

- `bmi_category` - Underweight, normal, overweight, obese
- `smoker_age` - Interaction feature (smoker × age)
- `smoker_bmi` - Interaction feature (smoker × bmi)
- `obese` - Binary flag for BMI > 30

---

## 🤖 Model Training

5 regression models were trained and compared:

| Model | RMSE | MAE | R² Score |
|-------|------|-----|----------|
| **Ridge Regression** | **$4,544** | **$2,809** | **0.888** |
| Linear Regression | $4,551 | $2,817 | 0.887 |
| Lasso Regression | $4,551 | $2,817 | 0.887 |
| Random Forest | $4,719 | $2,679 | 0.879 |
| XGBoost | $5,005 | $2,804 | 0.864 |

**Best Model:** Ridge Regression selected for deployment

---

## 📈 MLflow Integration

- All experiments logged with MLflow
- Metrics tracked: RMSE, MAE, R² Score
- Best model registered in MLflow Model Registry
- Model versioned and promoted to **Production** stage

---

## 🚀 Streamlit Web App

An interactive web application was built for real-time predictions.

### Features:
- User input form for personal details
- Real-time cost prediction
- 95% confidence interval display
- EDA visualizations
- Model performance metrics

### Run Locally:

```bash
# Clone the repository
git clone https://github.com/yourusername/medical-insurance-prediction.git
cd medical-insurance-prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run medical_insurance_app.py
