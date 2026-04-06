# Medical-Insurance-Cost-Prediction

📌 Project Overview
This project aims to build an end-to-end regression solution for predicting individual medical insurance costs based on personal health and demographic factors. The system helps insurance companies estimate personalized premiums and enables individuals to compare and plan their medical expenses.

The project covers the complete data science lifecycle:

Exploratory Data Analysis (EDA) to uncover key cost drivers

Feature engineering and data preprocessing

Training and evaluating multiple regression models

Experiment tracking and model registry using MLflow

Deployment of the best model via an interactive Streamlit web application

🎯 Problem Statement
Given a dataset containing information about age, gender, BMI, number of children, smoking status, and region, the goal is to predict the individual’s medical insurance charges. The challenge is to build a reliable regression model that can be used in a real‑world insurance pricing tool.

📊 Dataset
The dataset (medical_insurance.csv) consists of 1337 rows (after removing duplicates) and the following features:

Feature	Description
age	Age of the primary beneficiary
sex	Gender (male / female)
bmi	Body Mass Index (kg/m²)
children	Number of dependents covered by the insurance
smoker	Smoking status (yes / no)
region	Residential area (northeast, northwest, southeast, southwest)
charges	Target – individual medical costs billed by health insurance
🔍 Key Insights from EDA
Smoking is the strongest predictor: smokers pay 3–4 times higher premiums than non‑smokers.

BMI has a moderate positive correlation with charges, especially for smokers.

Age also shows a positive trend: older individuals tend to have higher charges.

Obese smokers (BMI > 30) have significantly higher costs than any other group.

Gender and region have very weak correlations with charges.

All visualizations are included in the Streamlit app and the Jupyter notebook.

⚙️ Feature Engineering
Created bmi_category: Underweight, normal, overweight, obese.

Added interaction features:

smoker_age = smoker × age

smoker_bmi = smoker × bmi

Encoded categorical variables using label encoding and one‑hot encoding.

Added a binary flag obese (BMI > 30) for deeper analysis.

🧠 Models Trained & Evaluated
Five regression models were trained and compared:

Model	RMSE	MAE	R² Score
Ridge Regression	4544.33	2809.45	0.8876
Linear Regression	4550.78	2816.54	0.8873
Lasso Regression	4550.77	2816.52	0.8873
Random Forest	4719.19	2678.88	0.8788
XGBoost	5004.59	2804.16	0.8637
Ridge Regression was selected as the final model due to its lowest RMSE and highest R².

📈 MLflow Integration
All experiments were logged using MLflow:

Hyperparameters
