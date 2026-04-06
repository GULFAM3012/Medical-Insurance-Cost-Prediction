import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import json
from sklearn.preprocessing import LabelEncoder

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go To", [
    "Project Introduction", "Insurance Prediction", "Exploratory Data Analysis (EDA)"
])

if page == "Project Introduction":
    st.title("🏥 Medical Insurance Cost Prediction")

    st.markdown("""
    ### 📌 Project Overview

    I built this project to predict medical insurance costs based on personal health and demographic information. 
    The model takes inputs like age, BMI, smoking status, number of children, and region to estimate annual insurance charges.

    This kind of tool can help insurance companies determine personalized premiums, and also help individuals understand 
    what factors affect their insurance costs.

    ---

    ### 🎯 Problem Statement

    The dataset contained information about insurance policy holders - their age, gender, BMI, smoking habits, 
    family details, location, and the actual medical charges they paid.

    My task was to analyze this data and build a regression model that could predict insurance costs for new users 
    based on their personal information.

    ---

    ### 📊 Dataset

    After cleaning the data (removing duplicates), I had **1,337 records** to work with.

    **Original Features:**
    - `age` - Age of the person
    - `sex` - male or female
    - `bmi` - Body Mass Index
    - `children` - Number of dependents
    - `smoker` - yes or no
    - `region` - northeast, northwest, southeast, southwest
    - `charges` - **Target variable** (what I needed to predict)

    **Features I Created:**
    - `bmi_category` - Underweight, normal, overweight, obese
    - `smoker_age` - smoker × age (interaction feature)
    - `smoker_bmi` - smoker × bmi (interaction feature)
    - `obese` - flag for BMI > 30

    ---

    ### 🤖 Model Training

    I trained and compared **5 different regression models**:

    | Model | RMSE | R² Score |
    |-------|------|----------|
    | Ridge Regression | $4,544 | 0.888 |
    | Linear Regression | $4,551 | 0.887 |
    | Lasso Regression | $4,551 | 0.887 |
    | Random Forest | $4,719 | 0.879 |
    | XGBoost | $5,005 | 0.864 |

    **Ridge Regression** performed the best, so I selected it as my final model and deployed it in this app.

    ---

    ### 💡 What I Learned from the Data

    Some interesting patterns I found during analysis:

    - **Smoking is the biggest factor** - Smokers pay 3-4 times more than non-smokers
    - **Obese smokers (BMI > 30)** have the highest insurance costs
    - **Age** increases costs, but not as dramatically as smoking
    - **Gender and region** have almost no impact on charges
    - **Number of children** also doesn't affect costs much

    """)


elif page == "Insurance Prediction":
    st.title("Medical Insurance Cost Prediction")

    # Load the best model
    model_path = "model/best_pipeline_Ridge Regression.joblib"
    metrics_path = "model/metrics.json"

    # Load model and metrics
    try:
        model = joblib.load(model_path)
        st.success(f"Model loaded successfully from {model_path}")
    except Exception as e:
        st.error(f"Error in model loading from {model_path}: {e}")
        st.stop()

    # Load metrics
    metrics = None
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            st.info("Loaded metrics.json — will use reported RMSE/MAE to show approximate error margins.")
        except Exception as e:
            st.warning(f"Could not load metrics.json: {e}")
    else:
        st.info("No metrics.json found. The app will not show error margins if desired.")


    st.subheader("Enter Your Information")


    age = st.number_input("Age", min_value=18, max_value=120, value=30)
    sex = st.selectbox("Sex", options=["male", "female"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
    children = st.number_input("Number of Children", min_value=0, value=0)
    smoker = st.selectbox("Smoker", options=["yes", "no"])
    region = st.selectbox("Region", options=["northeast", "northwest", "southeast", "southwest"])

    # Show confidence intervals / error margins
    show_CI = st.checkbox("Show approximate 95% confidence interval / error margin (if metrics available)", value=True)
    use_fallback_percentage = st.slider("If metrics not available, use fallback relative error (%)", min_value=5, max_value=100, value=20)

    # Create a DataFrame for the input
    input_data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region
    }])

    # Apply binary encoding for 'sex', 'smoker' and one-hot encoding for 'region'
    input_data['smoker'] = input_data['smoker'].map({"yes":1,"no":0})
    input_data['sex'] = input_data['sex'].map({"male":0,"female":1})
    input_data = pd.get_dummies(input_data, columns=['region'], drop_first=True)

    # Define the same feature engineering steps as in training
    def bmi_category(bmi):
        if bmi<18.5:
            return "Underweight"
        if 18.5<= bmi <25:
            return "normal"
        elif 25<= bmi <30:
            return "overweight"
        else:
            return "obese"

    input_data["bmi_category"] = input_data["bmi"].apply(bmi_category)
    lb = LabelEncoder()
    input_data['bmi_category'] = lb.fit_transform(input_data['bmi_category'])

    input_data['smoker_age'] = input_data['smoker'] * input_data['age']
    input_data['smoker_bmi'] = input_data['smoker'] * input_data['bmi']

    # Ensure the input data has the same columns as the training data
    # This is crucial for the model to work correctly
    try:
        input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)
    except Exception as e:
        st.error(f"Error aligning input features with model features: {e}")
        st.stop()   

    st.write("Preprocessed Input Data:")
    st.dataframe(input_data)

    # Make prediction when button is clicked
    if st.button("Predict Insurance Cost"):
        try:
            prediction = model.predict(input_data)[0]
        except Exception as e:
            st.error(f"Failed to make prediction: {e}")
            st.stop()

        #Determine an uncertainty estimate
        uncertainty = None
        method_used = None
        if metrics and show_CI:
            if "test_rmse" in metrics:
                uncertainty = float(metrics["test_rmse"])
                method_used = "RMSE"
            elif "test_mae" in metrics:
                uncertainty = float(metrics["test_mae"])
                method_used = "MAE"

        if show_CI and uncertainty is None:
            uncertainty = abs(prediction) * (use_fallback_percentage / 100.0)
            method_used = f"Fallback {use_fallback_percentage}%"

        st.subheader("Predicted Annual Insurance Cost:")
        st.write(f"${prediction:,.2f}")

        if show_CI and uncertainty is not None:
            ci_lower = prediction - 1.96 * uncertainty
            ci_upper = prediction + 1.96 * uncertainty
            st.info(f"📊 Using **{method_used}** for uncertainty estimate")
            st.write(f"**95% Confidence Interval:** ${max(0, ci_lower):,.2f} — ${ci_upper:,.2f}")


        # Show a warning so users understand the interval
        st.markdown("---")
        st.write("**Note:** The interval shown is an approximate error margin. The app uses a saved training metric (if available) such as RMSE/MAE to produce a rough 95% interval. If no metric is available, a user-selected fallback percentage is used. For statistically rigorous prediction intervals you would need access to residuals/variance information from the training procedure or use probabilistic models (e.g., Bayesian models) or techniques like conformal prediction or bootstrapping saved at training time.")




elif page == "Exploratory Data Analysis (EDA)":
    st.subheader("Exploratory Data Analysis (EDA)")

    df = pd.read_csv("medical_insurance.csv")
    df.head()


    # Plot 1: Age vs. Charges
    st.write("#### Age vs. Charges")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='age', y='charges', data=df, ax=ax)
    st.pyplot(fig)
    st.write("This scatter plot shows the relationship between age and medical insurance charges. Generally, charges tend to increase with age.")

    # Plot 2: BMI vs. Charges
    st.write("#### BMI vs. Charges")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='bmi', y='charges', data=df, ax=ax)
    st.pyplot(fig)
    st.write("This scatter plot illustrates the relationship between Body Mass Index (BMI) and medical insurance charges. Higher BMI often correlates with higher charges.")

    # Plot 3: Smoker vs. Charges
    st.write("#### Smoking Status vs. Charges")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x='smoker', y='charges', data=df, ax=ax)
    st.pyplot(fig)
    st.write("This box plot compares the distribution of medical insurance charges for smokers (1) and non-smokers (0). Smokers typically have significantly higher charges.")

    # Plot 4: Region vs. Charges
    st.write("#### Region vs. Charges")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x='region', y='charges', data=df, ax=ax)
    st.pyplot(fig)
    st.write("This box plot shows how medical insurance charges vary across different regions.")


