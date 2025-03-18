"""
Cervical Cancer Risk Assessment Web Application

This script creates a Streamlit web application that predicts cervical cancer risk
using machine learning models. Users input health information through a user-friendly
interface, and the application returns a risk assessment based on the selected model.

Key features:
- Multiple machine learning model options for prediction
- Detailed patient information collection through a form interface
- SHAP (SHapley Additive exPlanations) integration for model interpretability
- Responsive and accessible medical-themed UI design
- Risk visualization with informative graphics

The application flow:
1. Loads selected machine learning model from disk
2. Collects patient information through input fields
3. Processes input data for model compatibility
4. Generates risk prediction when user submits the form
5. Displays prediction results with appropriate risk level indicators
6. Provides option to view SHAP explanation visualizations

Dependencies:
- streamlit: For web application framework
- joblib: For loading saved ML models
- shap: For model interpretability visualizations
- matplotlib: For visualization support
- numpy: For numerical operations
- pandas: For data manipulation
- sklearn: For model metrics and preprocessing

Note: This application is intended for risk assessment only and not for medical diagnosis.
Users are advised to consult healthcare providers for proper medical advice.
"""

import streamlit as st
import joblib
import shap  # Import SHAP library
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data/output.csv")

ChoosenModel = "svm"

X = np.array(df.drop(columns = ['Biopsy'])).astype('float32')
y = np.array(df['Biopsy']).astype('float32')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
X_test, x_val, y_test, y_val = train_test_split(X, y, test_size = 0.5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

# Streamlit App Title with a medical theme
st.title("🏥 Cervical Cancer Risk Assessment Tool")

# Choose machine learning model with a sleek radio button
choice = st.radio(
    "🔬 Select Prediction Model:",
    ("Random Forest Classifier", "GBoost Classifier", "SVM", "CatBoost Classifier"),
    key="model_choice",
    index=0,
    help="Select the machine learning model to predict your cervical cancer risk.",
)

# Load selected model
model_mapping = {
    "Random Forest Classifier": "genModels/random_forest.pkl",
    "GBoost Classifier": "genModels/xgboost.pkl",
    "SVM": "genModels/svm.pkl",
    "CatBoost Classifier": "genModels/catboost.pkl",
}
model = joblib.load(model_mapping.get(choice))

# Load the SHAP explainer (This will be model-specific)
if choice == "Random Forest Classifier":
    explainer = shap.TreeExplainer(model)
elif choice == "GBoost Classifier":
    explainer = shap.TreeExplainer(model)
elif choice == "SVM":
    explainer = shap.KernelExplainer(model.predict, X_train)  # X_train should be the training data
elif choice == "CatBoost Classifier":
    explainer = shap.TreeExplainer(model)

# Custom CSS for medical theme appearance
st.markdown("""
        <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f8f0f7; /* Soft pastel pink background */
            color: #4a4a4a; /* Soft gray text for readability */
        }

        .stRadio > div {
            display: flex;
            justify-content: space-around;
            background-color: #f8e1f4; /* Soft pastel pink background for radio buttons */
            border-radius: 6px;
            padding: 10px;
        }

        .stTextInput, .stNumberInput, .stRadio {
            margin-bottom: 20px;
            border-radius: 6px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            border: 1px solid #f1a7cb; /* Light pink border */
            background-color: #f1c6d4; /* Soft pink background */
        }

        .stButton > button {
            background-color: #f7a0c5; /* Soft pink button */
            color: white;
            font-size: 16px;
            border-radius: 6px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            font-weight: bold;
        }

        .stButton > button:hover {
            background-color: #e26f92; /* Slightly darker pink on hover */
        }

        .stSubheader {
            font-size: 18px;
            color: #f7a0c5; /* Soft pink for subheaders */
            font-weight: bold;
            text-align: left;
            border-bottom: 1px solid #f1a7cb;
            padding-bottom: 5px;
        }

        .stTitle {
            color: #f7a0c5; /* Soft pink */
            font-size: 32px;
            text-align: center;
            margin-bottom: 20px;
            border-bottom: 2px solid #f1a7cb;
            padding-bottom: 10px;
        }

        .stText, .stMarkdown {
            font-size: 16px;
            color: #4a4a4a; /* Soft gray text */
            text-align: left;
        }

        .stRadio label {
            font-size: 16px;
            color: #4a4a4a; /* Soft gray text */
            font-weight: 500;
        }

        .stNumberInput input {
            font-size: 16px;
        }

        /* Header styling */
        [data-testid="stHeader"] {
            background-color: #f1c6d4;
            padding: 20px;
            border-bottom: 3px solid #f7a0c5;
        }

        /* Prediction result styling */
        .prediction-result {
            background-color: #fceff4;
            border-left: 4px solid #f7a0c5;
            padding: 10px 15px;
            margin: 20px 0;
            border-radius: 0 6px 6px 0;
        }

        /* Info card styling */
        .info-card {
            background-color: #f9e5f2;
            padding: 15px;
            border-radius: 6px;
            margin: 15px 0;
            border: 1px solid #f1c6d4;
        }

        /* Warning/important result styling */
        .warning-result {
            background-color: #fff4e1;
            border-left: 4px solid #f7c100;
            padding: 10px 15px;
            margin: 20px 0;
            border-radius: 0 6px 6px 0;
        }
    </style>
""", unsafe_allow_html=True)

# Information notice about the tool
st.markdown("""
<div class="info-card">
    <strong>IMPORTANT:</strong> This tool is for risk assessment only and is not a substitute for medical advice. 
    Please consult with a healthcare provider for proper diagnosis and treatment.
</div>
""", unsafe_allow_html=True)

# User Inputs Section
st.header("📋 Patient Information")
st.write("Please enter accurate health information to receive a risk assessment.")

# Input Fields organized by category

# Basic health information
st.subheader("📊 Demographic Information")
age = st.number_input("Age", min_value=0, step=1, key="age")
num_pregnancies = st.number_input("Number of Pregnancies", min_value=0, step=1, key="num_pregnancies")

# Sexual history
st.subheader("🔄 Sexual History")
num_sexual_partners = st.number_input("Number of Sexual Partners", min_value=0, step=1, key="num_sexual_partners")
first_sexual_intercourse = st.number_input("Age at First Sexual Intercourse", min_value=0, step=1, key="first_sexual_intercourse")

# Lifestyle and Contraceptive History
st.subheader("🚬 Lifestyle Factors")
smokes = st.radio("Smoking Status", ("Yes", "No"), key="smokes")
smokes_years = st.number_input("Years of Smoking", min_value=0, step=1, key="smokes_years") if smokes == "Yes" else 0
smokes_packs_per_year = st.number_input("Packs per Year", min_value=0.0, step=0.1, key="smokes_packs_per_year") if smokes == "Yes" else 0

st.subheader("💊 Contraceptive History")
hormonal_contraceptives = st.radio("Hormonal Contraceptive Use", ("Yes", "No"), key="hormonal_contraceptives")
hormonal_contraceptives_years = st.number_input("Years of Hormonal Contraceptive Use", min_value=0, step=1, key="hormonal_contraceptives_years") if hormonal_contraceptives == "Yes" else 0

iud = st.radio("IUD Use", ("Yes", "No"), key="iud")
iud_years = st.number_input("Years of IUD Use", min_value=0, step=1, key="iud_years") if iud == "Yes" else 0

# STDs section with detailed options
st.subheader("🦠 STD History")
stds = st.radio("History of STDs", ("Yes", "No"), key="stds")
stds_number = st.number_input("Number of STDs", min_value=0, step=1, key="stds_number") if stds == "Yes" else 0
stds_diagnosis = st.number_input("Number of STD Diagnoses", min_value=0, step=1, key="stds_diagnosis")  # New input for number of diagnosis

# Diagnosis fields (for Cancer, CIN, HPV, etc.)
st.subheader("🔬 Previous Diagnosis Information")
dx_cancer = st.radio("Previous Cancer Diagnosis", ("Yes", "No"), key="dx_cancer")
dx_cin = st.radio("Previous CIN Diagnosis", ("Yes", "No"), key="dx_cin")
dx_hpv = st.radio("Previous HPV Diagnosis", ("Yes", "No"), key="dx_hpv")
dx_hinselmann = st.radio("Previous Hinselmann Test (Positive)", ("Yes", "No"), key="dx_hinselmann")
dx_schiller = st.radio("Previous Schiller Test (Positive)", ("Yes", "No"), key="dx_schiller")
dx_citology = st.radio("Previous Abnormal Cytology Result", ("Yes", "No"), key="dx_citology")

# Adding the new "Dx" input as a boolean (Yes/No)
dx = st.radio("Do you have any cervical disorder diagnosis?", ("Yes", "No"), key="dx")

# Prepare the input list including the new "Dx" input
input_list = [
    age, num_sexual_partners, first_sexual_intercourse, num_pregnancies, smokes, smokes_years,
    smokes_packs_per_year, hormonal_contraceptives, hormonal_contraceptives_years, iud, iud_years, stds,
    stds_number, stds_diagnosis, dx_cancer, dx_cin, dx_hpv, dx_hinselmann, dx_schiller, dx_citology, dx
]

# Convert "Yes" to 1 and "No" to 0 for all categorical values
for i in range(len(input_list)):
    if isinstance(input_list[i], str):
        input_list[i] = 1 if input_list[i] == "Yes" else 0

# Submit Button with medical styling
if st.button("Generate Risk Assessment", key="submit_button"):
    # Show entered information in a collapsible section
    with st.expander("View Entered Information"):
        for feature, value in zip(
            [
                "Age", "Number of Sexual Partners", "Age at First Sexual Intercourse", "Number of Pregnancies", 
                "Smoking Status", "Years of Smoking", "Packs per Year", "Hormonal Contraceptives", 
                "Years of Hormonal Contraceptive Use", "IUD Use", "Years of IUD Use", "STD History", 
                "Number of STDs", "Number of STD Diagnoses", "Previous Cancer Diagnosis", "Previous CIN Diagnosis", 
                "Previous HPV Diagnosis", "Previous Hinselmann Test (Positive)", "Previous Schiller Test (Positive)", 
                "Previous Abnormal Cytology", "Cervical Disorder Diagnosis"
            ], 
            input_list
        ):
            st.write(f"**{feature}:** {value}")
    
    # Make the prediction
    prediction = model.predict([input_list])
    
    # Display prediction result with appropriate styling
    st.write("-" * 40)
    st.subheader("📊 Risk Assessment Results:")
    
    if prediction == 0:
        st.markdown("""
        <div class="prediction-result">
            <h3 style='color: #2e7d32;'>Low Risk Assessment</h3>
            <p>Based on the information provided, the model predicts a lower risk for cervical cancer (precision: 74%).</p>
            <p><strong>Note:</strong> Regular screening is still recommended as per medical guidelines.</p>
        </div>
        """, unsafe_allow_html=True)
    elif prediction == 1:
        st.markdown("""
        <div class="warning-result">
            <h3 style='color: #c62828;'>Elevated Risk Assessment</h3>
            <p>Based on the information provided, the model predicts a potential elevated risk for cervical cancer (precision: 64%).</p>
            <p><strong>Recommendation:</strong> Please consult with a healthcare provider for proper evaluation and possible further testing.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div style='background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-top: 20px; font-size: 14px;'>
        <strong>Disclaimer:</strong> This tool provides a statistical risk assessment based on the entered data and is not a clinical diagnosis. 
        All individuals should follow recommended screening guidelines regardless of this assessment result.
        Please consult with a healthcare provider for personalized medical advice.
    </div>
    """, unsafe_allow_html=True)

if 'show' not in st.session_state:
    st.session_state.show = False

toggle_button = st.button("Show/Hide Images")

if toggle_button:
    st.session_state.show = not st.session_state.show

# Display images if the state is True
if st.session_state.show:
    # Load images (make sure to replace these with your actual image paths)
    image_paths = ["shap_results/shap_bar_plot.png", "shap_results/shap_decision_plot.png",
                   "shap_results/shap_force_plot_example_1.png", "shap_results/shap_summary_plot.png"]

    # Create a 2x2 layout to display images
    col1, col2 = st.columns(2)

    with col1:
        st.image(image_paths[0], caption="shap_summary_plot", use_container_width=True)
        st.image(image_paths[1], caption="shap_decision_plot", use_container_width=True)

    with col2:
        st.image(image_paths[2], caption="shap_force_plot", use_container_width =True)
        st.image(image_paths[3], caption="shap_summary_plot", use_container_width=True)