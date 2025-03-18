# Cervical Cancer Risk Assessment Tool
[![Streamlit CI/CD](https://github.com/Mar-Dev-One/Cervical-cancer-risk-assessement-with-ML/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/Mar-Dev-One/Cervical-cancer-risk-assessement-with-ML/actions/workflows/ci-cd.yml)

## Overview
This project is a web-based application that uses machine learning models to assess cervical cancer risk based on patient health information. The tool provides a user-friendly interface where individuals can input their health data and receive a risk assessment using one of several machine learning models.

## Contributors
- Alahyane Marouane
- Azouggagh Sara
- Es-salmy Zakaria
- Anaouch Mohamed
- Afoud Yasine

## Features
- **Multiple ML Models**: Choose between Random Forest, GBoost, SVM, and CatBoost classifiers
- **Interactive UI**: User-friendly medical-themed interface built with Streamlit
- **Risk Assessment**: Receive low or elevated risk assessment based on your health information
- **Model Interpretability**: View SHAP (SHapley Additive exPlanations) visualizations to understand influential factors
- **Responsive Design**: Accessible on various devices with a clean, medical-themed appearance

## Technical Components
The application consists of two main parts:

### 1. Model Generation (`src/genModels.py`)
- Processes raw cervical cancer risk factor data
- Applies data preprocessing techniques
- Handles class imbalance using SMOTE
- Trains various machine learning models
- Evaluates model performance
- Saves trained models for later use

### 2. Web Application (`src/main.py`)
- Built with Streamlit framework
- Loads pre-trained machine learning models
- Collects detailed patient information
- Predicts cervical cancer risk using the selected model
- Provides visual explanations using SHAP
- Displays risk assessment in an easy-to-understand format

## Data
The application uses a cervical cancer risk factors dataset with the following key features:
- Demographic information
- Sexual history
- Lifestyle factors
- Contraceptive history
- STD history
- Previous diagnosis information

## Installation & Setup

### Prerequisites
- Python 3.7+
- pip package manager

### Installation Steps
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/cervical-cancer-risk-assessment.git
   cd cervical-cancer-risk-assessment
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Generate the models (if not already generated):
   ```
   python genModels.py
   ```

4. Run the web application:
   ```
   streamlit run main.py
   ```

## Usage
1. Open the application in your web browser
2. Select your preferred machine learning model
3. Fill in your health information in the form
4. Click "Generate Risk Assessment"
5. View your risk assessment result
6. Optionally, view SHAP visualizations for more insights

## Important Notes
- This tool is for risk assessment only and is not a substitute for medical advice
- Regular cervical cancer screening is recommended regardless of the risk assessment
- Consult with healthcare providers for proper diagnosis and treatment

## Requirements
- streamlit
- joblib
- shap
- matplotlib
- numpy
- pandas
- scikit-learn
- imbalanced-learn

## Future Improvements
- Add more machine learning models
- Implement real-time SHAP explanations
- Enhance mobile responsiveness
- Add multi-language support
- Incorporate more detailed risk factors


## Acknowledgments
- Data source: [Insert data source if applicable]
- SHAP library for model interpretability
- Streamlit for the web application framework