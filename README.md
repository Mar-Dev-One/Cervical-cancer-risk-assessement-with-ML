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

### 1. Model Generation (`genModels.py`)
- Processes raw cervical cancer risk factor data
- Applies data preprocessing techniques
- **Handles class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)**
  - The dataset contains an imbalance between positive and negative cases
  - SMOTE generates synthetic samples of the minority class to balance the dataset
  - This approach improves model performance and reduces bias toward the majority class
- Trains various machine learning models
- **XGBoost model demonstrated the best overall performance** among all tested models
- Evaluates model performance
- Saves trained models for later use

### 2. Web Application (`main.py`)
- Built with Streamlit framework
- Loads pre-trained machine learning models
- Collects detailed patient information
- Predicts cervical cancer risk using the selected model
- Provides visual explanations using SHAP
- Displays risk assessment in an easy-to-understand format

## Key Findings
- **Feature Importance**: The Schiller test result was identified as the most important feature for predicting cervical cancer risk
- The SHAP analysis reveals clear patterns in how different risk factors contribute to the overall prediction
- Model achieved good performance metrics with ~74% precision for negative cases and ~64% precision for positive cases

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

## Prompt Engineering Insights
Prompt engineering provided several key insights for this project:

1. **Effective data visualization**: Careful crafting of the user interface prompts helped balance technical accuracy with accessibility, making complex medical information understandable to non-specialists
2. **Clear risk communication**: Iterative refinement of how risk assessments are presented ensured users understand results without causing undue anxiety
3. **Ethical considerations**: Designing appropriate disclaimers and medical guidance prompts was essential for responsible deployment of an AI health tool
4. **User engagement**: Structuring the input form with logical groupings and progressive disclosure improved completion rates and data quality

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
- imblearn library for handling class imbalance