##
# @file dataProcessing.py
# @brief Data preprocessing utilities for handling missing values in medical/health datasets.
#
# This module provides functions to clean and impute missing values in datasets 
# containing demographic and health-related information. It specifically handles
# STD-related columns, numeric conversions, and various imputation strategies
# based on grouping variables like age and number of sexual partners.
#
# @author Unknown
# @date Unknown
#

import numpy as np
import pandas as pd

from math import floor

import seaborn as sns

import matplotlib.pyplot as plt


##
# @brief Fills NaN values in a specific column with the mean value from the same age group.
#
# @param row The current row being processed
# @param df_filtered The filtered dataframe containing only valid age groups
# @param column The column name where NaN values should be filled
#
# @return The original value if not NaN, otherwise the age group's mean value
#
def fill_nan_with_group_mean(row, df_filtered, column):
    if pd.isna(row[column]):

        mean_value = df_filtered[df_filtered['Age'] == row['Age']][column].mean()
        if column != 'Hormonal Contraceptives (years)':
            return floor(mean_value)
        else:
            return mean_value
    else:
        return row[column]
    
##
# @brief Fills NaN values in a group with the mode (most common value) of that group.
#
# If no mode exists (empty group), defaults to filling with True.
#
# @param group A pandas Series or DataFrame group containing values
#
# @return The group with NaN values filled with the mode or fallback value
#
def fill_with_mode(group):
    # Try to get the mode of the group
    mode_value = group.mode()
    
    # If there is a mode, fill with that, otherwise fill with a fallback value (e.g., the global mode or a default value)
    if not mode_value.empty:
        return group.fillna(mode_value[0])
    else:
        # Provide a fallback value in case of empty groups (can be set to True or False or the global mode)
        return group.fillna(True)  # Example fallback value: True

##
# @brief Main data processing function that handles cleaning and imputation of a medical dataset.
#
# This function performs the following operations:
#   1. Removes specific STD-related time columns
#   2. Removes columns that start with 'STDs:' except for 'STDs: Number of diagnosis'
#   3. Converts all columns to numeric where possible
#   4. Filters for age groups with more than one observation
#   5. Fills missing numeric values using age-specific means
#   6. Imputes binary columns using group-specific modes
#   7. Sets missing pregnancy counts to zero
#   8. Saves the processed data to a CSV file
#
# @param df The input pandas DataFrame containing the medical/health data
#
# @return The processed pandas DataFrame with imputed values
#
def process_data(df : pd.DataFrame):
    df.drop('STDs: Time since last diagnosis', axis=1, inplace=True)
    df.drop('STDs: Time since first diagnosis', axis=1, inplace=True)

    df = df.loc[:, ~((df.columns.str.startswith('STDs:')) & (df.columns != 'STDs: Number of diagnosis'))]

    df = df.apply(pd.to_numeric, errors='coerce')

    age_counts = df['Age'].value_counts()

    valid_ages = age_counts[age_counts > 1].index

    df_filtered = df[df['Age'].isin(valid_ages)]

    columns = ['Number of sexual partners', 'First sexual intercourse', 'Num of pregnancies', 'Hormonal Contraceptives (years)', 'IUD (years)', 'Smokes (years)', 'STDs (number)', 'Smokes (packs/year)']
    for column in columns:
        df_filtered[column] = df_filtered.apply(lambda row: fill_nan_with_group_mean(row, df_filtered, column), axis=1)


    df.update(df_filtered)

    # Fill NaN values in 'Hormonal Contraceptives' based on 'Number of Sexual Relations'
    df['Hormonal Contraceptives'] = df.groupby('Number of sexual partners')['Hormonal Contraceptives'].transform(fill_with_mode)
    df['IUD'] = df.groupby('Number of sexual partners')['IUD'].transform(fill_with_mode)
    df['STDs'] = df.groupby('Number of sexual partners')['STDs'].transform(fill_with_mode)
    df['Smokes'] = df.groupby('Age')['Smokes'].transform(fill_with_mode)

    df.loc[df['Num of pregnancies'].isna(), 'Num of pregnancies'] = 0

    df.to_csv('../data/output.csv', index=False)  # index=False to exclude the index from being saved
    
    return df