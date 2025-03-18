##
# @file momory_opt.py
# @brief A utility module for optimizing DataFrame memory usage.
#
# This module provides functionality to reduce memory consumption of pandas DataFrames
# by downcasting numerical columns to more memory-efficient data types.
#

import pandas as pd

##
# @brief Optimizes memory usage of a pandas DataFrame by downcasting numerical columns.
#
# This function iterates over all columns in the DataFrame and optimizes memory usage
# by downcasting numerical values:
# - float64 columns are downcasted to float32 or smaller where possible
# - int64 columns are downcasted to int32, int16, or int8 where possible
#
# @param df A pandas DataFrame to optimize
# @return The memory-optimized pandas DataFrame
#
def optimize_memory_usage(df):
   """
   Iterates over all columns in the DataFrame and optimizes memory usage 
   by downcasting numerical columns (float64 and int64) to more efficient types.
   
   Parameters:
   df (pd.DataFrame): The DataFrame to optimize.

   Returns:
   pd.DataFrame: The DataFrame with optimized memory usage.
   """
   for col in df.columns:
       col_type = df[col].dtype
       
       if col_type == 'float64':
           # Try to downcast float64 to float32 or smaller if possible
           df[col] = pd.to_numeric(df[col], downcast='float')
       
       elif col_type == 'int64':
           # Try to downcast int64 to int32, int16, or int8 if possible
           df[col] = pd.to_numeric(df[col], downcast='integer')
   
   return df