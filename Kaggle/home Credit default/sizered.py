# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 11:58:18 2018

@author: ssurya200
"""

import pandas as pd
import numpy as np

df = pd.read_csv('Data/X2.csv')

start_mem_usg = df.memory_usage().sum() / 1024**2 
print("Memory usage of properties dataframe is :",start_mem_usg," MB")
NAlist = [] # Keeps track of columns that have missing values filled in. 
for col in df.columns:
    if df[col].dtype != object:  # Exclude strings

        # Print current column type
        print("******************************")
        print("Column: ",col)
        print("dtype before: ",df[col].dtype)

        # make variables for Int, max and min
        IsInt = False
        mx = df[col].max()
        mn = df[col].min()

        # Integer does not support NA, therefore, NA needs to be filled
        if not np.isfinite(df[col]).all(): 
            NAlist.append(col)
            df[col].fillna(mn-1,inplace=True)  

        # test if column can be converted to an integer
        asint = df[col].fillna(0).astype(np.int64)
        result = (df[col] - asint)
        result = result.sum()
        if result > -0.01 and result < 0.01:
            IsInt = True


        # Make Integer/unsigned Integer datatypes
        if IsInt:
            if mn >= 0:
                if mx < 255:
                    df[col] = df[col].astype(np.uint8)
                elif mx < 65535:
                    df[col] = df[col].astype(np.uint16)
                elif mx < 4294967295:
                    df[col] = df[col].astype(np.uint32)
                else:
                    df[col] = df[col].astype(np.uint64)
            else:
                if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)    

        # Make float datatypes
        else:
            if mn > np.finfo(np.float16).min and mx < np.finfo(np.float16).max:
                df[col] = df[col].astype(np.float16)
            elif mn > np.finfo(np.float32).min and mx < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            elif mn > np.finfo(np.float64).min and mx < np.finfo(np.float64).max:
                df[col] = df[col].astype(np.float64)   

        # Print new column type
        print("dtype after: ",df[col].dtype)
        print("******************************")

# Print final result
print("___MEMORY USAGE AFTER COMPLETION:___")
mem_usg = df.memory_usage().sum() / 1024**2 
print("Memory usage is: ",mem_usg," MB")
print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
# X2_df = reduce_mem_usage(df)
col_list = list(df.columns.values)

store = pd.HDFStore('store.h5')
store['df'] = df
store['df']
print(store)