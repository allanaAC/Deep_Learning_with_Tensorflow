#this function is to load data 
import pandas as pd
import numpy as np
import logging
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split

#data_path = "/data/real_estate.csv"
def load_and_preprocess_data(file_path):
    
    try:
        df = pd.read_csv(file_path)
        Y = df.Attrition
        X = df.drop(columns=['Attrition'])
    
        sc = StandardScaler()
        X_scaled = sc.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
        return train_test_split(X_scaled, Y, test_size=0.2, random_state=1, stratify=Y)
    except Exception as e:
        logging.error(" Error in processing data: {}". format(e))
     