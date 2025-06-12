from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import os
import sys
sys.path.append(os.path.abspath(r'D:\Guvi_Project\Personalized Learning Assistant\src'))
from Model_Training import pickle_dump
import pandas as pd

def normalisation(data, name):
    # Initialize StandardScaler and fit-transform the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Save the scaler for future use
    pickle_dump(scaler, name)
    
    # Return scaled data as DataFrame with original columns
    return pd.DataFrame(scaled_data, columns=data.columns)

def oversampling(x, y):
    # Initialize SMOTE for oversampling minority class
    sm = SMOTE(random_state=42)
    
    # Resample features and target
    x_resampled, y_resampled = sm.fit_resample(x, y)
    
    # Return as DataFrame and Series preserving original column and name
    return pd.DataFrame(x_resampled, columns=x.columns), pd.Series(y_resampled, name=y.name)

def label_encoder(data, name):
    # Initialize LabelEncoder and fit-transform the data
    le = LabelEncoder()
    label = le.fit_transform(data)
    
    # Save the encoder for later use
    pickle_dump(le, name)
    
    # Return encoded labels as DataFrame with original name as column
    return pd.DataFrame(label, columns=[data.name])
