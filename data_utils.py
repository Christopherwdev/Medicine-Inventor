import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(data_path):
    data = pd.read_csv(data_path)
    # Assuming the last column is the target variable (efficacy)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    #Feature Scaling (important for neural networks)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, scaler
