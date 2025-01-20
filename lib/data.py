from enum import Enum

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


class Data(Enum):
    IRIS = 'iris'
    WINE = 'wine'
    GLASS = 'glass'
    WHOLESALE = 'wholesale'
    RAISIN = 'raisin'
    HCV = 'hcv'
    YEAST = 'yeast'

    
def get_data(data, binary_target=False):
    if data == Data.IRIS:
        return load_iris_(binary_target)
    if data == Data.WINE:
        return load_wine_(binary_target)
    if data == Data.GLASS:
        return load_glass(binary_target)
    if data == Data.WHOLESALE:
        return load_wholesale(binary_target)
    if data == Data.RAISIN:
        return load_raisin(binary_target)
    if data == Data.HCV:
        return load_hcv(binary_target)
    if data == Data.YEAST:
        return load_yeast(binary_target)
    raise ValueError('Invalid data type')


def get_binary_target(y):
    """
    return 1 if y is 1, -1 otherwise
    """
    return (y == 1).astype(int) * 2 - 1


def extract_X_y_from_sklearn_data(data, binary_target):
    X = data.data
    y = data.target
    if binary_target:
        y = get_binary_target(y)
    return X, y


def load_iris_(binary_target):
    data = load_iris()
    return extract_X_y_from_sklearn_data(data, binary_target)


def load_wine_(binary_target):
    data = load_wine()
    return extract_X_y_from_sklearn_data(data, binary_target)


def load_glass(binary_target):
    data_path = './datasets/glass_identification/glass.data'
    columns = ["Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Type"]
    data = pd.read_csv(data_path, header=None, names=columns)
    data.drop(columns=['Id'], inplace=True)

    if binary_target:
        data.loc[data['Type'] == 1, 'Type'] = 1
        data.loc[data['Type'] != 1, 'Type'] = -1

    X = data.drop('Type', axis=1).values
    y = data['Type'].values
    
    return X, y

def load_wholesale(binary_target):
    data_path = './datasets/wholesale_customers/Wholesale_customers_data.csv'
    data = pd.read_csv(data_path)
    X = data.drop(columns=['Channel', 'Region']).values  # Example of features
    y = data['Channel'].values  # Example target (can be modified as needed)
    
    y[y == 2] = -1  # Change target to -1 if it is 2
    
    return X, y

def load_raisin(binary_target):
    data_path = './datasets/raisin_dataset/raisin_dataset.csv'
    data = pd.read_csv(data_path)
    X = data.drop(columns=['Class']).values  # Example of features
    y = data['Class'].values  # Example target (can be modified as needed)
    
    y = np.where(y == 'Besni', 1, -1).astype(int)
    
    return X, y

def load_hcv(binary_target):
    data_path = './datasets/hcv/hcvdat0.csv'
    data = pd.read_csv(data_path)

    # Drop non-numeric and unnecessary columns
    if "Unnamed: 0" in data.columns:
        data = data.drop(columns=["Unnamed: 0"])  # Drop unnecessary index column

    data["Sex"] = data["Sex"].apply(lambda x: 1 if x.lower() == "m" else 0)
    
    data = data.dropna()
    
    X = data.drop(columns=["Category"]).copy()
    y = data["Category"].values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    
    return X.values, y

def load_yeast(binary_target):
    column_names = [
        "Sequence Name", "mcg", "gvh", "alm", "mit", "erl", "pox", "vac", "nuc", "Class"
    ]
    data = pd.read_csv('./datasets/yeast/yeast.data', sep="\s+", names=column_names)
    
    # Drop the "Sequence Name" column (not a feature)
    data = data.drop(columns=["Sequence Name"])
    
    # Separate features and target
    X = data.drop(columns=["Class"])
    y = data["Class"]

    # Ensure features are numeric
    X = X.astype(float)

    # Convert target 'Class' to integers starting from 0
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    return X.values, y