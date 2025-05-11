import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

import os

# Find the diabetes dataset
diabetes_dataset_files = [file for file in os.listdir() if 'diabetes' in file and '.csv' in file]
print(f'Found {len(diabetes_dataset_files)} files: {diabetes_dataset_files}')

# Loading the diabetes dataset into a pandas dataframe
diabetes_dataset = pd.read_csv(diabetes_dataset_files[0])
print(diabetes_dataset.head())