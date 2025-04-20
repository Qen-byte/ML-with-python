import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# Load the dataset
df = pd.read_csv('Housing.csv')
print("Dataset Info:")
print(df.info())
print("Dataset Info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())
#DATA PREPROCCESSING
# Convert 'date' to datetime and extract year/month
df['date'] = pd.to_datetime(df['date'])





