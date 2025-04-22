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
#DATA PREPROCCESSING
# Convert 'date' to datetime and extract year/month
df['date'] = pd.to_datetime(df['date'])
df['sale_year'] = df['date'].dt.year
df['sale_month'] = df['date'].dt.month
# Drop columns not useful for prediction
df = df.drop(['id', 'date'], axis=1)

# Check for missing values
print("Missing Values:")
print(df.isnull().sum())

price_cap = df['price'].quantile(0.99)  # Cap at 99th percentile
df = df[df['price'] <= price_cap]

df['sqft_check'] = df['sqft_living'] - (df['sqft_above'] + df['sqft_basement'])
print("Rows where sqft_living != sqft_above + sqft_basement:", (df['sqft_check'] != 0).sum())

df = df.drop('sqft_check', axis=1)  # Drop temporary column

df = pd.get_dummies(df, columns=['zipcode'], drop_first=True)
print("Summary Statistics:")
print(df.describe())

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=False, cmap='coolwarm')





