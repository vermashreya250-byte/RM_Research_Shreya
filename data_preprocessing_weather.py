import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
sns.set_style('whitegrid')
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pickle
import os

csv_path = r"C:\Users\Shreya\OneDrive\Desktop\RM-research\RM_Research_Shreya\weatherAUS.csv"
print("Exists?", os.path.isfile(csv_path))  # Should now print True


import os
import pandas as pd
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "weatherAUS.csv")
df = pd.read_csv(csv_path)
print("CSV loaded successfully!")
print(" Dataset loaded")
df.head()
print(f'Shape of the dataset is {df.shape}')
print(f'There are {df.shape[0]} rows and {df.shape[1]} columns')
print(f'The data types of the dataset are {df.dtypes}')
print("\n Dataset Info")
df.info()
print("\n Summary Statistics ")
df.describe(include='all')
print("\n Checking for missing values ")
print(df.isnull().sum().sort_values(ascending=False))
round(df.isnull().sum()/df.shape[0]*100,2).sort_values(ascending=False)
print("Before removing duplicates:", df.shape)
df.drop_duplicates(inplace=True)
print("After removing duplicates:", df.shape)
df['MinTemp'] = df['MinTemp'].fillna(df['MinTemp'].median())
df['MaxTemp'] = df['MaxTemp'].fillna(df['MaxTemp'].median())
df['Rainfall'] = df['Rainfall'].fillna(method='bfill')
df['Evaporation'] = df['Evaporation'].fillna(method='ffill')
df['Evaporation'] = df['Evaporation'].fillna(method='bfill')
df['Sunshine'] = df['Sunshine'].fillna(method='ffill')
df['Sunshine'] = df['Sunshine'].fillna(method='bfill')
df['WindGustDir'] = df['WindGustDir'].fillna(method='bfill')
df['WindGustSpeed'] = df['WindGustSpeed'].fillna(method='bfill')
df['WindDir9am'] = df['WindDir9am'].fillna(method='ffill')
df['WindDir3pm'] = df['WindDir3pm'].fillna(method='ffill')
df['WindSpeed9am'] = df['WindSpeed9am'].fillna(df['WindSpeed9am'].median())
df['WindSpeed3pm'] = df['WindSpeed3pm'].fillna(df['WindSpeed3pm'].median())
df['Humidity9am'] = df['Humidity9am'].fillna(method='ffill')
df['Humidity3pm'] = df['Humidity3pm'].fillna(method='ffill')
df['Pressure9am'] = df['Pressure9am'].fillna(method='bfill')
df['Pressure3pm'] = df['Pressure3pm'].fillna(method='bfill')
df['Cloud9am'] = df['Cloud9am'].fillna(method='ffill')
df['Cloud9am'] = df['Cloud9am'].fillna(method='bfill')
df['Cloud3pm'] = df['Cloud3pm'].fillna(method='ffill')
df['Cloud3pm'] = df['Cloud3pm'].fillna(method='bfill')
df['Temp9am'] = df['Temp9am'].fillna(df['Temp9am'].median())
df['Temp3pm'] = df['Temp3pm'].fillna(df['Temp3pm'].median())
df['RainToday'] = df['RainToday'].fillna(method='ffill')
df.columns = df.columns.str.strip()
if 'row ID' in df.columns:
    df.drop(columns=['row ID'], inplace=True)
else:
    print("Column 'row ID' not found, skipping drop.")
df_encoded = pd.get_dummies(df, drop_first=True)
print("Shape after encoding:", df_encoded.shape)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# select only float columns
float_cols = df.select_dtypes(include=['float64']).columns
df[float_cols] = scaler.fit_transform(df[float_cols])
df.head()
df_encoded.to_csv("weatherAUS_cleaned.csv", index=False)
print("Cleaned dataset saved as 'weatherAUS_cleaned.csv'")
df.to_csv("weatherAUS_cleaned.csv", index=False)

