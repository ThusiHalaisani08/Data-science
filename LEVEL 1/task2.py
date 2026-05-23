from pathlib import Path
import pandas as pd
import numpy as np  
from sklearn.preprocessing import StandardScaler, LabelEncoder

csv_path = Path(__file__).resolve().parent / "iris.csv"
df = pd.read_csv(csv_path)

#FOR HANDLING MISSING VALUES
for col in df.select_dtypes(include="number").columns:
    df[col] = df[col].fillna(df[col].median())
print("Missing values handled.")
print(df.isnull().sum())

#Detect & Remove Outliers (IQR method)
num_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

before = len(df)
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

after = len(df)
print(f"Outliers removed: {before - after} , {after} row/s are remaining")

#Encode Categorical Variable
le = LabelEncoder()
df["species"] = le.fit_transform(df["species"])

#ONE-HOT ENCODING
df = pd.get_dummies(df, columns=["species"], prefix="species")
print("\n Species encoded")
print(df.head(3))

#STANDARDIZE NUMERICAL FEATURES
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
print("\n Numerical columns standardized.")
df = df.dropna(how="all") 

df.to_csv("iris_cleaned.csv", index=False)
print("\n SAVED CLEANED DATA TO iris_cleaned.csv")

