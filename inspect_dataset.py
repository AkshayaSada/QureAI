import pandas as pd
import os

file_path = "datasets/disease_prediction/Training.csv"

if not os.path.exists(file_path):
    print("❌ Training.csv not found!")
    print("Check that the file is inside:")
    print("../datasets/disease_prediction/")
    exit()

data = pd.read_csv(file_path)

print("\n==============================")
print("       QUREAI DATASET")
print("==============================")

print("\nDataset shape:")
print(data.shape)

print("\nColumn names:")
print(data.columns.tolist())

print("\nFirst 5 rows:")
print(data.head())

print("\nMissing values:")
print(data.isnull().sum())

if "prognosis" in data.columns:
    print("\nDisease column found: prognosis")
    print("\nNumber of unique diseases:")
    print(data["prognosis"].nunique())

    print("\nDisease names:")
    diseases = sorted(data["prognosis"].unique())

    for i, disease in enumerate(diseases, start=1):
        print(f"{i}. {disease}")
else:
    print("\n⚠️ 'prognosis' column was not found.")
    print("Available columns:")
    print(data.columns.tolist())