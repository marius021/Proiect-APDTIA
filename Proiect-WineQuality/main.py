import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score

# Încarcă setul de date
df = pd.read_csv('winequality.csv')

# Statistici descriptive
print(df.describe())

# Distribuția variabilei de ieșire `quality`
sns.histplot(df['quality'], bins=10, kde=True)
plt.title("Distribuția calității vinului")
plt.xlabel("Quality")
plt.ylabel("Frequency")
plt.show()

# Matricea de corelație între variabile
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Matricea de corelație")
plt.show()


# Verifică valorile lipsă
print("Valori lipsă pe coloane:\n", df.isnull().sum())

# Înlocuiește valorile lipsă, dacă există, cu media coloanei respective
df.fillna(df.mean(), inplace=True)

# Detectare outlieri folosind metoda IQR
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]


