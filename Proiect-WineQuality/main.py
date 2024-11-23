import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from imblearn.over_sampling import SMOTE  # Importăm SMOTE pentru oversampling


os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# 1. Încărcarea și analiza exploratorie a datelor
df = pd.read_csv('winequality.csv')

# Statistici descriptive
print(df.describe())

# Distribuția variabilei țintă
plt.figure()
sns.histplot(df['quality'], bins=10, kde=True)
plt.title("Distribuția calității vinului")
plt.xlabel("Quality")
plt.ylabel("Frequency")
plt.show()

# Matricea de corelație
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Matricea de corelație")
plt.show()

# 2. Curățarea datelor
# Verificăm valori lipsă
print("Valori lipsă pe coloane:\n", df.isnull().sum())

# Eliminăm outlierii folosind metoda IQR
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

# 3. Împărțirea datelor în seturi de antrenament și testare
X = df.drop('quality', axis=1)
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Verificarea și ajustarea pentru dezechilibrul claselor
# Verificăm distribuția claselor
sns.histplot(y_train)
plt.title("Distribuția claselor din setul de antrenament")
plt.show()

# Aplicăm SMOTE pentru a echilibra setul de date de antrenament
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 5. Crearea și antrenarea modelului MLP
mlp = MLPClassifier(max_iter=1000, hidden_layer_sizes=(100,), random_state=42)
mlp.fit(X_train_resampled, y_train_resampled)

# 6. Prezicerea și evaluarea performanței modelului
y_pred = mlp.predict(X_test)

# Raportul de clasificare cu zero_division=1 pentru a evita eroarea de precizie
print("MLP Classification Report:")
print(classification_report(y_test, y_pred, zero_division=1))

# Alte metrice de evaluare
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# 7. Evaluarea performanței modelului cu MSE și R2 (pentru regresie liniară)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred_lr)
r2 = r2_score(y_test, y_pred_lr)
print(f"\nLinear Regression - MSE: {mse:.4f}, R2 Score: {r2:.4f}")

# 8. Clustering cu K-means
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)
df['Cluster'] = clusters

# Vizualizarea clusterelor
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='alcohol', y='pH', hue='Cluster', palette='viridis')
plt.title("Clusterele K-means pe baza caracteristicilor 'alcohol' și 'pH'")
plt.show()
