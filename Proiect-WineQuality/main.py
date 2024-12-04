import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score

# 1. Analiza exploratorie a datelor
# Citirea setului de date

df = pd.read_csv('winequality.csv')

# Explorarea setului de date
print(df.info())
print(df.describe())
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Heatmap of Feature Correlations")
plt.show()

# 2. Curățarea/preprocesarea datelor
# se verifica valorile lipsa
print(df.isnull().sum())

# se elimina valorile lipsa
df.dropna(inplace=True)

# 3. Împărțirea datelor într-un set de antrenament și un set de testare
X = df.drop(columns=['quality'])
y = df['quality']

# Împărțirea datelor într-un set de antrenament și un set de testare
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Crearea și ajustarea unui model inițial - Regresia Liniară
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Construirea modelului - Prezicerea calității vinului
y_pred = model.predict(X_test)

# Calcularea metricilor de performanță pentru regresia liniară
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")
print(f"R2 Score: {r2}")

# 6. Aplicarea K-means pentru clustering
# Alegerea unui număr de grupuri (clusters)
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X)

# Afișarea centrelor grupurilor și etichetarea datelor
print(f"Cluster Centers: \n{kmeans.cluster_centers_}")
df['Cluster'] = kmeans.labels_

# Evaluarea calității grupărilor folosind Silhouette Score
silhouette_avg = silhouette_score(X, kmeans.labels_)
print(f"Silhouette Score: {silhouette_avg}")

# Vizualizarea grupurilor
sns.scatterplot(x='alcohol', y='volatile acidity', hue='Cluster', data=df, palette='viridis')
plt.title("K-means Clustering of Wine Data")
plt.xlabel("Alcohol")
plt.ylabel("Volatile Acidity")
plt.show()

# Salvarea rezultatelor
df.to_csv('winequality_with_clusters.csv', index=False)

#citeste read-me file pentru modificari
#
#
# este necesara testarea fiecarei cerinte in parte
#
#
