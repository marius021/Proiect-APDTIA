import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt

# Citirea datelor
df = pd.read_csv('winequality.csv')

# Preprocesare
df.dropna(inplace=True)
X = df.drop(columns=['quality'])

# Aplicarea K-means pentru clustering
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X)

# Afișarea rezultatelor clusteringului
df['Cluster'] = kmeans.labels_
print(f"Cluster Centers: \n{kmeans.cluster_centers_}")

# Calcularea Silhouette Score
silhouette_avg = silhouette_score(X, kmeans.labels_)
print(f"Silhouette Score: {silhouette_avg}")

# Sampling pentru reducerea numărului de puncte afișate
sampled_df = df.sample(frac=0.1, random_state=42)  # ia 10% din date

# Calcularea centroidelor
centroids = kmeans.cluster_centers_

# Vizualizarea clusterelor cu sampling și evidențierea centroidelor
sns.scatterplot(
    x='alcohol', y='volatile acidity', hue='Cluster', 
    data=sampled_df, palette='viridis', alpha=0.7
)
plt.scatter(
    centroids[:, X.columns.get_loc('alcohol')], 
    centroids[:, X.columns.get_loc('volatile acidity')], 
    c='red', marker='X', s=50, label='Centroids'
)
plt.title("K-means Clustering with Sampled Data and Centroids")
plt.xlabel("Alcohol")
plt.ylabel("Volatile Acidity")
plt.legend()
plt.show()

# Salvarea rezultatelor
df.to_csv('winequality_with_clusters.csv', index=False)