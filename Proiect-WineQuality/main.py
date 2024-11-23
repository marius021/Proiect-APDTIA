# Importăm bibliotecile necesare
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

# 1. Încărcarea și analiza exploratorie a datelor
df = pd.read_csv('winequality.csv')

# Statistici descriptive
print(df.describe())

# Distribuția variabilei țintă
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

# 4. Crearea și ajustarea modelelor
# a. KNN
best_k = 1
best_knn_accuracy = 0
for k in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"KNN k={k}, Accuracy={acc:.4f}")
    if acc > best_knn_accuracy:
        best_k = k
        best_knn_accuracy = acc

# b. SVM
best_kernel = ''
best_svm_accuracy = 0
for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
    svm = SVC(kernel=kernel)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"SVM Kernel={kernel}, Accuracy={acc:.4f}")
    if acc > best_svm_accuracy:
        best_kernel = kernel
        best_svm_accuracy = acc

# c. MLP
best_mlp_config = {}
best_mlp_accuracy = 0
for max_iter in [200, 400, 600]:
    for hidden_layer_sizes in [(50,), (100,), (50, 50)]:
        mlp = MLPClassifier(max_iter=max_iter, hidden_layer_sizes=hidden_layer_sizes)
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"MLP max_iter={max_iter}, hidden_layer_sizes={hidden_layer_sizes}, Accuracy={acc:.4f}")
        if acc > best_mlp_accuracy:
            best_mlp_config = {'max_iter': max_iter, 'hidden_layer_sizes': hidden_layer_sizes}
            best_mlp_accuracy = acc

# 5. Construirea modelelor finale
final_knn = KNeighborsClassifier(n_neighbors=best_k)
final_knn.fit(X_train, y_train)

final_svm = SVC(kernel=best_kernel)
final_svm.fit(X_train, y_train)

final_mlp = MLPClassifier(**best_mlp_config)
final_mlp.fit(X_train, y_train)

# 6. Testarea și validarea modelelor
print("\n### Validarea modelelor ###\n")

# KNN
y_pred_knn = final_knn.predict(X_test)
print("KNN Classification Report:")
print(classification_report(y_test, y_pred_knn))

# SVM
y_pred_svm = final_svm.predict(X_test)
print("SVM Classification Report:")
print(classification_report(y_test, y_pred_svm))

# MLP
y_pred_mlp = final_mlp.predict(X_test)
print("MLP Classification Report:")
print(classification_report(y_test, y_pred_mlp))

# Linear Regression (opțional, pentru predicția quality)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred_lr)
r2 = r2_score(y_test, y_pred_lr)
print(f"\nLinear Regression - MSE: {mse:.4f}, R2 Score: {r2:.4f}")

# Clustering (K-means)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)
df['Cluster'] = clusters

# Vizualizarea clusterelor
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='alcohol', y='pH', hue='Cluster', palette='viridis')
plt.title("Clusterele K-means pe baza caracteristicilor 'alcohol' și 'pH'")
plt.show()
