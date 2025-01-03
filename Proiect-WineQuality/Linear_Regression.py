import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Citirea datelor
df = pd.read_csv('winequality.csv')

# Preprocesare
df.dropna(inplace=True)
X = df.drop(columns=['quality'])
y = df['quality']

# Împărțirea datelor în seturi de antrenament și testare
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crearea și antrenarea modelului de regresie liniară
model = LinearRegression()
model.fit(X_train, y_train)

# Prezicerea și evaluarea modelului
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Afișarea metricilor
print(f"Mean Squared Error (MSE): {mse}")
print(f"R2 Score: {r2}")
