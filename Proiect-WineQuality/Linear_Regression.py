import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# se citesc datele
df = pd.read_csv('winequality.csv')

# preprocesare
df.dropna(inplace=True)
X = df.drop(columns=['quality'])
y = df['quality']

# impartirea datelor în seturi de antrenament si testare
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# crearea si antrenarea modelului de regresie liniara
model = LinearRegression()
model.fit(X_train, y_train)

# prezicerea si evaluarea modelului
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# afisarea metricilor
print(f"Mean Squared Error (MSE): {mse}")
print(f"R2 Score: {r2}")
