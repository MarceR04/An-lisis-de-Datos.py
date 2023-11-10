import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Cargar los datos desde el archivo "processed_data.csv"
data = pd.read_csv("processed_data.csv")

# Elimina las columnas DEATH_EVENT y age
X = data.drop(columns=["DEATH_EVENT", "age"])
y = data["age"]

# Codificar la columna Age_Category utilizando one-hot encoding
X = pd.get_dummies(X, columns=["Age_Category"], drop_first=True)

# Ajusta un modelo de regresión lineal
model = LinearRegression()
model.fit(X, y)

# Predice las edades
y_pred = model.predict(X)

# Calcula el error cuadrático medio (MSE)
mse = mean_squared_error(y, y_pred)

print(f"Error Cuadrático Medio (MSE): {mse}")
