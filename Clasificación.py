import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Cargar el dataset (asegúrate de ajustar la ruta del archivo CSV)
df = pd.read_csv("processed_data.csv")

# Eliminar la columna 'Age_Category'
df = df.drop(columns=["Age_Category"])

# Graficar la distribución de clases
plt.figure(figsize=(8, 6))
df["DEATH_EVENT"].value_counts().plot(kind="bar", color="crimson", edgecolor="black")
plt.title("Distribución de Clases")
plt.xlabel("Clase")
plt.ylabel("Frecuencia")
plt.xticks(rotation=0)
plt.show()


# Realizar la partición del dataset en conjunto de entrenamiento y test de manera estratificada
X_train, X_test, y_train, y_test = train_test_split(
    df.drop("DEATH_EVENT", axis=1),  # Features
    df["DEATH_EVENT"],  # Labels
    test_size=0.2,  # Porcentaje para test
    random_state=42,  # Semilla para reproducibilidad
    stratify=df["DEATH_EVENT"],  # Estratificación basada en las etiquetas de clase
)

# Definir el clasificador RandomForest y los hiperparámetros a ajustar
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

rf_model = RandomForestClassifier(random_state=42)

# Realizar la búsqueda en cuadrícula para encontrar los mejores hiperparámetros
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring="accuracy")
grid_search.fit(X_train, y_train)

# Obtener los mejores hiperparámetros encontrados
best_params = grid_search.best_params_
print("Mejores hiperparámetros:", best_params)

# Crear el modelo con los mejores hiperparámetros
best_rf_model = RandomForestClassifier(random_state=42, **best_params)
best_rf_model.fit(X_train, y_train)

# Hacer predicciones en el conjunto de test
y_pred = best_rf_model.predict(X_test)

# Calcular la accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy del Modelo: {accuracy:.4f}")
