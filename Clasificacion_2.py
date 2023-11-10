import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

# Cargar el dataset (ajusta la ruta del archivo CSV según tu configuración)
df = pd.read_csv("processed_data.csv")

# Eliminar la columna 'categoria_edad'
df = df.drop(columns=["age"])

# Codificar variables categóricas usando one-hot encoding
df_encoded = pd.get_dummies(df, columns=["Age_Category"], drop_first=True)

# Realizar la partición del dataset en conjunto de entrenamiento y test de manera estratificada
X_train, X_test, y_train, y_test = train_test_split(
    df_encoded.drop("DEATH_EVENT", axis=1),  # Features
    df_encoded["DEATH_EVENT"],  # Labels
    test_size=0.2,  # Porcentaje para test
    random_state=42,  # Semilla para reproducibilidad
    stratify=df_encoded[
        "DEATH_EVENT"
    ],  # Estratificación basada en las etiquetas de clase
)

# Ajustar un Random Forest con diferentes valores de n_estimators y max_depth
# Puedes ajustar estos valores según tus necesidades
n_estimators_values = [50, 100, 150]
max_depth_values = [None, 10, 20]

best_f1 = 0
best_params = None

for n_estimators in n_estimators_values:
    for max_depth in max_depth_values:
        # Crear y ajustar el modelo
        rf_model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
        rf_model.fit(X_train, y_train)

        # Hacer predicciones en el conjunto de test
        y_pred = rf_model.predict(X_test)

        # Calcular el F1-Score
        f1 = f1_score(y_test, y_pred)

        # Actualizar los mejores parámetros si se encuentra un modelo mejor
        if f1 > best_f1:
            best_f1 = f1
            best_params = {"n_estimators": n_estimators, "max_depth": max_depth}

# Imprimir los mejores parámetros y su F1-Score asociado
print("Mejores Parámetros:")
print(best_params)
print(f"Mejor F1-Score: {best_f1:.2f}")

# Ajustar el modelo Random Forest con los mejores parámetros
best_rf_model = RandomForestClassifier(
    n_estimators=best_params["n_estimators"],
    max_depth=best_params["max_depth"],
    random_state=42,
)
best_rf_model.fit(X_train, y_train)

# Hacer predicciones en el conjunto de test
y_pred_rf = best_rf_model.predict(X_test)

# Calcular la matriz de confusión
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
print("Matriz de Confusión:")
print(conf_matrix_rf)

# Calcular el F1-Score y el Accuracy en el conjunto de test
accuracy_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

# Imprimir los resultados
print(f"Accuracy del Random Forest: {accuracy_rf:.2f}")
print(f"F1-Score del Random Forest: {f1_rf:.2f}")
