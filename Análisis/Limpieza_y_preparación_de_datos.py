import requests
import pandas as pd


def download_csv(url: str, filename: str) -> None:
    response = requests.get(url)
    with open(filename, "w") as f:
        f.write(response.text)


def process_and_save_data(df: pd.DataFrame) -> None:
    # Verificar valores faltantes
    missing_values = df.isnull().sum().any()
    if missing_values:
        print("Existen valores faltantes en el DataFrame.")
        # Puedes decidir cómo manejar los valores faltantes, por ejemplo, reemplazarlos o eliminar las filas con valores faltantes.

    # Verificar filas duplicadas
    duplicate_rows = df.duplicated().sum()
    if duplicate_rows:
        print("Existen filas duplicadas en el DataFrame.")
        # Puedes eliminar las filas duplicadas si es necesario.

    # Verificar y eliminar valores atípicos
    # Puedes aplicar tu lógica para eliminar valores atípicos aquí.

    # Crear columna de categorización por edades
    def categorize_age(age):
        if age <= 12:
            return "Niño"
        elif age <= 19:
            return "Adolescente"
        elif age <= 39:
            return "Jóvenes adulto"
        elif age <= 59:
            return "Adulto"
        else:
            return "Adulto mayor"

    df["Age_Category"] = df["age"].apply(categorize_age)

    # Guardar el resultado como CSV
    df.to_csv("processed_data.csv", index=False)


# Descargar el CSV
url = "https://huggingface.co/datasets/mstz/heart_failure/raw/main/heart_failure_clinical_records_dataset.csv"
download_csv(url, "heart_failure_clinical_records_dataset.csv")

# Cargar el CSV en un DataFrame
df = pd.read_csv("heart_failure_clinical_records_dataset.csv")

# Procesar y guardar los datos
process_and_save_data(df)
