import pandas as pd
from datasets import load_dataset


def calcular_promedio_edades():
    # Cargar el conjunto de datos
    dataset = load_dataset("mstz/heart_failure")

    # Convertir los datos en un DataFrame de Pandas
    df = pd.DataFrame(dataset["train"])

    # Separar el dataframe en dos diferentes
    df_dead = df.groupby("is_dead").get_group(1)
    df_alive = df.groupby("is_dead").get_group(0)

    # Calcular el promedio de las edades
    promedio_edades = df["age"].mean()

    return promedio_edades
