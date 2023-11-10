import pandas as pd
from datasets import load_dataset


dataset = load_dataset("mstz/heart_failure")
df = pd.DataFrame(dataset["train"])
# Verificar los tipos de datos de cada columna
print(df.dtypes)

# Calcular la cantidad de hombres fumadores vs mujeres fumadoras
grouped = df.groupby(["is_male", "is_smoker"]).size().reset_index(name="count")
smokers = grouped[grouped["is_smoker"] == True]
print(smokers)
