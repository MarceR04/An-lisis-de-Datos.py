from datasets import load_dataset
import numpy as np


#Descargar el conjunto de datos
dataset = load_dataset("mstz/heart_failure")
data = dataset["train"]
edades = data["age"]
print(edades)

#Convertir la lista de edades a un arreglo de NumPy
edades_np = np.array(edades)
print(edades_np)
#Calcular el promedio de edad de las personas participantes en el estudio.
promedio_edad = np.mean(edades_np)
print(promedio_edad)







