import matplotlib.pyplot as plt
import pandas as pd

# Cargar el CSV en un DataFrame
df = pd.read_csv("processed_data.csv")

# Especificar las categorías a analizar y las etiquetas personalizadas
categories_to_plot = ["anaemia", "diabetes", "smoking", "DEATH_EVENT"]
categories_labels = ["Anémicos", "Diabéticos", "Fumadores", "Muertos"]

# Crear una figura con subplots en una sola fila
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle("Distribuciones")

# Generar gráficas de torta para cada categoría
for i, category in enumerate(categories_to_plot):
    ax = axes[i]

    category_counts = df[category].value_counts()
    labels = ["Sí", "No"]  # Etiquetas "Sí" y "No"
    sizes = category_counts.values
    colors = ["salmon", "cadetblue"]
    explode = (0, 0)

    ax.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        shadow=True,
        startangle=90,
    )
    ax.axis("equal")  # Aspecto igual para que sea un círculo

    # Agregar la etiqueta en la parte inferior
    ax.text(
        0.5,
        -0.15,
        categories_labels[i],
        horizontalalignment="center",
        transform=ax.transAxes,
    )

# Ajustar el espaciado entre subplots
plt.tight_layout(rect=[0, 0, 0, 0])

# Mostrar el gráfico
plt.show()
