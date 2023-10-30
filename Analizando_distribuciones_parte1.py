import matplotlib.pyplot as plt
import pandas as pd

plt.figure(figsize=(8, 6))  # Tamaño del gráfico (opcional)
df = pd.read_csv("processed_data.csv")

# Crear el histograma
plt.hist(df["age"], bins=20, color="blue", alpha=0.7, edgecolor="black")

# Personaliza el gráfico
plt.title("Distribución de Edades")
plt.xlabel("Edad")
plt.ylabel("Frecuencia")
plt.grid(axis="y", linestyle="", alpha=0.7)
plt.show()


def generate_grouped_bar_chart(df, categories, labels, title):
    plt.figure(figsize=(10, 5))

    # Dividir el DataFrame en dos grupos: hombres y mujeres
    male_df = df[df["sex"] == 1]
    female_df = df[df["sex"] == 0]

    # Contar la cantidad de cada categoría para hombres y mujeres
    male_counts = [male_df[category].sum() for category in categories]
    female_counts = [female_df[category].sum() for category in categories]

    # Definir las posiciones de las barras y su ancho
    x = range(len(categories))
    bar_width = 0.35

    # Crear barras para hombres (azul)
    plt.bar(x, male_counts, width=bar_width, color="blue", label="Hombres")
    # Crear barras para mujeres (rojo)
    plt.bar(
        [i + bar_width for i in x],
        female_counts,
        width=bar_width,
        color="red",
        label="Mujeres",
    )

    # Etiquetas de categorías
    x_labels = labels

    plt.xlabel("Categorías")
    plt.ylabel("Cantidad")
    plt.xticks([i + bar_width / 2 for i in x], x_labels, rotation=0)
    plt.title(title)
    plt.legend()
    plt.grid(axis="y", linestyle="", alpha=0.7)
    plt.yticks(range(0, 71, 10))
    plt.show()


# Cargar el CSV en un DataFrame
df = pd.read_csv("processed_data.csv")

# Especificar las categorías a analizar y sus etiquetas personalizadas
categories_to_plot = ["anaemia", "diabetes", "smoking", "DEATH_EVENT"]
categories_labels = ["Anémicos", "Diabéticos", "Fumadores", "Muertos"]

# Generar gráfico de barras agrupadas por categorías y género
generate_grouped_bar_chart(
    df, categories_to_plot, categories_labels, "Histograma Agrupado por Sexo"
)
