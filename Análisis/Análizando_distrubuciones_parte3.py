import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px

# Cargar el DataFrame con los datos
df = pd.read_csv("processed_data.csv")

# Eliminar columnas no numéricas o categóricas
df = df.select_dtypes(include=[np.number])

# Eliminar la columna "DEATH_EVENT"
X = df.drop(columns=["DEATH_EVENT"]).values

# Crear un array unidimensional para la columna objetivo (DEATH_EVENT)
y = df["DEATH_EVENT"].values

# Aplicar t-SNE para reducir la dimensionalidad a 3 componentes
X_embedded = TSNE(
    n_components=3, learning_rate="auto", init="random", perplexity=3
).fit_transform(X)

# Crear un DataFrame con los datos reducidos
df_embedded = pd.DataFrame(
    data=X_embedded, columns=["Dimension 1", "Dimension 2", "Dimension 3"]
)

# Agregar la columna objetivo al DataFrame reducido
df_embedded["DEATH_EVENT"] = y

# Crear un gráfico de dispersión 3D con Plotly
fig = px.scatter_3d(
    df_embedded, x="Dimension 1", y="Dimension 2", z="Dimension 3", color="DEATH_EVENT"
)

# Configurar etiquetas y título
fig.update_layout(
    scene=dict(
        xaxis_title="Dimension 1", yaxis_title="Dimension 2", zaxis_title="Dimension 3"
    ),
    title="Gráfico de dispersión 3D de datos t-SNE con colores de clase (Vivo/Muerto)",
)

# Mostrar el gráfico
fig.show()
