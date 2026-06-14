import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Configurar estilo
sns.set_theme(style="whitegrid")

print("Cargando datos...")
df = pd.read_excel('Clientes.xlsx')

# Realizamos el PCA con las variables clave
columnas_relevantes = ['ingreso_mensual', 'cant_vuelos', 'gasto_acumulado']
df_numeric = df[columnas_relevantes].fillna(df[columnas_relevantes].mean())

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_numeric)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_scaled)

# Añadimos los componentes al dataframe original para graficar
df['PC1'] = pca_result[:, 0]
df['PC2'] = pca_result[:, 1]

# Variables categóricas que vamos a graficar
vars_categoricas = ['sexo', 'provincia', 'ocupacion', 'clase_preferida', 'programaMillas', 'canal_compra']

# Limites para el zoom (excluyendo el 0.1% de outliers de cada lado)
x_min, x_max = np.percentile(df['PC1'], [0.1, 99.9])
y_min, y_max = np.percentile(df['PC2'], [0.1, 99.9])

for col in vars_categoricas:
    if col in df.columns:
        print(f"Generando gráfico para la categoría: {col}...")
        plt.figure(figsize=(10, 8))
        
        # Como hay ~50k puntos, para que se vean los colores usamos un marker más chico y un poco de alpha
        # Ordenamos aleatoriamente el df para que un color no tape completamente al otro (sobre todo si hay clases mayoritarias)
        df_shuffled = df.sample(frac=1, random_state=42)
        
        sns.scatterplot(
            data=df_shuffled, 
            x='PC1', 
            y='PC2', 
            hue=col, 
            alpha=0.5, 
            s=12,
            palette='tab10' # Paleta con buenos contrastes
        )
        
        plt.xlabel(f'PC1 (Varianza: {pca.explained_variance_ratio_[0]*100:.1f}%)')
        plt.ylabel(f'PC2 (Varianza: {pca.explained_variance_ratio_[1]*100:.1f}%)')
        plt.title(f'ACP Dispersión según {col.upper()}')
        
        plt.xlim(x_min - 0.5, x_max + 0.5)
        plt.ylim(y_min - 0.5, y_max + 0.5)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        
        # Ajustamos la leyenda para que no tape el gráfico
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title=col.upper())
        
        filename = f'acp_dispersion_{col}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

print("Todos los gráficos generados.")
