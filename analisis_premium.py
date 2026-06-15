import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analisis_premium_cualitativas():
    print("Iniciando análisis cualitativo de Clientes Premium (Ballenas)...")
    
    # 1. Cargar datos de clientes premium
    df_premium = pd.read_csv('vistas_minables/clientes_premium.csv')
    
    categoricas = ['sexo', 'provincia', 'ocupacion', 'clase_preferida', 'programaMillas', 'canal_compra']
    
    # Directorio de salida
    out_dir_premium = 'TPI/graficos/premium/comparativas_cualitativas'
    os.makedirs(out_dir_premium, exist_ok=True)
    
    # 2. Generar countplots para cada variable cualitativa
    for col in categoricas:
        plt.figure(figsize=(10, 6))
        
        # Countplot ordenado por frecuencia
        sns.countplot(data=df_premium, y=col, hue=col, palette='viridis', 
                      order=df_premium[col].value_counts().index, legend=False)
        
        plt.title(f'Distribución de {col} en Clientes Premium (VIPs)')
        plt.xlabel('Cantidad de Clientes Premium')
        plt.ylabel(col)
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir_premium, f'premium_cualitativa_{col}.png'))
        plt.close()

    print(f"Gráficos de clientes premium guardados exitosamente en {out_dir_premium}")

if __name__ == '__main__':
    analisis_premium_cualitativas()
