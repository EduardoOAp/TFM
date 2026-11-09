
from bs4 import BeautifulSoup
import requests
import pandas as pd
import os
from io import StringIO

#Leer archivo
#df = pd.read_parquet( r"D:\pythonenvs\TFM\multiagente\data\insumos\curva_soberana.parquet")
#print(df.info())
#print(df.head(700))

# URL de origen
#url = "https://gee.bccr.fi.cr/indicadoreseconomicos/Cuadros/frmVerCatCuadro.aspx?idioma=1&CodCuadro=2786"

#import requests
import time

url = (
    "https://gee.bccr.fi.cr/"
    "indicadoreseconomicos/Cuadros/"
    "frmVerCatCuadro.aspx?idioma=1&CodCuadro=2786"
)

for intento in range(1, 4):
    try:
        print(f"Intento {intento} de 3...")

        respuesta = requests.get(
            url,
            timeout=(15, 120)
        )

        respuesta.raise_for_status()

        print("Código HTTP:", respuesta.status_code)
        print("Tamaño de la respuesta:", len(respuesta.text))
        print(respuesta.text[:300])

        break

    except requests.exceptions.ConnectionError as e:
        print("Error de conexión:", e)

    except requests.exceptions.Timeout:
        print("La petición superó el tiempo de espera.")

    except requests.exceptions.RequestException as e:
        print("Error HTTP:", e)

    if intento < 3:
        time.sleep(5)



# Solicitud HTTP y análisis del contenido HTML
pagina = requests.get(url)
contenido = pagina.text

#df_fechas = pd.read_html(contenido)[2] # 2 Fechas
df_fechas = pd.read_html(StringIO(contenido))[2]

df_fechas["fecha_inicio"] = (
    df_fechas[0]
    .str.extract(r"(\d{2}/\d{2}/\d{4})")[0]           # extrae la primera fecha
    .pipe(pd.to_datetime, dayfirst=True)             # convierte a datetime
    .dt.strftime("%Y/%m/%d")                          # formato final
)

df = pd.read_html(StringIO(contenido))[13].T
#df = pd.read_html(contenido)[13].T # A un año, pero no trae comas, lo esta poniendo como miles
df.columns = ["valor"]
# Tomar la única celda, dividir por espacios y convertir en lista
lista = df.loc[0, "valor"].split()

# Crear nuevo DataFrame vertical
df_cs1A = pd.DataFrame({"valor": lista})

df_final = pd.DataFrame({
    "tipoCurva": ["CS1A"] * len(df_cs1A),             # columna constante
    "fecha": df_fechas["fecha_inicio"].values,
    "valor": df_cs1A["valor"].values
})
df_final["valor"] = df_final["valor"].str.replace(",", ".", regex=False)
df_final["valor"] = df_final["valor"].astype(float)

#print(df_final)

csv_path = r"D:\pythonenvs\TFM\multiagente\data\Curva_soberana1A.csv"
parquet_path = r"D:\pythonenvs\TFM\multiagente\data\insumos\curva_soberana.parquet"

# Cargar CSV nuevo
df_final.to_csv(csv_path,sep=";",index=False)

df_nuevos = pd.read_csv(csv_path,sep=';',parse_dates=['fecha'])
#print(df_nuevos.info())

# Claves para evitar duplicados
columnas_clave = ["fecha"]  # ajusta según lo que defina unicidad

if os.path.exists(parquet_path):
    df_existente = pd.read_parquet(parquet_path)
    df_combinado = pd.concat([df_existente, df_nuevos], ignore_index=True)
    df_combinado = df_combinado.drop_duplicates(subset=columnas_clave)
    print(f"Parquet actualizado con {len(df_combinado)} registros.")
else:
    df_combinado = df_nuevos
    print(f"Parquet creado desde cero con {len(df_combinado)} registros.")

# Guardar (sobrescribe el archivo)
df_combinado.to_parquet(parquet_path, index=False)
#print(df_combinado.info())
#print(df_combinado.head(700))