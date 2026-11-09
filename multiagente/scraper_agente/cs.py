import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from datetime import datetime

import requests

url = "https://gee.bccr.fi.cr/indicadoreseconomicos/Cuadros/frmVerCatCuadro.aspx?idioma=1&CodCuadro=2786"
response = requests.get(url)

if response.status_code == 200:
    html_content = response.text
else:
    print(f"Error al obtener la página: {response.status_code}")

from bs4 import BeautifulSoup
import pandas as pd

soup = BeautifulSoup(html_content, 'html.parser')
tables = soup.find_all('table')

# Suponiendo que la tabla deseada es la primera en la página
table = tables[0]

# Leer la tabla en un DataFrame de pandas
df = pd.read_html(str(table))[0]
print(df.tail(50))

