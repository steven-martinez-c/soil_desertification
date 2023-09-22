# Proyecto Soil Desertification

Este proyecto tiene como objetivo abordar el problema de la desertificación del suelo y proporcionar herramientas para su análisis y mitigación.

## Estructura del proyecto

El proyecto se organiza de la siguiente manera:

- **data**: Directorio que contiene los datos utilizados en el proyecto.
  - **images**: Directorio que almacena las imágenes relacionadas con el análisis de la desertificación del suelo.
    - **processed**: Directorio que contiene las imágenes procesadas utilizadas en el análisis.
    - **raw**: Directorio que contiene las imágenes originales sin procesar obtenidas de fuentes externas.
  - **shapes**: Directorio que almacena los archivos de formas (shapefiles) utilizados en el proyecto para representar regiones geográficas.

- **models**: Directorio que contiene los modelos entrenados o utilizados en el proyecto para predecir o clasificar datos relacionados con la desertificación del suelo.

- **notebooks**: Directorio que contiene los notebooks Jupyter utilizados para el análisis exploratorio de datos, la visualización y el desarrollo de modelos.

- **reports**: Directorio que contiene los informes o reportes generados a partir del análisis de datos y los resultados obtenidos en el proyecto.

- **src**: Directorio que contiene el código fuente del proyecto.
  - **utils**: Directorio que almacena los archivos de utilidades y funciones auxiliares utilizados en el proyecto.

## Librerías requeridas

El proyecto utiliza las siguientes librerías de Python:

- numpy: Biblioteca para cálculos numéricos en Python.
- pandas: Biblioteca para manipulación y análisis de datos.
- matplotlib: Biblioteca para la visualización de datos en gráficos.
- scikit-learn: Biblioteca para el aprendizaje automático y la minería de datos.
- tensorflow: Biblioteca para la creación y entrenamiento de modelos de aprendizaje automático.
- gdal: Biblioteca para el procesamiento de datos geoespaciales.
- geopandas: Biblioteca para el análisis y manejo de datos geoespaciales.

Puedes instalar estas librerías utilizando el gestor de paquetes pip. Por ejemplo:

```
conda install tensorflow
```

Recuerda crear y activar un entorno virtual para el proyecto antes de instalar las librerías.

```
conda env create -f environment.yml
```


