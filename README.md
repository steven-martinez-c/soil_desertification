# Proyecto de detección de la desertificación del suelo

La desertificación es un problema global que afecta a millones de personas y al medio ambiente. Este proyecto tiene como objetivo abordar este problema proporcionando herramientas para la detección de la desertificación del suelo.


## Estructura del proyecto

El proyecto se organiza de la siguiente manera:

- **Data**: Directorio que contiene los datos utilizados en el proyecto.
  - **Images**: Directorio que almacena las imágenes relacionadas con el análisis de la desertificación del suelo.
    - **Processed**: Directorio que contiene las imágenes procesadas utilizadas en el análisis.
    - **Raw**: Directorio que contiene las imágenes originales sin procesar obtenidas de fuentes externas.
  - **Shapes**: Directorio que almacena los archivos de formas (shapefiles) utilizados  en el proyecto para representar regiones geográficas.
  - **Models**: Directorio que contiene los modelos entrenados o utilizados en el proyecto para predecir o clasificar datos relacionados con la desertificación del suelo.
  - **Notebooks**: Directorio que contiene los notebooks Jupyter utilizados para el análisis exploratorio de datos, la visualización y el desarrollo de modelos.
  - **Reports**: Directorio que contiene los informes o reportes generados a partir del análisis de datos y los resultados obtenidos en el proyecto.
  - **Src**: Directorio que contiene el código fuente del proyecto.
    - **Layers**: Directorio almacena funciones útiles durante el proceso de exploración.
      - **Dict_class**: Diccionario de clases landsat y sentinel.
      - **Utilities**: Funciones de utilidad durante el proceso de exploración.
    - **Controller**: Clases que proporcionan una interfaz para el manejo, creación y generación de raster, pixeles y mosaicos.

## Instalación de los paquetes GDAL

Para instalar los paquetes GDAL necesarios, siga estos pasos en el entorno conda de Jupyter:

1. **Actualice los paquetes existentes:**

```
sudo apt-get update && sudo apt upgrade -y && sudo apt autoremove
```

2. **Instale los paquetes GDAL:**

```
sudo apt-get install -y cdo nco gdal-bin libgdal-dev
```

3. **Verifique que GDAL se encuentre correctamente instalado:**

```
gdalinfo --version
```
Este comando debe devolver la versión instalada en su sistema:

```
GDAL 3.4.1, released 2021/12/27
```

## Instalación de librerías

Para instalar las librerías necesarias para el proyecto, puedes utilizar uno de los siguientes métodos:

**Instalación individual**: Instala cada librería de forma individual utilizando el gestor de paquetes pip o conda. El proyecto utiliza las siguientes librerías de Python:


  - **numpy:** Biblioteca para cálculos numéricos en Python.
  - **pandas:** Biblioteca para manipulación y análisis de datos.
  - **matplotlib:** Biblioteca para la visualización de datos en gráficos.
  - **scikit-learn:** Biblioteca para el aprendizaje automático y la minería de datos.
  - **tensorflow:** Biblioteca para la creación y entrenamiento de modelos de aprendizaje automático.
  - **gdal:** Biblioteca para el procesamiento de datos geoespaciales.
  - **geopandas:** Biblioteca para el análisis y manejo de datos geoespaciales.
  - **tqdm:** Biblioteca para mostrar una barra de progreso.
  - **keras:** Biblioteca para la creación de modelos de aprendizaje automático profundos.
  - **pyproj:** Biblioteca para la conversión de coordenadas geográficas.
  - **rasterio:** Biblioteca para el procesamiento de datos ráster.
  - **scipy:** Biblioteca para cálculo científico.
  - **seaborn:** Biblioteca para la visualización de datos estadísticos.
  - **shapely:** Biblioteca para el manejo de datos vectoriales.
  - **nvidia-cublas-cu11** y **nvidia-cudnn-cu11:** Librerías para la aceleración de GPU con CUDA.

Por ejemplo, para instalar TensorFlow, ejecuta el siguiente comando:
```
pip install tensorflow
```

O crear un entorno de conda definido en el archivo "environment.yml", con las librerias necesarias con el siguiente comando:

```
conda env create -f environment.yml
```

## Instalar GPU para Python

Para utilizar la GPU para el procesamiento de datos en Python, es necesario instalar las siguientes librerías:

* **cudatoolkit**
* **nvidia-cudnn**
* **tensorflow**

**Verificar si las librerías están instaladas**

Para verificar si las librerías están instaladas, ejecuta los siguientes comandos en la terminal:

```
conda list | grep cudatoolkit
conda list | grep nvidia-cudnn
conda list | grep tensorflow
```

Si las librerías no están instaladas, sigue los pasos a continuación.

**Instalar las librerías**

Para instalar las librerías, ejecuta los siguientes comandos en la terminal:
```
conda install -c conda-forge cudatoolkit=11.8.0
python3 -m pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.10.*
```
**Configurar las variables de entorno**
Para que TensorFlow pueda utilizar la GPU, es necesario configurar las siguientes variables de entorno:
```
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```
Para configurar estas variables de entorno de forma permanente, puedes añadirlas al archivo ~/.bashrc.

### Verificar la instalación
Para verificar que la instalación fue exitosa, ejecuta el siguiente código en la terminal:
```
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Este código debería devolver una lista de dispositivos GPU disponibles.
```
Versión de TensorFlow: 2.10.1
GPU: ['device: 0, name: NVIDIA GeForce RTX 3060 Laptop GPU, pci bus id: 0000:01:00.0, compute capability: 8.6']
Versión Cuda: 11.2
Versión Cudnn: 8
```