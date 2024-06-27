# Detector De Gritos

Este proyecto implementa un sistema automatizado de reconocimiento de gritos en un laboratorio de rocas y materiales. Utiliza técnicas avanzadas de procesamiento de audio y modelos de aprendizaje automático para detectar y responder rápidamente a emergencias, mejorando la seguridad y eficiencia del entorno de trabajo.

## Tabla de Contenidos

- [Descripción](#descripción)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Uso](#uso)
- [Entrenamiento del Modelo](#entrenamiento-del-modelo)
- [Evaluación del Modelo](#evaluación-del-modelo)
- [Autores](#autores)

## Descripción

El sistema de reconocimiento de gritos está diseñado para capturar sonidos en tiempo real, preprocesarlos y analizar sus características para detectar gritos. Utiliza Redes Neuronales Convolucionales (CNN) y ResNet-34 para clasificar los sonidos y generar alertas en caso de emergencias.

## Estructura del Proyecto

- `dataset/`: Contiene los datos de audio utilizados para el entrenamiento y evaluación del modelo.
- `images/`: Almacena imágenes de espectrogramas y otros gráficos relevantes.
- `Models/`: Contiene los modelos entrenados de aprendizaje automático.
- `SoundRecord/`: Scripts y módulos responsables de la captura y preprocesamiento de audio.
- `templates/`: Plantillas HTML para la interfaz de usuario web.
- `test/`: Scripts de prueba para evaluar la funcionalidad del sistema.
- `uploads/`: Directorio para almacenar archivos subidos por los usuarios.
- `App.py`: Punto de entrada principal de la aplicación.
- `GritosModelo.ipynb`: Notebook Jupyter utilizado para el desarrollo y experimentación del modelo.
- `ModelEval.py`: Script para evaluar la precisión y eficacia del modelo entrenado.
- `RecordSound.py`: Script para grabar sonidos en el laboratorio.
- `requirements.txt`: Lista de dependencias necesarias para ejecutar el proyecto.
- `voice_image.png`: Imagen utilizada en la documentación o interfaz de usuario.

## Requisitos

- Python 3.7+
- Librerías especificadas en `requirements.txt`

## Instalación

1. Clonar el repositorio:
    ```bash
    git clone https://github.com/usuario/DetectorDeGritos.git
    cd DetectorDeGritos
    ```

2. Crear y activar un entorno virtual (opcional pero recomendado):
    ```bash
    python -m venv env
    source env/bin/activate  # En Windows: env\Scripts\activate
    ```

3. Instalar las dependencias:
    ```bash
    pip install -r requirements.txt
    ```

## Uso

1. Iniciar la aplicación:
    ```bash
    python App.py
    ```

2. La aplicación web se ejecutará en `http://localhost:5000`. Desde aquí, se puede subir archivos de audio y visualizar los resultados del análisis.

## Entrenamiento del Modelo

El entrenamiento del modelo se realiza utilizando el notebook Jupyter `GritosModelo.ipynb`. Este notebook contiene el código para la carga de datos, preprocesamiento, entrenamiento y evaluación del modelo.

## Evaluación del Modelo

Para evaluar el modelo entrenado, se puede ejecutar el script `ModelEval.py`:
```bash
python ModelEval.py
```
## Autores

Manuel Isaác Armijos - Correo: manuel.i.armijos@unl.edu.ec
Carlos Enrique Armijos - Correo: carlos.e.armijos.l@unl.edu.ec 
