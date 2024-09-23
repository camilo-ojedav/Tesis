# Tesis: Estimación de Ángulo de Movilidad utilizando un Modelo de Computer Vision

## 💻 Instalación

- Clonar el repositorio

  ```bash
  git clone https://github.com/camilo-ojedav/tesis.git
  ```

- Configurar el entorno de Python 3.10.14 con conda y activarlo. [Descarga de conda](https://docs.anaconda.com/free/miniconda/index.html)
  ```bash
  conda create --name tesis python=3.10.14
  conda activate tesis
  ```
  
- Instalación de las dependencias necesarias desde el archivo requirements.txt
  ```bash
  pip install -r requirements.txt
  ```
- Para desactivar el entorno virtual al terminar:
  ```bash
  conda deactivate
  ```

## 🎬 Modo de uso

### video.py
Este script permite la medición del angulo del brazo de una persona. Esto a traves de darle de entrada un video en donde se vea el brazo de una persona. 
Posteriormente el script detecta los puntos de hombro, codo y muñeca para posteriormente calcular el angulo que existe entre esos tres puntos. 
Se puede exportar el video con los puntos estimados del brazo si se desea. 
Por ultimo el script guarda los datos de la medición en una base de datos llamada angle_detections.

- `--input`: La ruta del video a analizar (Obligatorio).
- `-o`, `--output`: Exporta el video con los puntos estimados.
- `--show`: Muestra la inferencia en tiempo real.

Ejemplo de uso: 
```bash
python video.py --input videos/video_1.mp4 --show -o
```

## © license

Este programa integra un componente de google, que contiene la siguiente licencia:

- mediapipe: El modelo de human pose estimation utilizado en este programa, mediapipe, se distribuye bajo la [Apache License 2.0](https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE).
  Puede encontrar más detalles sobre esta licencia aquí.

