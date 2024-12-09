# IA-Proyect
Este proyecto utiliza las bibliotecas OpenCV y Mediapipe para detectar y clasificar amputaciones en imágenes o en tiempo real mediante cámara. Identifica tres tipos de amputaciones:
Por encima del codo.
Por debajo del codo.
A nivel de la mano.
El sistema también visualiza puntos clave y líneas para ilustrar las detecciones en las imágenes procesadas. 

Requisitos del Sistema
Python: Python 3.9 o superior.
Bibliotecas: opencv-python / mediapipe 

Cámara (opcional): Requerida para el análisis en tiempo real.

Entorno Virtual (opcional): Recomendado para gestionar dependencias.


Instalacion: 
1. Clona el repositorio: git clone https://github.com/usuario/proyecto-clasificacion-amputaciones.git
2. configura el entorno virtual: python -m venv venv
3. activa el entorno virtual: venv\Scripts\activate
4. instala las dependencias necesarias: pip install opencv-python mediapipe
5. ejecuta el programa principal: python principal.py

6. Analizar Imagen:
Ingresa la ruta de una imagen local.
Ejemplo:
C:/Users/usuario/imagen.jpg
El programa procesará la imagen y mostrará el resultado en una ventana emergente.

7. Analizar Cámara en Tiempo Real:
Analiza los puntos clave del cuerpo y clasifica amputaciones en tiempo real.

Presiona S para detener el análisis.

Notas Adicionales
* Umbral de Visibilidad:
* El programa usa un umbral predeterminado de 0.6 para determinar si un punto clave es visible. Puedes modificar este valor en la variable umbral_visibilidad dentro del código.

* Errores Comunes:

* Ruta de Imagen Inválida:
* Verifica que la ruta sea correcta y que el archivo exista.
* Cámara no Disponible:
* Asegúrate de que la cámara esté conectada y no esté siendo utilizada por otra aplicación.
* Resultados Visuales:
* Los puntos clave y las clasificaciones se dibujan directamente sobre las imágenes o video en tiempo real.
* En la carpeta modelos se encuentra los modelos entranados para poder hacer predicciones de tipo de protesis y en el archivo prediccionProtesis.py esta la manera de usarlo