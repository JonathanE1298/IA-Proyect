from flask import Flask, request, render_template
import os
import cv2
import mediapipe as mp

app = Flask(__name__)
UPLOAD_FOLDER = './static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Inicializar Mediapipe Pose y Hands
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Umbral de visibilidad
umbral_visibilidad = 0.6

# Función para clasificar amputaciones
def clasificar_amputacion(hombro, codo, munieca, mano, lado):
    if hombro.visibility > umbral_visibilidad and codo.visibility < umbral_visibilidad:
        return f"Amputación por encima del codo ({lado})", "Arriba del codo"
    elif codo.visibility > umbral_visibilidad and munieca.visibility < umbral_visibilidad:
        return f"Amputación por debajo del codo ({lado})", "Abajo del codo"
    elif munieca.visibility > umbral_visibilidad and mano is None:
        return f"Amputación en la mano ({lado})", "Mano"
    else:
        return f"Brazo completo detectado ({lado})", "Brazo completo"

# Procesar la imagen con Mediapipe
def procesar_imagen_mediapipe(ruta_imagen):
    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        return None, None, None, None

    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    resultados_pose = pose.process(imagen_rgb)
    resultados_hands = hands.process(imagen_rgb)

    altura, ancho, _ = imagen.shape

    lado_detectado = None
    amputacion_detectada = None

    if resultados_pose.pose_landmarks:
        hombro_izquierdo = resultados_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        codo_izquierdo = resultados_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
        munieca_izquierda = resultados_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]

        hombro_derecho = resultados_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        codo_derecho = resultados_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        munieca_derecha = resultados_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

        mano_izquierda = mano_derecha = None
        if resultados_hands.multi_hand_landmarks:
            for hand_landmarks in resultados_hands.multi_hand_landmarks:
                cx = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * ancho
                if cx < ancho / 2:
                    mano_izquierda = hand_landmarks
                else:
                    mano_derecha = hand_landmarks

        # Clasificar amputación
        lado_detectado, amputacion_detectada = None, None
        texto_izquierdo, nivel_izquierdo = clasificar_amputacion(hombro_izquierdo, codo_izquierdo, munieca_izquierda, mano_izquierda, "Izquierda")
        texto_derecho, nivel_derecho = clasificar_amputacion(hombro_derecho, codo_derecho, munieca_derecha, mano_derecha, "Derecha")

        if nivel_izquierdo != "Brazo completo":
            lado_detectado, amputacion_detectada = "Izquierda", nivel_izquierdo
        elif nivel_derecho != "Brazo completo":
            lado_detectado, amputacion_detectada = "Derecha", nivel_derecho

    ruta_procesada = os.path.join(app.config['UPLOAD_FOLDER'], 'procesada.png')
    cv2.imwrite(ruta_procesada, imagen)

    return lado_detectado, amputacion_detectada, "Procesamiento exitoso.", ruta_procesada

# Ruta para manejar el formulario
@app.route('/upload', methods=['POST'])
def upload_file():
    nombre = request.form.get('nombre')
    if 'imagen' not in request.files or not nombre:
        return "Por favor, completa todos los campos y sube una imagen.", 400

    archivo = request.files['imagen']
    if archivo.filename == '':
        return "No se seleccionó ninguna imagen.", 400

    if archivo:
        ruta_archivo = os.path.join(app.config['UPLOAD_FOLDER'], archivo.filename)
        archivo.save(ruta_archivo)

        lado_detectado, amputacion_detectada, mensaje, ruta_procesada = procesar_imagen_mediapipe(ruta_archivo)

        return render_template(
            './index.html',
            mensaje=mensaje,
            lado_detectado=lado_detectado,
            amputacion_detectada=amputacion_detectada,
            ruta_procesada=ruta_procesada
        )

# Ejecutar Flask
if __name__ == '__main__':
    app.run(debug=True)
