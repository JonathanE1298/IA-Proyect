import cv2 as cv
import mediapipe as mp
#opcoin para leer imagenes

# Inicializar Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Ruta de la imagen
ruta_imagen = "./proyecto/imagen1.png"
imagen = cv.imread(ruta_imagen)
imagen_rgb = cv.cvtColor(imagen, cv.COLOR_BGR2RGB)
resultados = pose.process(imagen_rgb)

altura, ancho, _ = imagen.shape

if resultados.pose_landmarks:
    # Obtener puntos clave del brazo derecho
    try:
        hombro_derecho = resultados.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        codo_derecho = resultados.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        muñeca_derecha = resultados.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    except IndexError:
        hombro_derecho = codo_derecho = muñeca_derecha = None

    # Obtener puntos clave del brazo izquierdo
    try:
        hombro_izquierdo = resultados.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        codo_izquierdo = resultados.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
        muñeca_izquierda = resultados.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    except IndexError:
        hombro_izquierdo = codo_izquierdo = muñeca_izquierda = None

    # Definir coordenadas en píxeles para el brazo derecho
    puntos_derecho = {
        "hombro": (int(hombro_derecho.x * ancho), int(hombro_derecho.y * altura)) if hombro_derecho else None,
        "codo": (int(codo_derecho.x * ancho), int(codo_derecho.y * altura)) if codo_derecho else None,
        "munieca": (int(muñeca_derecha.x * ancho), int(muñeca_derecha.y * altura)) if muñeca_derecha else None,
    }

    # Definir coordenadas en píxeles para el brazo izquierdo
    puntos_izquierdo = {
        "hombro": (int(hombro_izquierdo.x * ancho), int(hombro_izquierdo.y * altura)) if hombro_izquierdo else None,
        "codo": (int(codo_izquierdo.x * ancho), int(codo_izquierdo.y * altura)) if codo_izquierdo else None,
        "munieca": (int(muñeca_izquierda.x * ancho), int(muñeca_izquierda.y * altura)) if muñeca_izquierda else None,
    }

    # Dibujar las líneas y los puntos clave para el brazo derecho
    if puntos_derecho["hombro"] and puntos_derecho["codo"]:
        cv.line(imagen, puntos_derecho["hombro"], puntos_derecho["codo"], (0, 255, 255), 2)
    if puntos_derecho["codo"] and puntos_derecho["munieca"]:
        cv.line(imagen, puntos_derecho["codo"], puntos_derecho["munieca"], (0, 255, 255), 2)

    for nombre, punto in puntos_derecho.items():
        if punto:
            cv.circle(imagen, punto, 5, (0, 255, 0), -1)

    # Dibujar las líneas y los puntos clave para el brazo izquierdo
    if puntos_izquierdo["hombro"] and puntos_izquierdo["codo"]:
        cv.line(imagen, puntos_izquierdo["hombro"], puntos_izquierdo["codo"], (255, 0, 255), 2)
    if puntos_izquierdo["codo"] and puntos_izquierdo["munieca"]:
        cv.line(imagen, puntos_izquierdo["codo"], puntos_izquierdo["munieca"], (255, 0, 255), 2)

    for nombre, punto in puntos_izquierdo.items():
        if punto:
            cv.circle(imagen, punto, 5, (255, 0, 0), -1)

# Mostrar la imagen procesada
cv.imshow("Deteccion en imagen", imagen)
cv.waitKey(0)
cv.destroyAllWindows()
