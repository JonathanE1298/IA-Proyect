import cv2 as cv
import mediapipe as mp

# Inicializar Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Captura desde la cámara
camara = cv.VideoCapture(0)

while True:
    _, captura = camara.read()
    captura_rgb = cv.cvtColor(captura, cv.COLOR_BGR2RGB)
    resultados = pose.process(captura_rgb)

    altura, ancho, _ = captura.shape

    if resultados.pose_landmarks:
        # Obtener puntos clave del brazo derecho
        try:
            hombro_derecho = resultados.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            codo_derecho = resultados.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
            muñeca_derecha = resultados.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        except IndexError:
            hombro_derecho = codo_derecho = muñeca_derecha = None

        # Definir coordenadas en píxeles (si los puntos existen)
        puntos_derecho = {
            "hombro": (int(hombro_derecho.x * ancho), int(hombro_derecho.y * altura)) if hombro_derecho else None,
            "codo": (int(codo_derecho.x * ancho), int(codo_derecho.y * altura)) if codo_derecho else None,
            "muñeca": (int(muñeca_derecha.x * ancho), int(muñeca_derecha.y * altura)) if muñeca_derecha else None,
        }

        # Detectar amputaciones
        if puntos_derecho["hombro"] and not puntos_derecho["codo"]:
            texto = "Amputación por encima del codo"
        elif puntos_derecho["codo"] and not puntos_derecho["muñeca"]:
            texto = "Amputación por debajo del codo"
        elif puntos_derecho["hombro"] and puntos_derecho["codo"] and not puntos_derecho["muñeca"]:
            texto = "Amputación en la muñeca"
        else:
            texto = "Brazo completo detectado"

        # Mostrar texto en pantalla
        cv.putText(captura, texto, (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

        # Dibujar los puntos clave detectados
        for nombre, punto in puntos_derecho.items():
            if punto:
                cv.circle(captura, punto, 5, (0, 255, 0), -1)

    # Mostrar la captura en tiempo real
    cv.imshow("Reconocimiento de amputaciones", captura)
    if cv.waitKey(1) == ord("s"):
        break

camara.release()
cv.destroyAllWindows()
