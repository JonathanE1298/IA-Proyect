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
            "muñeca": (int(muñeca_derecha.x * ancho), int(muñeca_derecha.y * altura)) if muñeca_derecha else None,
        }

        # Definir coordenadas en píxeles para el brazo izquierdo
        puntos_izquierdo = {
            "hombro": (int(hombro_izquierdo.x * ancho), int(hombro_izquierdo.y * altura)) if hombro_izquierdo else None,
            "codo": (int(codo_izquierdo.x * ancho), int(codo_izquierdo.y * altura)) if codo_izquierdo else None,
            "muñeca": (int(muñeca_izquierda.x * ancho), int(muñeca_izquierda.y * altura)) if muñeca_izquierda else None,
        }

        # Detectar amputaciones para el brazo derecho
        if puntos_derecho["hombro"] and not puntos_derecho["codo"]:
            texto_derecho = "Amputación por encima del codo (derecho)"
        elif puntos_derecho["codo"] and not puntos_derecho["muñeca"]:
            texto_derecho = "Amputación por debajo del codo (derecho)"
        elif puntos_derecho["hombro"] and puntos_derecho["codo"] and not puntos_derecho["muñeca"]:
            texto_derecho = "Amputación en la muñeca (derecho)"
        else:
            texto_derecho = "Brazo derecho completo detectado"

        # Detectar amputaciones para el brazo izquierdo
        if puntos_izquierdo["hombro"] and not puntos_izquierdo["codo"]:
            texto_izquierdo = "Amputación por encima del codo (izquierdo)"
        elif puntos_izquierdo["codo"] and not puntos_izquierdo["muñeca"]:
            texto_izquierdo = "Amputación por debajo del codo (izquierdo)"
        elif puntos_izquierdo["hombro"] and puntos_izquierdo["codo"] and not puntos_izquierdo["muñeca"]:
            texto_izquierdo = "Amputación en la muñeca (izquierdo)"
        else:
            texto_izquierdo = "Brazo izquierdo completo detectado"

        # Mostrar texto en pantalla para ambos brazos
        cv.putText(captura, texto_derecho, (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv.LINE_AA)
        cv.putText(captura, texto_izquierdo, (10, 80), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv.LINE_AA)

        # Dibujar las líneas y los puntos clave para el brazo derecho
        if puntos_derecho["hombro"] and puntos_derecho["codo"]:
            cv.line(captura, puntos_derecho["hombro"], puntos_derecho["codo"], (0, 255, 255), 2)
        if puntos_derecho["codo"] and puntos_derecho["muñeca"]:
            cv.line(captura, puntos_derecho["codo"], puntos_derecho["muñeca"], (0, 255, 255), 2)

        for nombre, punto in puntos_derecho.items():
            if punto:
                cv.circle(captura, punto, 5, (0, 255, 0), -1)

        # Dibujar las líneas y los puntos clave para el brazo izquierdo
        if puntos_izquierdo["hombro"] and puntos_izquierdo["codo"]:
            cv.line(captura, puntos_izquierdo["hombro"], puntos_izquierdo["codo"], (255, 0, 255), 2)
        if puntos_izquierdo["codo"] and puntos_izquierdo["muñeca"]:
            cv.line(captura, puntos_izquierdo["codo"], puntos_izquierdo["muñeca"], (255, 0, 255), 2)

        for nombre, punto in puntos_izquierdo.items():
            if punto:
                cv.circle(captura, punto, 5, (255, 0, 0), -1)

    # Mostrar la captura en tiempo real
    cv.imshow("Reconocimiento de amputaciones", captura)
    if cv.waitKey(1) == ord("s"):
        break

camara.release()
cv.destroyAllWindows()
