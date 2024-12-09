import cv2
import mediapipe as mp

# Inicializar Mediapipe Pose y Hands
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Umbral de visibilidad
umbral_visibilidad = 0.6

# Funcion para clasificar amputaciones
def clasificar_amputacion(hombro, codo, munieca, mano, lado):
    if hombro.visibility > umbral_visibilidad and codo.visibility < umbral_visibilidad:
        return f"Amputacion por encima del codo ({lado})"
    elif codo.visibility > umbral_visibilidad and munieca.visibility < umbral_visibilidad:
        return f"Amputacion por debajo del codo ({lado})"
    elif munieca.visibility > umbral_visibilidad and mano is None:
        return f"Amputacion en la mano ({lado})"
    else:
        return f"Brazo completo detectado ({lado})"

# Procesar una imagen con Mediapipe Pose y Hands
def procesar_imagen_mediapipe(imagen):
    try:
        # Convertir a RGB
        imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

        # Pose: Deteccion de puntos clave del cuerpo
        resultados_pose = pose.process(imagen_rgb)

        # Hands: Deteccion de puntos clave de las manos
        resultados_hands = hands.process(imagen_rgb)

        altura, ancho, _ = imagen.shape

        if resultados_pose.pose_landmarks:
            # Puntos del brazo izquierdo
            hombro_izquierdo = resultados_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            codo_izquierdo = resultados_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
            munieca_izquierda = resultados_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]

            # Puntos del brazo derecho
            hombro_derecho = resultados_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            codo_derecho = resultados_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
            munieca_derecha = resultados_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

            # Verificar presencia de manos
            mano_izquierda = None
            mano_derecha = None

            if resultados_hands.multi_hand_landmarks:
                for hand_landmarks in resultados_hands.multi_hand_landmarks:
                    # Determinar si la mano es izquierda o derecha
                    cx = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * ancho
                    if cx < ancho / 2:
                        mano_izquierda = hand_landmarks
                    else:
                        mano_derecha = hand_landmarks

            # Clasificar amputaciones
            texto_izquierdo = clasificar_amputacion(hombro_izquierdo, codo_izquierdo, munieca_izquierda, mano_izquierda, "Izquierda")
            texto_derecho = clasificar_amputacion(hombro_derecho, codo_derecho, munieca_derecha, mano_derecha, "Derecha")

            # Dibujar puntos clave y lineas
            def dibujar_puntos_y_lineas(hombro, codo, munieca, color_punto, color_linea):
                puntos = {
                    "hombro": (int(hombro.x * ancho), int(hombro.y * altura)) if hombro.visibility > umbral_visibilidad else None,
                    "codo": (int(codo.x * ancho), int(codo.y * altura)) if codo.visibility > umbral_visibilidad else None,
                    "munieca": (int(munieca.x * ancho), int(munieca.y * altura)) if munieca.visibility > umbral_visibilidad else None,
                }

                if puntos["hombro"] and puntos["codo"]:
                    cv2.line(imagen, puntos["hombro"], puntos["codo"], color_linea, 2)
                if puntos["codo"] and puntos["munieca"]:
                    cv2.line(imagen, puntos["codo"], puntos["munieca"], color_linea, 2)
                for punto in puntos.values():
                    if punto:
                        cv2.circle(imagen, punto, 5, color_punto, -1)

            # Dibujar brazo izquierdo
            dibujar_puntos_y_lineas(hombro_izquierdo, codo_izquierdo, munieca_izquierda, (255, 0, 255), (255, 0, 0))

            # Dibujar brazo derecho
            dibujar_puntos_y_lineas(hombro_derecho, codo_derecho, munieca_derecha, (0, 255, 0), (0, 255, 255))

            # Dibujar puntos de las manos
            if mano_izquierda:
                for lm in mano_izquierda.landmark:
                    x, y = int(lm.x * ancho), int(lm.y * altura)
                    cv2.circle(imagen, (x, y), 3, (255, 255, 0), -1)
            if mano_derecha:
                for lm in mano_derecha.landmark:
                    x, y = int(lm.x * ancho), int(lm.y * altura)
                    cv2.circle(imagen, (x, y), 3, (0, 255, 255), -1)

            # Mostrar texto en la imagen
            cv2.putText(imagen, texto_izquierdo, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(imagen, texto_derecho, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    except Exception as e:
        print(f"Error procesando imagen con Mediapipe: {e}")

    return imagen

# Procesar imagen o cámara según la opcion
def main():
    print("Selecciona una opcion:")
    print("1. Analizar imagen")
    print("2. Analizar cámara en tiempo real")
    opcion = input("Introduce el número de tu opcion: ")

    if opcion == "1":
        ruta_imagen = input("Introduce la ruta de la imagen: ")
        imagen = cv2.imread(ruta_imagen)
        if imagen is None:
            print("No se pudo cargar la imagen. Verifica la ruta.")
            return
        imagen_procesada = procesar_imagen_mediapipe(imagen)
        cv2.imshow("Deteccion y Clasificacion de Amputaciones", imagen_procesada)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif opcion == "2":
        camara = cv2.VideoCapture(0)
        try:
            while True:
                ret, frame = camara.read()
                if not ret:
                    print("Error al capturar fotograma.")
                    break

                frame = procesar_imagen_mediapipe(frame)
                cv2.imshow("Deteccion y Clasificacion de Amputaciones", frame)

                if cv2.waitKey(1) & 0xFF == ord("s"):
                    break
        except Exception as e:
            print(f"Error inesperado: {e}")
        finally:
            camara.release()
            cv2.destroyAllWindows()
    else:
        print("Opcion inválida. Por favor, selecciona 1 o 2.")

# Ejecutar el programa
if __name__ == "__main__":
    main()
