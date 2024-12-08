import cv2 as cv
import mediapipe as mp

# Inicializar Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Funcion para procesar la camara
def procesar_camara():
    camara = cv.VideoCapture(0)
    while True:
        ret, captura = camara.read()
        if not ret:
            print("No se pudo acceder a la camara.")
            break

        procesar_imagen(captura)

        cv.imshow("Deteccion de amputaciones - Camara", captura)
        if cv.waitKey(1) & 0xFF == ord("s"):  # Presiona 's' para salir
            break

    camara.release()
    cv.destroyAllWindows()

# Funcion para procesar una imagen
def procesar_imagen(imagen):
    imagen_rgb = cv.cvtColor(imagen, cv.COLOR_BGR2RGB)
    resultados = pose.process(imagen_rgb)

    altura, ancho, _ = imagen.shape

    # Umbral de confianza
    umbral_confianza = 0.3

    if resultados.pose_landmarks:
        # Obtener puntos clave del brazo derecho
        try:
            hombro_derecho = resultados.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            codo_derecho = resultados.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
            munieca_derecha = resultados.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        except IndexError:
            hombro_derecho = codo_derecho = munieca_derecha = None

        puntos_derecho = {
            "hombro": (int(hombro_derecho.x * ancho), int(hombro_derecho.y * altura)) if hombro_derecho and hombro_derecho.visibility > umbral_confianza else None,
            "codo": (int(codo_derecho.x * ancho), int(codo_derecho.y * altura)) if codo_derecho and codo_derecho.visibility > umbral_confianza else None,
            "munieca": (int(munieca_derecha.x * ancho), int(munieca_derecha.y * altura)) if munieca_derecha and munieca_derecha.visibility > umbral_confianza else None,
        }

        # Obtener puntos clave del brazo izquierdo
        try:
            hombro_izquierdo = resultados.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            codo_izquierdo = resultados.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
            munieca_izquierda = resultados.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        except IndexError:
            hombro_izquierdo = codo_izquierdo = munieca_izquierda = None

        puntos_izquierdo = {
            "hombro": (int(hombro_izquierdo.x * ancho), int(hombro_izquierdo.y * altura)) if hombro_izquierdo and hombro_izquierdo.visibility > umbral_confianza else None,
            "codo": (int(codo_izquierdo.x * ancho), int(codo_izquierdo.y * altura)) if codo_izquierdo and codo_izquierdo.visibility > umbral_confianza else None,
            "munieca": (int(munieca_izquierda.x * ancho), int(munieca_izquierda.y * altura)) if munieca_izquierda and munieca_izquierda.visibility > umbral_confianza else None,
        }

        # Dibujar lineas y puntos
        def dibujar_puntos_y_lineas(puntos, color_linea, color_punto):
            if puntos["hombro"] and puntos["codo"]:
                cv.line(imagen, puntos["hombro"], puntos["codo"], color_linea, 2)
            if puntos["codo"] and puntos["munieca"]:
                cv.line(imagen, puntos["codo"], puntos["munieca"], color_linea, 2)
            for punto in puntos.values():
                if punto:
                    cv.circle(imagen, punto, 5, color_punto, -1)

        dibujar_puntos_y_lineas(puntos_derecho, (0, 255, 255), (0, 255, 0))  # Derecho
        dibujar_puntos_y_lineas(puntos_izquierdo, (255, 0, 255), (255, 0, 0))  # Izquierdo

        # Detectar amputaciones
        def detectar_amputacion(puntos, lado):
            if puntos["hombro"] and not puntos["codo"]:
                return f"Amputacion por encima del codo ({lado})"
            elif puntos["codo"] and not puntos["munieca"]:
                return f"Amputacion por debajo del codo ({lado})"
            elif puntos["hombro"] and puntos["codo"] and not puntos["munieca"]:
                return f"Amputacion en la munieca ({lado})"
            else:
                return f"Brazo completo detectado ({lado})"

        texto_derecho = detectar_amputacion(puntos_derecho, "derecho")
        texto_izquierdo = detectar_amputacion(puntos_izquierdo, "izquierdo")

        # Mostrar texto en la imagen
        cv.rectangle(imagen, (10, 30), (500, 100), (0, 0, 0), -1)  # Fondo negro
        cv.putText(imagen, texto_derecho, (20, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)
        cv.putText(imagen, texto_izquierdo, (20, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)

# Inicio del programa
print("Seleccione el modo de operacion:")
print("1. Analizar camara en tiempo real")
print("2. Analizar una imagen especifica")
opcion = input("Ingrese su opcion (1 o 2): ")

if opcion == "1":
    procesar_camara()
elif opcion == "2":
    ruta_imagen = input("Ingrese la ruta de la imagen: ")
    imagen = cv.imread(ruta_imagen)
    if imagen is None:
        print("No se pudo cargar la imagen. Verifique la ruta.")
    else:
        procesar_imagen(imagen)
        cv.imshow("Deteccion de amputaciones - Imagen", imagen)
        cv.waitKey(0)
        cv.destroyAllWindows()
else:
    print("Opcion no valida. Saliendo del programa.")