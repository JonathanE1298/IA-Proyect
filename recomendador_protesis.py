import pandas as pd
import cv2 as cv
import mediapipe as mp

def load_data():
    # Cargar los datos desde los archivos CSV
    data_procesos = pd.read_csv('/mnt/data/Proceso de Donacion-DOUGLAS.csv')
    data_disenos = pd.read_csv('/mnt/data/Disenos de Protesis-Dispositivos (ACTIVOS).csv')
    return data_procesos, data_disenos

def recomendar_protesis(lado_brazo, tipo_amputacion):
    """
    Recomienda una protesis basada en el lado del brazo y el tipo de amputacion.

    :param lado_brazo: 'Izquierda' o 'Derecha'
    :param tipo_amputacion: Ejemplo: 'Brazo (Abajo del Codo)', 'Mano'
    :return: Lista de protesis recomendadas.
    """
    # Cargar datos
    _, data_disenos = load_data()

    # Filtrar dispositivos activos
    dispositivos_activos = data_disenos[data_disenos['ESTADO'] == 'ACTIVA']

    # Realizar recomendaciones con base en la categ
    recomendaciones = dispositivos_activos[dispositivos_activos['Categoria'].str.contains(tipo_amputacion, case=False, na=False)]

    if recomendaciones.empty:
        return f"No se encontraron protesis disponibles para el tipo de amputacion: {tipo_amputacion}."

    # Filtrar prótesis adicionales segun lado del brazo (si aplica en notas o caracteristicas)
    recomendaciones = recomendaciones[recomendaciones['Notes'].str.contains(lado_brazo, case=False, na=False)]

    if recomendaciones.empty:
        return f"No se encontraron protesis disponibles para el lado del brazo: {lado_brazo}."

    return recomendaciones[['Nombre de Dispositivo', 'Categoria', 'Notes']].to_dict(orient='records')

# Configurar Mediapipe Pose
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
            muneca_derecha = resultados.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        except IndexError:
            hombro_derecho = codo_derecho = muneca_derecha = None

        # Definir coordenadas en píxeles (si los puntos existen)
        puntos_derecho = {
            "hombro": (int(hombro_derecho.x * ancho), int(hombro_derecho.y * altura)) if hombro_derecho else None,
            "codo": (int(codo_derecho.x * ancho), int(codo_derecho.y * altura)) if codo_derecho else None,
            "muneca": (int(muneca_derecha.x * ancho), int(muneca_derecha.y * altura)) if muneca_derecha else None,
        }

        # Detectar amputaciones
        if puntos_derecho["hombro"] and not puntos_derecho["codo"]:
            tipo_amputacion = "Brazo (Arriba del Codo)"
            lado_brazo = "Derecha"
        elif puntos_derecho["codo"] and not puntos_derecho["muneca"]:
            tipo_amputacion = "Brazo (Abajo del Codo)"
            lado_brazo = "Derecha"
        elif puntos_derecho["hombro"] and puntos_derecho["codo"] and not puntos_derecho["muneca"]:
            tipo_amputacion = "Mano"
            lado_brazo = "Derecha"
        else:
            tipo_amputacion = None

        # Detectar brazo izquierdo
        try:
            hombro_izquierdo = resultados.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            codo_izquierdo = resultados.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
            muneca_izquierda = resultados.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        except IndexError:
            hombro_izquierdo = codo_izquierdo = muneca_izquierda = None

        puntos_izquierdo = {
            "hombro": (int(hombro_izquierdo.x * ancho), int(hombro_izquierdo.y * altura)) if hombro_izquierdo else None,
            "codo": (int(codo_izquierdo.x * ancho), int(codo_izquierdo.y * altura)) if codo_izquierdo else None,
            "muñeca": (int(muneca_izquierda.x * ancho), int(muneca_izquierda.y * altura)) if muneca_izquierda else None,
        }

        if puntos_izquierdo["hombro"] and not puntos_izquierdo["codo"]:
            tipo_amputacion = "Brazo (Arriba del Codo)"
            lado_brazo = "Izquierda"
        elif puntos_izquierdo["codo"] and not puntos_izquierdo["muneca"]:
            tipo_amputacion = "Brazo (Abajo del Codo)"
            lado_brazo = "Izquierda"
        elif puntos_izquierdo["hombro"] and puntos_izquierdo["codo"] and not puntos_izquierdo["muneca"]:
            tipo_amputacion = "Mano"
            lado_brazo = "Izquierda"

        # Si detecta una amputación, invocar el recomendador
        if tipo_amputacion:
            recomendaciones = recomendar_protesis(lado_brazo, tipo_amputacion)
            texto = f"Recomendado: {recomendaciones}"
        else:
            texto = "Brazo completo detectado"

        # Mostrar texto en pantalla
        cv.putText(captura, texto, (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

        # Dibujar los puntos clave detectados
        for nombre, punto in {**puntos_derecho, **puntos_izquierdo}.items():
            if punto:
                cv.circle(captura, punto, 5, (0, 255, 0), -1)

    # Mostrar la captura en tiempo real
    cv.imshow("Reconocimiento de amputaciones", captura)
    if cv.waitKey(1) == ord("s"):
        break

camara.release()
cv.destroyAllWindows()
