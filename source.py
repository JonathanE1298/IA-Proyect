import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Cargar los datos desde el archivo CSV con el delimitador correcto
data = pd.read_csv('datos_protesis.csv', delimiter=';', on_bad_lines='skip')

import cv2
print(cv2.__version__)

# Print the columns to check their names
print("Column names in the DataFrame:", data.columns)

# Seleccionar columnas relevantes
columns_relevant = ['Modelo de Prótesis', 'edad', 'sexousuario', 
                    'amputacionusuario', 'ladodeprotesisusuario']

# Check if all relevant columns exist in the DataFrame
missing_columns = [col for col in columns_relevant if col not in data.columns]
if missing_columns:
    print(f"Missing columns: {missing_columns}")
else:
    data = data[columns_relevant]

    # Eliminar filas con valores faltantes
    data = data.dropna()

    # Verificar los datos
    print(data.head())

    # Codificar columnas categóricas
    categorical_columns = ['sexousuario', 'amputacionusuario', 'ladodeprotesisusuario']
    label_encoder = LabelEncoder()

    # Codificar las etiquetas (Modelo de Prótesis)
    data['Modelo de Prótesis'] = label_encoder.fit_transform(data['Modelo de Prótesis'])

    # Crear transformador para las columnas categóricas
    column_transformer = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_columns)
        ],
        remainder='passthrough'
    )

    # Separar características (X) y etiquetas (y)
    X = data.drop(columns=['Modelo de Prótesis'])
    y = data['Modelo de Prótesis']

    # Transformar los datos
    X = column_transformer.fit_transform(X)

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar el modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluar el modelo
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Precisión del modelo: {accuracy * 100:.2f}%')

    import cv2

def capturar_imagen(nombre_archivo="foto_usuario.jpg"):
    # Abrir la cámara
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Captura de Imagen")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Error al acceder a la cámara.")
            break
        cv2.imshow("Captura de Imagen", frame)

        # Presiona 'Espacio' para capturar o 'Esc' para salir
        k = cv2.waitKey(1)
        if k % 256 == 32:  # Espacio
            cv2.imwrite(nombre_archivo, frame)
            print(f"Imagen guardada como {nombre_archivo}")
            break
        elif k % 256 == 27:  # Esc
            print("Cerrando la cámara.")
            break

    cam.release()
    cv2.destroyAllWindows()

import mediapipe as mp
import cv2

def procesar_imagen(nombre_archivo="foto_usuario.jpg"):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # Leer la imagen
    image = cv2.imread(nombre_archivo)
    if image is None:
        print("No se pudo cargar la imagen.")
        return None

    # Convertir BGR a RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Procesar la imagen para detectar manos
    result = hands.process(image_rgb)

    if result.multi_hand_landmarks:
        print("Manos detectadas:")
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        return "Mano detectada"
    else:
        print("No se detectaron manos. Posible amputación completa.")
        return "Amputación completa"

# Procesar la imagen capturada
nivel_amputacion = procesar_imagen("foto_usuario.jpg")
print(f"Nivel de amputación detectado: {nivel_amputacion}")

# Datos del usuario a predecir
nuevo_usuario = {
    'edad': 30,
    'sexousuario': 'Hombre',
    'amputacionusuario': nivel_amputacion,  # Resultado del análisis de imagen
    'ladodeprotesisusuario': 'Izquierda'
}

# Convertir a DataFrame
nuevo_usuario_df = pd.DataFrame([nuevo_usuario])

# Transformar los datos
nuevo_usuario_transformed = column_transformer.transform(nuevo_usuario_df)

# Hacer la predicción
prediccion = model.predict(nuevo_usuario_transformed)
protesis_recomendada = label_encoder.inverse_transform(prediccion)
print(f'Prótesis recomendada: {protesis_recomendada[0]}')


