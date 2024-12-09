import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# Paso 1: Cargar el modelo y los codificadores
modelo = joblib.load('./models/modelo_protesis.joblib')  # Cargar el modelo entrenado
le_amputacion = joblib.load('./models/le_amputacion.joblib')  # Cargar el codificador de amputación
le_lado = joblib.load('./models/le_lado.joblib')  # Cargar el codificador del lado
le_modelo = joblib.load('./models/le_modelo.joblib')  # Cargar el codificador del modelo de prótesis

# Paso 2: Definir las entradas del nuevo caso
amputacion_input = "Brazo (Abajo del Codo)"  # Ejemplo de tipo de amputación
lado_input = "Izquierda"  # Ejemplo de lado de la prótesis

# Paso 3: Codificar las entradas utilizando los codificadores cargados
amputacion_input_encoded = le_amputacion.transform([amputacion_input])  # Codificar tipo de amputación
lado_input_encoded = le_lado.transform([lado_input])  # Codificar lado de la prótesis

# Paso 4: Crear el DataFrame para la entrada (según el formato usado para entrenar)
X_input = pd.DataFrame({'amputacionusuario': amputacion_input_encoded, 
                        'ladodeprotesisusuario': lado_input_encoded})

# Paso 5: Realizar la predicción con el modelo cargado
protesis_predicha = modelo.predict(X_input)

# Paso 6: Convertir la predicción a su forma original (nombre de la prótesis)
protesis_final = le_modelo.inverse_transform(protesis_predicha)

# Paso 7: Mostrar la prótesis recomendada
print(f"La prótesis recomendada para una amputación de tipo '{amputacion_input}' en el lado '{lado_input}' es: {protesis_final[0]}")
