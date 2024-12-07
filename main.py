import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
# Resto de tu código

# Cargar datos
data = pd.read_csv('datos_protesis.csv')

# Limpiar los nombres de las columnas
data.columns = data.columns.str.strip()

# Seleccionar solo las columnas relevantes
data_relevant = data[['Nombre de Dispositivo', 'Categoria']]

# Codificar las etiquetas (Nombre de Dispositivo)
label_encoder = LabelEncoder()
data_relevant['Nombre de Dispositivo'] = label_encoder.fit_transform(data_relevant['Nombre de Dispositivo'])

# Convertir la columna 'Categoria' en variables dummy (One-Hot Encoding)
data_dummies = pd.get_dummies(data_relevant, columns=['Categoria'])


# Separar características (X) y etiquetas (y)
X = data_dummies.drop(columns=['Nombre de Dispositivo'])
y = data_dummies['Nombre de Dispositivo']

# Convertir booleanos a flotantes para TensorFlow
X = X.astype('float32')

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))  # Capa de entrada con el número de características
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))  # Número de clases

#print(X_train.dtypes)
#print(y_train.dtypes)

# Compilar el modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=50, batch_size=10, validation_split=0.2)

# Evaluar el modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Precisión del modelo: {accuracy * 100:.2f}%')
