import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Definir la arquitectura del modelo
model = tf.keras.Sequential([
    layers.Conv1D(32, 3, activation='relu', input_shape=(n, 20)), # n es la longitud de la secuencia
    layers.Conv1D(64, 3, activation='relu'),
    layers.GlobalMaxPooling1D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(3) # 3 coordenadas para predecir la estructura tridimensional
])

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

# Evaluar el modelo
loss = model.evaluate(X_test, y_test)

# Realizar predicciones
predictions = model.predict(X_test)
