import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# Título de la app
st.title('Clasificador de Dígitos Manuscritos - MNIST')

# Descripción breve
st.write("Este proyecto utiliza una red neuronal para clasificar imágenes de dígitos manuscritos del dataset MNIST.")

# Cargar y preprocesar el conjunto de datos MNIST
@st.cache_data
def load_data():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalización
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = load_data()

# Mostrar algunas imágenes del conjunto de entrenamiento
st.subheader("Ejemplos de imágenes del conjunto de entrenamiento")
fig, ax = plt.subplots(5, 5, figsize=(10, 10))
for i in range(25):
    ax[i // 5, i % 5].imshow(x_train[i], cmap="gray")
    ax[i // 5, i % 5].axis("off")
    ax[i // 5, i % 5].set_title(f"Dígito: {y_train[i]}")
st.pyplot(fig)

# Definir el modelo de la red neuronal
@st.cache_resource
def build_model():
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Entrenar el modelo
if st.button('Entrenar Modelo'):
    st.write("Entrenando el modelo...")

    model = build_model()
    history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

    # Evaluar el modelo
    st.subheader("Resultados del entrenamiento")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    st.write(f'**Precisión en el conjunto de prueba:** {test_acc:.4f}')

    # Graficar precisión y pérdida durante el entrenamiento
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Gráfico de precisión
    ax1.plot(history.history['accuracy'], label='Precisión en entrenamiento')
    ax1.plot(history.history['val_accuracy'], label='Precisión en validación')
    ax1.set_xlabel('Épocas')
    ax1.set_ylabel('Precisión')
    ax1.legend()

    # Gráfico de pérdida
    ax2.plot(history.history['loss'], label='Pérdida en entrenamiento')
    ax2.plot(history.history['val_loss'], label='Pérdida en validación')
    ax2.set_xlabel('Épocas')
    ax2.set_ylabel('Pérdida')
    ax2.legend()

    st.pyplot(fig)

    # Hacer predicciones
    predicciones = model.predict(x_test)

    # Mostrar algunas predicciones junto con sus imágenes
    st.subheader("Ejemplos de predicciones")
    fig, ax = plt.subplots(5, 5, figsize=(10, 10))
    for i in range(25):
        ax[i // 5, i % 5].imshow(x_test[i], cmap="gray")
        ax[i // 5, i % 5].axis("off")
        pred_label = np.argmax(predicciones[i])
        true_label = y_test[i]
        color = "blue" if pred_label == true_label else "red"
        ax[i // 5, i % 5].set_title(f"Pred: {pred_label}, Real: {true_label}", color=color)
    st.pyplot(fig)
