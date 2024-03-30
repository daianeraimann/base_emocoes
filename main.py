import os
import cv2
import numpy as np
import tensorflow as tf

# Função para carregar e pré-processar as imagens
def load_and_preprocess_images(directory):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg'):
            image_path = os.path.join(directory, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Carrega imagem em escala de cinza
            image = cv2.resize(image, (128, 128))  # Redimensiona para 128x128 pixels
            image = image / 255.0  # Normaliza os valores dos pixels
            images.append(image)
            if 'happy' in directory:
                labels.append(1)  # 1 para imagens felizes
            elif 'sad' in directory:
                labels.append(0)  # 0 para imagens tristes
    return np.array(images), np.array(labels)

# Carrega e pré-processa as imagens de treinamento
happy_images, happy_labels = load_and_preprocess_images('./happy')
sad_images, sad_labels = load_and_preprocess_images('./sad')

# Concatena as imagens e os rótulos
all_images = np.concatenate([happy_images, sad_images], axis=0)
all_labels = np.concatenate([happy_labels, sad_labels], axis=0)

# Cria o modelo de rede neural
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(128, 128)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compila o modelo
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Treina o modelo
model.fit(all_images, all_labels, epochs=10)

# Função para fazer previsões
def predict_emotion(image_path):
    # Carrega e pré-processa a imagem carregada pelo usuário
    user_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    user_image = cv2.resize(user_image, (128, 128))
    user_image = user_image / 255.0
    user_image = np.expand_dims(user_image, axis=0)

    # Faz a previsão usando o modelo treinado
    prediction = model.predict(user_image)

    # Retorna a mensagem correspondente com base na previsão
    if prediction < 0.5:
        return "Que pena... Me parece que você está triste. Posso fazer algo para lhe ajudar?"
    else:
        return "Que bom, me parece que você está feliz!"

# Função para permitir que o usuário insira uma imagem
def load_user_image():
    image_path = input("Insira o caminho da imagem que deseja analisar: ")
    if not os.path.exists(image_path):
        print("Caminho de imagem inválido.")
        return
    result = predict_emotion(image_path)
    print(result)

# Exemplo de uso:
load_user_image()
