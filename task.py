from venv import create
import cv2
import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.layers import Conv2D, UpSampling2D, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.callbacks import ModelCheckpoint
from datetime import datetime

# найти готовое решение, попробовать запустить и переобучить. 

def create_super_resolution_model(input_shape=(144, 256, 3)):
    input_img = Input(shape=input_shape)
    
    # Сверточные слои для извлечения признаков
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    
    # Увеличение разрешения изображения
    x = UpSampling2D((2, 2))(x)
    
    # Еще несколько сверточных слоев
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)

    x = UpSampling2D((2, 2))(x)
    
    # Выходной слой с тремя каналами (RGB)
    x = Conv2D(3, (3, 3), activation='relu', padding='same')(x)
    
    # Создание модели
    model = Model(inputs=input_img, outputs=x)
    return model

def load_and_preprocess_images(folder_path):
    images = []
    count = 0  # Счетчик загруженных изображений
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".jpg"):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (1024, 576))  # Масштабируем изображение
            images.append(img)
            count += 1
            if count >= 600:
                break
    return np.array(images)

def load_images(folder_path):
    images = []
    count = 0  # Счетчик загруженных изображений
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".jpg"):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (256, 144))  # Масштабируем изображение
            images.append(img)
            count += 1
            if count >= 600:
                break
    return np.array(images)

# Папка для сохранения изображений
output_folder = 'output_480p_frames'
image_folder = 'output_144p_frames'

# Создаем папку для сохранения весов модели
if not os.path.exists('model_weights'):
    os.makedirs('model_weights')

# Создаем колбэк для бэкапа модели
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    filepath='2nd/super_resolution_model_{epoch:02d}.h5',
    save_freq='epoch',
    period = 1,
    verbose=1
)

# Предобрабатываем изображения и загружаем их
target_images = load_and_preprocess_images(output_folder)
input_images = load_images(image_folder)


# Нормализуем изображения к диапазону [0, 1]
target_images = target_images / 255.0
input_images = input_images / 255.0

# Создаем модель
model = keras.models.load_model('super_resolution_model_20230924044006.h5')

# Компилируем модель
model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())


# Обучаем модель на предобработанных изображениях
history = model.fit(
    input_images,
    target_images,
    batch_size=16,
    epochs=25,  # количество эпох
    validation_split=0.1,
    callbacks=[checkpoint_callback]
)

# Сохраняем обученную модель
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
model.save(f'super_resolution_model_{timestamp}.h5')