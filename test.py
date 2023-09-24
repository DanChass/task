import cv2
import os
import numpy as np
from tensorflow import keras

def load_test_images(folder_path):
    images = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".jpg"):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (256, 144))  # Масштабируем изображение
            images.append(img)
    return np.array(images)

def convert_video_to_images(video_path, output_folder):
    # Проверяем, существует ли папка для сохранения изображений, и создаем ее, если она не существует
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Открываем видеофайл
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0

    # Перебираем кадры видео
    while True:
        ret, frame = cap.read()
        
        # Если кадры закончились или достигнут лимит, завершаем цикл
        if not ret:
            break

        # Сохраняем кадр как изображение .jpg
        img_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(img_filename, frame)
        frame_count += 1

    # Закрываем видеофайл
    cap.release()

# Папка с тестовыми изображениями
test_image_folder = 'frames_bad'

# Конвертируем видео в картинки
convert_video_to_images('0_144.mp4', test_image_folder)

# Загрузка и предобработка тестовых изображений
test_images = load_test_images(test_image_folder)

# Загрузка обученной модели
model = keras.models.load_model('super_resolution_model_20230924092656.h5')

output_folder = 'super_res_test_results'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Сделаем предсказания и сохранение улучшенных изображений одновременно
for i, test_image in enumerate(test_images):
    # Предсказываем улучшенное изображение
    super_res_image = model.predict(np.expand_dims(test_image / 255.0, axis=0))[0]

    # Сохраняем улучшенное изображение
    img_filename = os.path.join(output_folder, f"super_res_image_{i:04d}.jpg")
    super_res_image = (super_res_image * 255).astype(np.uint8)
    cv2.imwrite(img_filename, super_res_image)


images = [img for img in os.listdir(output_folder) if img.endswith(".jpg")]

# Получить размер первого изображения (предполагается, что все изображения имеют одинаковый размер)
frame = cv2.imread(os.path.join(output_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(output_folder, image)))

cv2.destroyAllWindows()
video.release()