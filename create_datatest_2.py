import cv2
import numpy as np
import os
import random

# Папки для сохранения кадров
output_folder_144p = "output_144p_frames"
output_folder_480p = "output_480p_frames"

# Создаем папки, если их еще нет
os.makedirs(output_folder_144p, exist_ok=True)
os.makedirs(output_folder_480p, exist_ok=True)

# Папки с видео
video_folder_144p = "144p"
video_folder_480p = "480p"

# Получаем список видеофайлов из папок
video_files_144p = os.listdir(video_folder_144p)[:300]
video_files_480p = os.listdir(video_folder_480p)[:300]

# Проходим по списку видеофайлов и извлекаем кадры
for i in range(300):
    video_file_144p = os.path.join(video_folder_144p, video_files_144p[i])
    video_file_480p = os.path.join(video_folder_480p, video_files_480p[i])

    # Открываем видеофайлы
    cap_144p = cv2.VideoCapture(video_file_144p)
    cap_480p = cv2.VideoCapture(video_file_480p)

    # Проверяем, что видеофайлы успешно открыты
    if not cap_144p.isOpened() or not cap_480p.isOpened():
        print(f"Ошибка: Не удается открыть видеофайлы {video_file_144p} и {video_file_480p}.")
        continue

    # Получаем общее количество кадров в видео
    frame_count = int(min(cap_144p.get(cv2.CAP_PROP_FRAME_COUNT), cap_480p.get(cv2.CAP_PROP_FRAME_COUNT)))

    # Выбираем 15 случайных индексов кадров
    random_indices = random.sample(range(frame_count), 15)

    # Извлекаем и сохраняем кадры
    for frame_index in random_indices:
        cap_144p.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        cap_480p.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        ret_144p, frame_144p = cap_144p.read()
        ret_480p, frame_480p = cap_480p.read()

        if ret_144p and ret_480p:
            # Сохраняем кадры в соответствующие папки
            frame_filename_144p = os.path.join(output_folder_144p, f"frame_{i}_{frame_index}.jpg")
            frame_filename_480p = os.path.join(output_folder_480p, f"frame_{i}_{frame_index}.jpg")

            cv2.imwrite(frame_filename_144p, frame_144p)
            cv2.imwrite(frame_filename_480p, frame_480p)
        else:
            print(f"Ошибка при чтении кадра {frame_index} из видео {video_file_144p} и {video_file_480p}")

    # Закрываем видеофайлы
    cap_144p.release()
    cap_480p.release()

print("Кадры успешно сохранены в папки:", output_folder_144p, "и", output_folder_480p)