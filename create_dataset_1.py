import os
import shutil

# Путь к папке с видео
video_folder = 'train'

# Создаем папки '144p' и '480p', если они не существуют
os.makedirs('144p', exist_ok=True)
os.makedirs('480p', exist_ok=True)

# Проходим по всем файлам в папке 'train'
for filename in os.listdir(video_folder):
    if filename.endswith('_144.mp4'):
        # Если файл заканчивается на '_144.mp4', перемещаем его в папку '144p'
        shutil.move(os.path.join(video_folder, filename), os.path.join('144p', filename))
    elif filename.endswith('_480.mp4'):
        # Если файл заканчивается на '_480.mp4', перемещаем его в папку '480p'
        shutil.move(os.path.join(video_folder, filename), os.path.join('480p', filename))
    else:
        # Если файл не соответствует ни одному из форматов, можно его проигнорировать
        pass

print("Видео успешно перемещены в соответствующие папки.")