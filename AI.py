import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance

# Путь к датасету
DATASET_PATH = 'dataset/'
class_names = sorted(os.listdir(DATASET_PATH))  # Сортируем для предсказуемости

# Определяем режим классификации
num_classes = len(class_names)
class_mode = "binary" if num_classes == 2 else "categorical"

def enhance_sharpness(image_path):
    """Повышает чёткость изображения с помощью PIL"""
    image = Image.open(image_path)
    enhancer = ImageEnhance.Sharpness(image)
    enhanced_image = enhancer.enhance(2.0)  # 2.0 = резче, можно варьировать
    return cv2.cvtColor(np.array(enhanced_image), cv2.COLOR_RGB2BGR)

def predict_image(image_path):
    # Проверка на существование файла
    if not os.path.exists(image_path):
        print(f"❌ Ошибка: Файл не найден по пути: {image_path}")
        return

    # Проверка на повреждение
    try:
        Image.open(image_path).verify()
    except (OSError, IOError):
        print(f"❌ Ошибка: Поврежденное изображение - {image_path}")
        return

    # Загрузка и предобработка изображения
    img = enhance_sharpness(image_path)
    if img is None:
        print(f"❌ Ошибка: Не удалось прочитать изображение - {image_path}")
        return

    img = cv2.resize(img, (128, 128))
    img = img.astype("float32") / 255.0  # Нормализация
    img = tf.expand_dims(img, axis=0)   # Добавляем размерность батча

    # Загрузка модели
    try:
        model = tf.keras.models.load_model("image_classifier.h5")
    except Exception as e:
        print(f"❌ Ошибка при загрузке модели: {e}")
        return

    # Предсказание
    prediction = model.predict(img)
    if class_mode == "binary":
        predicted_class = class_names[int(prediction[0] > 0.5)]
    else:
        predicted_class = class_names[int(tf.argmax(prediction, axis=1)[0])]

    # Отображение изображения
    original_image = Image.open(image_path)
    plt.imshow(original_image)
    plt.title(f"Модель определила: {predicted_class}")
    plt.axis('off')
    plt.show()

    print(f"✅ Модель определила: {predicted_class}")

# Пример запуска
predict_image("dataset/dogs/australian-shepherd-3237735_1280.jpg")
