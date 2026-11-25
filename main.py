from PIL import Image
import numpy as np
import datetime

INPUT_DIR = "Lab2_Media/data/input/"
OUTPUT_PATH = "Lab2_Media/data/output/result.jpg"
LOG_PATH = "Lab2_Media/logs/history.txt"
FILTER_NAME = "Свёртка (Гаусс 3x3) + Автоконтраст"

KERNEL = np.array([[1, 2, 1],
                   [2, 4, 2],
                   [1, 2, 1]], dtype=float) / 16


def load_image(name):
    path = INPUT_DIR + name
    with open(path, "rb+") as f:
        img = Image.open(f)
        img = img.convert("RGB")
        arr = np.array(img, dtype=float)
    return arr


def convolve(arr, kernel):
    h = arr.shape[0]
    w = arr.shape[1]
    pad = np.pad(arr, ((1, 1), (1, 1), (0, 0)), mode="edge")
    result = np.zeros((h, w, 3))
    for i in range(3):
        for j in range(3):
            result = result + kernel[i][j] * pad[i:i + h, j:j + w]
    return result


def autocontrast(arr):
    result = np.zeros(arr.shape)
    for c in range(3):
        channel = arr[:, :, c]
        y_min = channel.min()
        y_max = channel.max()
        if y_max > y_min:
            result[:, :, c] = (channel - y_min) * 255 / (y_max - y_min)
        else:
            result[:, :, c] = channel
    return result


def save_image(arr):
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(OUTPUT_PATH)


def write_log(name, status):
    time = datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    with open(LOG_PATH, "a", encoding="UTF-8") as log:
        log.write(f"[{time}] Файл: {name} | Фильтр: {FILTER_NAME} | Статус: {status}\n")


def process(name):
    try:
        arr = load_image(name)
        arr = convolve(arr, KERNEL)
        arr = autocontrast(arr)
        save_image(arr)
        print("Готово! Результат сохранен в папку output")
        return "Успешно"
    except FileNotFoundError:
        print("Ошибка: Файл не найден")
        return "Файл не найден"
    except PermissionError:
        print(f"Ошибка: Нет прав на чтение файла {name}. Обратитесь к администратору")
        return "Ошибка доступа"


def main():
    while True:
        name = input("Введите имя файла (пустая строка - выход): ")
        name = name.strip()
        if name == "":
            break
        status = process(name)
        write_log(name, status)


if __name__ == "__main__":
    main()