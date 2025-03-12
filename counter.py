import cv2
import os
from ultralytics import YOLO
import matplotlib.pyplot as plt

# ==== Конфигурация ====
VIDEO_PATH = 'input_timelapse.mov'
FRAMES_DIR = 'frames'
FPS_EXTRACTION = 15  # Сколько кадров в секунду извлекать (таймлапс может быть быстро ускорен)
MODEL_PATH = 'yolov8n.pt'  # Легкая модель YOLOv8
CONFIDENCE_THRESHOLD = 0.3  # Порог уверенности для учёта объектов
LINE_Y = 200  # Координаты по Y для линии пересечения (можно настроить)


# =======================

# 1. Извлечение кадров из видео
def extract_frames(video_path, output_dir, fps=1):
    os.makedirs(output_dir, exist_ok=True)
    vidcap = cv2.VideoCapture(video_path)
    video_fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)

    success, frame = vidcap.read()
    count = 0
    frame_count = 0

    print(f"[INFO] Извлечение кадров каждые {frame_interval} кадров (цель: {fps} кадр/сек)")

    while success:
        if count % frame_interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{frame_count:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_count += 1
        success, frame = vidcap.read()
        count += 1

    print(f"[INFO] Извлечено {frame_count} кадров в папку '{output_dir}'.")


# 2. Подсчет проходов через линию
def count_passes(frames_dir, model_path, conf_thres=0.3, line_y=200):
    model = YOLO(model_path)
    frame_files = sorted(os.listdir(frames_dir))
    passes_count = 0
    crossing_lines = set()  # Множество для отслеживания людей

    for idx, frame_file in enumerate(frame_files):
        frame_path = os.path.join(frames_dir, frame_file)
        results = model(frame_path, conf=conf_thres)

        # Отслеживаем людей, их координаты
        for box in results[0].boxes:
            class_id = int(box.cls)
            if class_id == 0:  # Только люди (class 0)
                # Получаем координаты bounding box
                x1, y1, x2, y2 = map(int, box.xywh[0])

                # Если человек пересекает линию (проверка по y-координате)
                if y2 > line_y and (frame_file not in crossing_lines):
                    passes_count += 1
                    crossing_lines.add(frame_file)  # Помечаем этот кадр как пересеченный

        print(f"[{idx + 1}/{len(frame_files)}] {frame_file}: пересечено — {passes_count} раз.")

    return passes_count


# 3. Построение графика количества проходов
def plot_passes(passes_count):
    plt.figure(figsize=(12, 6))
    plt.plot([passes_count], marker='o', linestyle='-', color='red', label='Количество проходов')
    plt.xlabel('Время')
    plt.ylabel('Количество проходов')
    plt.title('Динамика проходов людей по таймлапсу')
    plt.legend()
    plt.grid(True)
    plt.show()


# === Основной запуск ===
if __name__ == "__main__":
    # Шаг 1: Извлечь кадры
    extract_frames(VIDEO_PATH, FRAMES_DIR, FPS_EXTRACTION)

    # Шаг 2: Подсчитать проходы через линию
    passes = count_passes(FRAMES_DIR, MODEL_PATH, CONFIDENCE_THRESHOLD, LINE_Y)

    # Шаг 3: Построить график
    plot_passes(passes)

    # Шаг 4: Печать итоговой суммы
    print(f"\n[ИТОГО] Всего проходов через камеру: {passes}")
