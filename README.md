## Языки | Languages
- [Русский](#русский)
- [English](#english)

---

## Русский

# Веб-приложение для обнаружения нарушений на экзамене

### Описание
Это веб-приложение на Streamlit для автоматического обнаружения нарушений на экзамене с помощью компьютерного зрения. Система позволяет:
- Загружать видео с экзамена (фронтальный и тыловой план)
- Извлекать изображения людей из видео
- Размечать людей как "студент" или "преподаватель"
- Загружать свои модели для классификации людей и детекции телефонов (YOLOv8)
- Детектировать телефоны и связывать их с людьми на видео
- Получать подробный отчет о нарушениях с возможностью скачивания

### Технические детали
- Язык: Python 3.8+
- Фреймворк: [Streamlit](https://streamlit.io/)
- Детекция и классификация: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), [torchvision](https://pytorch.org/vision/)
- Обработка видео: OpenCV
- Работа с изображениями: PIL
- Веб-интерфейс: Streamlit, streamlit-player
- Вся логика реализована в одном файле: `web_service.py`

#### Структура кода
- `web_service.py` — основной файл приложения, содержит все классы и функции:
  - Классы для трекинга, детекции, классификации
  - Функции для извлечения людей из видео, разметки, обработки кадров, генерации отчетов
  - Streamlit-интерфейс с тремя вкладками: загрузка, разметка, детекция
- Конфигурационные файлы: `data.yaml`

#### Используемые модели
- YOLOv8 для детекции людей и телефонов (можно загрузить свои веса)
- Классификатор людей (студент/преподаватель) — поддержка кастомных моделей
Имеются 2 предобученные модели для использования, которые находятся по пути: detect_exam_cheating\yolo_phone_training\phone_detection\weights\best.pt, detect_exam_cheating\yolo_cls_training\person_classifier5\weights\best.pt

Возможно телефон не сможет детектиться, если видео имеет низкое разрешение!

### Как запустить
1. Установите Python 3.8+ и pip
2. Установите зависимости:
   ```
   pip install -r requirements.txt
   ```
3. Запустите приложение:
   ```
   streamlit run web_service.py
   ```
4. Перейдите по адресу, который покажет Streamlit (обычно http://localhost:8501)

#### Требования
- Python 3.8+
- CUDA (опционально, для ускорения на GPU)
- Все зависимости указаны в requirements.txt

#### Подготовка данных и моделей
- Видео для анализа загружаются через веб-интерфейс
- Для кастомных моделей YOLOv8 (детекция телефонов, классификация людей) загрузите свои `.pt` файлы через интерфейс

### Использование веб-интерфейса
- **Вкладка "Загрузка и Настройки"**: загрузка видео и моделей, настройка параметров извлечения
- **Вкладка "Разметка"**: ручная разметка людей (студент/преподаватель)
- **Вкладка "Детекция"**: запуск анализа, просмотр результатов, скачивание отчета и размеченного видео

## Функционал веб-приложения:
![Alt text](https://github.com/Sol1tud9/detect_exam_cheating/blob/main/images/schemerus.png?raw=true)

![Alt text](https://github.com/Sol1tud9/detect_exam_cheating/blob/main/images/interfaceRus.png?raw=true)

![Alt text](https://github.com/Sol1tud9/detect_exam_cheating/blob/main/images/crops.png?raw=true)

![Alt text](https://github.com/Sol1tud9/detect_exam_cheating/blob/main/images/labeling.png?raw=true)

![Alt text](https://github.com/Sol1tud9/detect_exam_cheating/blob/main/images/labels.png?raw=true)

![Alt text](https://github.com/Sol1tud9/detect_exam_cheating/blob/main/images/tracking.png?raw=true)

![Alt text](https://github.com/Sol1tud9/detect_exam_cheating/blob/main/images/detection.png?raw=true)

![Alt text](https://github.com/Sol1tud9/detect_exam_cheating/blob/main/images/report.png?raw=true)

## English

# Exam Violations Detection Web Application

### Description
This is a Streamlit-based web application for automatic detection of exam violations using computer vision. The system allows you to:
- Upload exam videos (front and back view)
- Extract images of people from video
- Label people as "student" or "teacher"
- Upload your own models for person classification and phone detection (YOLOv8)
- Detect phones and associate them with people in the video
- Get a detailed violations report with download option

### Technical Details
- Language: Python 3.8+
- Framework: [Streamlit](https://streamlit.io/)
- Detection & Classification: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), [torchvision](https://pytorch.org/vision/)
- Video processing: OpenCV
- Image processing: PIL
- Web interface: Streamlit, streamlit-player
- All logic is implemented in a single file: `web_service.py`

#### Code Structure
- `web_service.py` — main application file, contains all classes and functions:
  - Classes for tracking, detection, classification
  - Functions for extracting people from video, labeling, frame processing, report generation
  - Streamlit interface with three tabs: upload, labeling, detection
- Config files: `data.yaml`

#### Used Models
- YOLOv8 for person and phone detection (you can upload your own weights)
- Person classifier (student/teacher) — custom models supported

### How to Run
1. Install Python 3.8+ and pip
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   streamlit run web_service.py
   ```
4. Open the address shown by Streamlit (usually http://localhost:8501)

#### Requirements
- Python 3.8+
- CUDA (optional, for GPU acceleration)
- All dependencies are listed in requirements.txt

#### Data and Model Preparation
- Upload videos for analysis via the web interface
- For custom YOLOv8 models (phone detection, person classification), upload your `.pt` files via the interface

There are 2 pre-trained models to use, which are located at: detect_exam_cheating\yolo_phone_training\phone_detection\weights\best.pt, detect_exam_cheating\yolo_cls_training\person_classifier5\weights\best.pt

The phone may not be detected if the video has a low resolution!

### Web Interface Usage
- **"Upload & Settings" tab**: upload videos and models, set extraction parameters
- **"Labeling" tab**: manually label people (student/teacher)
- **"Detection" tab**: run analysis, view results, download report and annotated video

## Web application functionality:
![Alt text](https://github.com/Sol1tud9/detect_exam_cheating/blob/main/images/schemeeng.png?raw=true)

![Alt text](https://github.com/Sol1tud9/detect_exam_cheating/blob/main/images/interfaceEng.png?raw=true)

![Alt text](https://github.com/Sol1tud9/detect_exam_cheating/blob/main/images/cropsEng.png?raw=true)

![Alt text](https://github.com/Sol1tud9/detect_exam_cheating/blob/main/images/labelsEng.png?raw=true)

![Alt text](https://github.com/Sol1tud9/detect_exam_cheating/blob/main/images/saving.png?raw=true)

![Alt text](https://github.com/Sol1tud9/detect_exam_cheating/blob/main/images/trackingEng.png?raw=true)

![Alt text](https://github.com/Sol1tud9/detect_exam_cheating/blob/main/images/detectionEng.png?raw=true)

![Alt text](https://github.com/Sol1tud9/detect_exam_cheating/blob/main/images/reportEng.png?raw=true)
