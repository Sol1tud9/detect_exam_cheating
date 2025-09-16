import streamlit as st
import streamlit.components.v1 as components
import cv2
import os
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
from ultralytics import YOLO
import torch
from datetime import datetime
import time
import glob
import shutil
from collections import defaultdict, deque
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from streamlit_player import st_player
import copy


//старая модель для детекции телефонов
class PhoneDetectorResNet50(nn.Module):
    def __init__(self, pretrained=True):
        super(PhoneDetectorResNet50, self).__init__()
        self.resnet = models.resnet50(pretrained=pretrained)
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])
        self.phone_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(), 
            nn.Dropout(0.5), 
            nn.Linear(2048, 512), 
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1), 
            nn.Sigmoid() 
        )
    
    def forward(self, x):
        features = self.features(x)
        phone_score = self.phone_classifier(features)
        return phone_score

class ObjectTracker:
    def __init__(self, max_disappeared=30, max_distance=150):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
    def register(self, bbox, centroid, class_name, confidence):
        self.objects[self.next_id] = {
            'bbox': bbox,
            'centroid': centroid,
            'class': class_name,
            'confidence': confidence,
            'history': deque(maxlen=20),
            'first_seen': datetime.now(),
            'color': self._generate_color()
        }
        self.objects[self.next_id]['history'].append(centroid)
        self.disappeared[self.next_id] = 0
        self.next_id += 1
        
    def _generate_color(self):
        np.random.seed(self.next_id)
        return tuple(map(int, np.random.randint(0, 255, 3)))
        
    def deregister(self, object_id):
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]
        
    def update(self, detections):
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
        
        if len(self.objects) == 0:
            for detection in detections:
                bbox, centroid, class_name, confidence = detection
                self.register(bbox, centroid, class_name, confidence)
        else:
            object_centroids = [obj['centroid'] for obj in self.objects.values()]
            object_ids = list(self.objects.keys())
            
            detection_centroids = [det[1] for det in detections]

            if len(object_centroids) > 0 and len(detection_centroids) > 0:
                D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - 
                                 np.array(detection_centroids), axis=2)

                rows = D.min(axis=1).argsort()
                cols = D.argmin(axis=1)[rows]
                
                used_row_indices = set()
                used_col_indices = set()
                
                for (row, col) in zip(rows, cols):
                    if row in used_row_indices or col in used_col_indices:
                        continue
                        
                    if D[row, col] > self.max_distance:
                        continue
                        
                    object_id = object_ids[row]
                    bbox, centroid, class_name, confidence = detections[col]
                    
                    self.objects[object_id]['bbox'] = bbox
                    self.objects[object_id]['centroid'] = centroid
                    self.objects[object_id]['class'] = class_name
                    self.objects[object_id]['confidence'] = confidence
                    self.objects[object_id]['history'].append(centroid)
                    self.disappeared[object_id] = 0
                    
                    used_row_indices.add(row)
                    used_col_indices.add(col)
                
                unused_rows = set(range(0, D.shape[0])).difference(used_row_indices)
                unused_cols = set(range(0, D.shape[1])).difference(used_col_indices)
                
                if D.shape[0] >= D.shape[1]:
                    for row in unused_rows:
                        object_id = object_ids[row]
                        self.disappeared[object_id] += 1
                        
                        if self.disappeared[object_id] > self.max_disappeared:
                            self.deregister(object_id)
                else:
                    for col in unused_cols:
                        bbox, centroid, class_name, confidence = detections[col]
                        self.register(bbox, centroid, class_name, confidence)
        
        return self.objects

class RealTimeVideoObjectDetector:
    def __init__(self, person_classifier_model_path, yolo_phone_model_path=None, yolo_model='yolov8n.pt',
                 yolo_conf_detection=0.3, person_classification_conf=0.5, phone_detection_conf=0.5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Используется устройство: {self.device}")

        self.yolo_model = YOLO(yolo_model)
        self.yolo_model.to(self.device)
        print(f"Загружена YOLO модель для людей: {yolo_model}")

        self.phone_yolo_model = None
        if yolo_phone_model_path and os.path.exists(yolo_phone_model_path):
            self.phone_yolo_model = YOLO(yolo_phone_model_path)
            self.phone_yolo_model.to(self.device)
            print(f"Загружена кастомная YOLO модель для телефонов: {yolo_phone_model_path}")

        self.person_classifier_model = None
        if person_classifier_model_path and os.path.exists(person_classifier_model_path):
            self.person_classifier_model = YOLO(person_classifier_model_path)
            self.person_classifier_model.to(self.device)
            print(f"Загружена YOLO модель для классификации людей: {person_classifier_model_path}")

        self.yolo_conf_detection = yolo_conf_detection
        self.person_classification_conf = person_classification_conf
        self.phone_detection_conf = phone_detection_conf
        print(f"Пороги уверенности: YOLO детекция={self.yolo_conf_detection}, Классификация человека={self.person_classification_conf}, Детекция телефона={self.phone_detection_conf}")
        
        self.tracker = ObjectTracker(max_disappeared=30, max_distance=150)
        
        self.colors = {
            'student': (0, 255, 0),
            'teacher': (255, 0, 0),
            'phone': (0, 0, 255),
            'person': (255, 255, 0)
        }
        
        self.detection_stats = defaultdict(int)
        self.frame_count = 0
        self.fps_counter = deque(maxlen=30)
        
    def detect_objects_in_frame(self, frame):
        detections = []
        
        person_results = self.yolo_model(frame, conf=self.yolo_conf_detection, classes=[0], verbose=False)

        for result in person_results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    person_roi = frame[y1:y2, x1:x2]
                    
                    if person_roi.size == 0:
                        continue
                    
                    final_class = 'person'
                    final_confidence = confidence
                    
                    if self.person_classifier_model is not None:
                        _class_confidence = None
                        _predicted_index = None

                        try:
                            cls_results = self.person_classifier_model(person_roi, verbose=False)

                            if cls_results and len(cls_results) > 0 and cls_results[0] is not None and hasattr(cls_results[0], 'probs'):
                                probs = cls_results[0].probs

                                if probs is not None:
                                    if isinstance(probs, torch.Tensor):
                                        _class_confidence, _predicted_index = probs.max(0)
                                        _class_confidence = _class_confidence.item()
                                        _predicted_index = _predicted_index.item()
                                    elif hasattr(probs, 'top1conf') and hasattr(probs, 'top1'): # Assume Probs object if not Tensor
                                        _class_confidence = probs.top1conf
                                        _predicted_index = probs.top1
                                        if isinstance(_class_confidence, torch.Tensor):
                                            _class_confidence = _class_confidence.item()
                                    else:
                                        print(f"WARNING: Unexpected type or malformed probs object ({type(probs)}). Skipping detailed classification.")
                                else:
                                    print("WARNING: cls_results[0].probs returned None. Skipping detailed classification.")
                            else:
                                print("WARNING: No valid classification result object (cls_results is empty/None or missing .probs). Skipping detailed classification.")
                        except Exception as e:
                            print(f"Ошибка классификации человека: {e}")

                        if _class_confidence is not None and _class_confidence > self.person_classification_conf:
                            final_class = self.person_classifier_model.names[_predicted_index]
                            final_confidence = _class_confidence
                    
                    centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    bbox = (x1, y1, x2, y2)
                    detections.append((bbox, centroid, final_class, final_confidence))
        
        if self.phone_yolo_model is not None:
            phone_results = self.phone_yolo_model(frame, conf=self.phone_detection_conf, verbose=False)
            for result in phone_results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        
                        centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                        bbox = (int(x1), int(y1), int(x2), int(y2))
                        detections.append((bbox, centroid, 'phone', confidence))

        return detections
    
    def draw_tracking_info(self, frame, objects):
        for object_id, obj_info in objects.items():
            bbox = obj_info['bbox']
            centroid = obj_info['centroid']
            class_name = obj_info['class']
            confidence = obj_info['confidence']
            
            color = self.colors.get(class_name, obj_info.get('color', (255, 255, 255)))
            
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            cv2.circle(frame, centroid, 5, color, -1)
            
            text = f"ID:{object_id} {class_name} ({confidence:.2f})"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            cv2.rectangle(frame, (x1, y1 - text_size[1] - 10), 
                         (x1 + text_size[0], y1), color, -1)
            cv2.putText(frame, text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if len(obj_info['history']) > 1:
                points = list(obj_info['history'])
                for i in range(1, len(points)):
                    cv2.line(frame, points[i-1], points[i], color, 2)
        
        return frame


st.set_page_config(
    page_title="Exam Cheating",
    page_icon="🎥",
    layout="wide"
)

CONFIG = {
    "STEP": 15,      
    "CONF": 0.3,
    "MAX_CROPS_PER_ID": 120  
}

def convert_df_to_csv(df):
    """Преобразует DataFrame в CSV-байты для скачивания через Streamlit."""
    return df.to_csv(index=False).encode('utf-8')

def convert_df_to_html_table(df):
    """Преобразует DataFrame в форматированную HTML-таблицу для Streamlit."""
    # Добавляем CSS-стили для улучшения внешнего вида
    html_table = """
    <style>
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        tr:hover {
            background-color: #f1f1f1;
        }
    </style>
    """
    
    # Добавляем заголовок для отчета
    report_header = "<h2>Отчет о нарушениях на экзамене</h2>"
    report_summary = f"<p>Общее количество зафиксированных нарушений: <strong>{len(df)}</strong></p>"
    
    # Преобразуем DataFrame в HTML-таблицу
    table_string = df.to_html(index=False)
    
    return (html_table + report_header + report_summary + table_string).encode('utf-8')

def build_dataset(crops_dir, labels_file, dataset_dir, target_size=(150, 150)):
    """
    Формирует датасет для классификации людей (студент/преподаватель) по кропам и меткам.
    Копирует и ресайзит изображения в train/val директории.
    """
    for split in ['train', 'val']:
        for label in ['student', 'teacher']:
            Path(dataset_dir, split, label).mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(labels_file)
    label_map = dict(zip(df.tid.astype(str), df.label))
    
    for img_path in glob.glob(os.path.join(crops_dir, "*.jpg")):
        tid = os.path.basename(img_path).split("_")[0]
        label = label_map.get(tid)
        if label:
            split = 'val' if np.random.random() < 0.15 else 'train'
            dst_path = os.path.join(dataset_dir, split, label, os.path.basename(img_path))
            shutil.copy(img_path, dst_path)

            img = Image.open(img_path)
            img = img.resize(target_size)
            img.save(dst_path)


def initialize_detector(person_classifier_model_path=None, yolo_phone_model_path=None, 
                        yolo_conf_det=0.3, person_class_conf=0.5, phone_det_conf=0.5): 
    """
    Инициализирует детектор объектов (люди/телефоны) с выбранными моделями и порогами.
    Кэширует в сессии Streamlit для ускорения повторных запусков.
    """
    if 'detector' not in st.session_state or \
        st.session_state.get('person_model_path_loaded') != person_classifier_model_path or \
        st.session_state.get('phone_model_path_loaded') != yolo_phone_model_path or \
        st.session_state.get('yolo_conf_det_loaded') != yolo_conf_det or \
        st.session_state.get('person_class_conf_loaded') != person_class_conf or \
        st.session_state.get('phone_det_conf_loaded') != phone_det_conf:
        
        with st.spinner('Загрузка моделей и настройка детектора...'):
            try:
                detector = RealTimeVideoObjectDetector(
                    person_classifier_model_path=person_classifier_model_path,
                    yolo_phone_model_path=yolo_phone_model_path,  
                    yolo_model='yolov8n.pt',
                    yolo_conf_detection=yolo_conf_det,           
                    person_classification_conf=person_class_conf, 
                    phone_detection_conf=phone_det_conf          
                )
                st.session_state.detector = detector
                st.session_state.person_model_path_loaded = person_classifier_model_path
                st.session_state.phone_model_path_loaded = yolo_phone_model_path
                st.session_state.yolo_conf_det_loaded = yolo_conf_det
                st.session_state.person_class_conf_loaded = person_class_conf
                st.session_state.phone_det_conf_loaded = phone_det_conf

                if person_classifier_model_path:
                    st.success("✅ Модель классификации людей (YOLO) загружена/обновлена")
                else:
                    st.info("ℹ️ Модель классификации людей не загружена, будет использоваться только YOLO для людей")
                
                if yolo_phone_model_path:
                    st.success("✅ Модель детекции телефонов (YOLO) загружена/обновлена")
                else:
                    st.info("ℹ️ Модель детекции телефонов не загружена, детекция телефонов производиться не будет.")
                st.success(f"✅ Детектор инициализирован с новыми параметрами.")

            except Exception as e:
                st.error(f"❌ Ошибка инициализации детектора: {str(e)}")
                st.session_state.detector = None 
                return None
    elif st.session_state.detector is None: 
        st.error("❌ Детектор не был успешно инициализирован ранее. Проверьте модели и параметры.")
        return None
        
    return st.session_state.detector

def bbox_iou(boxA, boxB):
    """Вычисляет IoU (Intersection over Union) между двумя bbox-ами."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def extract_people_from_video(video_path, output_dir, tid_prefix, 
                              step=15, yolo_conf=0.3, max_crops=120):
    """
    Извлекает людей из видео с помощью YOLO, сохраняет кропы и метаинформацию.
    step — шаг по кадрам, max_crops — максимум кропов на одного человека.
    """
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    status_text = st.empty()
    id2cnt = {}
    frame_i = 0
    meta = []
    
    processed_frames_count = 0 
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        current_frame_for_progress = frame_i + 1 
        
        if frame_i % step: 
            frame_i += 1
            if frame_i % (step * 5) == 0 or current_frame_for_progress >= total_frames :
                 progress_bar.progress(current_frame_for_progress / total_frames)
            continue
        
        processed_frames_count +=1
        status_text.text(f"Обработка кадра {current_frame_for_progress}/{total_frames} (извлечено из {processed_frames_count})")
        res = model.track(frame, conf=yolo_conf, classes=[0], persist=True, verbose=False)[0] 
        
        if res.boxes.id is not None:
            for box, tid in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.id.cpu().numpy()):
                tid = int(tid)
                unique_tid = f"{tid_prefix}_{tid}"
                id2cnt.setdefault(unique_tid, 0)
                if id2cnt[unique_tid] >= max_crops: 
                    continue
                x1, y1, x2, y2 = map(int, box)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                fname = f"{unique_tid}_{frame_i}.jpg"
                cv2.imwrite(os.path.join(output_dir, fname), crop)
                meta.append({
                    'tid': unique_tid,
                    'frame': frame_i,
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'img': fname
                })
                id2cnt[unique_tid] += 1
        frame_i += 1
        progress_bar.progress(current_frame_for_progress / total_frames)
    
    cap.release()
    status_text.text(f"Извлечение из видео ({os.path.basename(video_path)}) завершено!")
    return id2cnt, meta


def label_people(crops_dir, meta_csv='crops_meta.csv', labels_file='labels.csv'):
    """
    Интерфейс Streamlit для ручной разметки людей как преподавателей или студентов.
    Сохраняет результат в labels.csv.
    """
    st.subheader("Выберите преподавателей")
    if not os.path.exists(meta_csv):
        st.warning("Нет метаданных для разметки!")
        return None
    meta = pd.read_csv(meta_csv)
    all_ids = sorted(meta['tid'].unique())
    if not all_ids:
        st.warning("Нет доступных изображений для разметки")
        return None
    cols_per_row = 4
    selected_teachers = set()
    for i in range(0, len(all_ids), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < len(all_ids):
                tid = all_ids[i + j]
                first_img_row = meta[meta['tid']==tid]['img'].values
                if len(first_img_row) > 0:
                    img_path = os.path.join(crops_dir, first_img_row[0])
                    if os.path.exists(img_path):
                        img = Image.open(img_path)
                        img = img.resize((180, 180))
                        col.image(img, caption=f"ID: {tid}")
                        if col.checkbox("Преподаватель", key=f"teacher_{tid}"):
                            selected_teachers.add(tid)
                    else:
                        col.warning(f"Img not found: {img_path}")
                else:
                    col.info(f"No images for ID {tid} in meta.")


    if st.button("Сохранить разметку"):
        meta['label'] = meta['tid'].apply(lambda tid: 'teacher' if tid in selected_teachers else 'student')
        meta.to_csv(labels_file, index=False)

        
        st.success("Разметка сохранена!")
        st.session_state.labeling_complete = True
        return meta
    return None


def load_bbox_labels(labels_file='labels.csv'):
    """
    Загружает bbox-метки из файла labels.csv, группирует по видео и кадру.
    Возвращает словарь для быстрого поиска меток по кадру.
    """
    by_video_and_frame = defaultdict(lambda: defaultdict(list))
    if os.path.exists(labels_file):
        df = pd.read_csv(labels_file)
        required_cols = ['frame', 'tid', 'x1', 'y1', 'x2', 'y2', 'label']
        if not all(col in df.columns for col in required_cols):
            st.error(f"Файл меток {labels_file} не содержит все необходимые колонки: {required_cols}")
            return by_video_and_frame

        for _, row in df.iterrows():
            tid = str(row['tid'])
            video_prefix = tid.split('_')[0]
            frame_num = int(row['frame'])
            by_video_and_frame[video_prefix][frame_num].append({
                'tid': row['tid'],
                'bbox': (row['x1'], row['y1'], row['x2'], row['y2']),
                'label': row['label']
            })
    return by_video_and_frame


def match_label(frame_id, bbox, by_video_and_frame, video_prefix, iou_thr=0.5):
    """
    Находит лучшую метку (label) для bbox по IoU среди размеченных объектов на кадре.
    Возвращает label и tid, если IoU выше порога.
    """
    candidates = by_video_and_frame.get(video_prefix, {}).get(frame_id, [])
    best_iou = 0
    best_label = None
    best_tid = None
    for c in candidates:
        iou = bbox_iou(bbox, c['bbox'])
        if iou > best_iou:
            best_iou = iou
            best_label = c['label']
            best_tid = c['tid']
    if best_iou > iou_thr:
        return best_label, best_tid
    return None, None


def process_video_frame(frame, detector, frame_idx=None, bbox_labels_by_video_and_frame=None, video_prefix=None, tid_manual_labels_map=None):
    """
    Обрабатывает кадр видео: детектирует объекты, трекает их, сопоставляет с разметкой,
    формирует список нарушений (например, найден телефон у человека).
    Возвращает кадр с аннотациями, треки и новые нарушения.
    """
    detections = detector.detect_objects_in_frame(frame)
    tracked_objects = detector.tracker.update(detections)

    new_violations_this_frame = []

    for obj_id, obj_info in tracked_objects.items():
        current_class_by_model = obj_info.get('class', 'unknown') 
        obj_bbox = obj_info['bbox']
        if current_class_by_model == 'phone':
            phone_confidence = obj_info.get('confidence', 0.0)
            violation_type = 'подозрительное нарушение'
            if phone_confidence >= 0.6 and phone_confidence < 0.9:
                violation_type = 'возможное использование телефона'
            elif phone_confidence >= 0.9:
                violation_type = 'использование телефона'

            phone_details = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'frame': frame_idx,
                'type': violation_type,
                'phone_tracker_id': obj_id,
                'phone_bbox': obj_bbox,
                'phone_confidence': phone_confidence,
                'associated_person_tracker_id': None,
                'associated_person_class': None,
                'associated_person_bbox': None,
                'raw_model_class_for_phone_obj': current_class_by_model 
            }
            for p_obj_id, p_obj_info in tracked_objects.items():
                if p_obj_id == obj_id:
                    continue
                p_class = p_obj_info.get('class', 'unknown')
                if p_class in ['student', 'teacher', 'person']: 
                    iou_with_person = bbox_iou(obj_bbox, p_obj_info['bbox'])
                    phone_centroid_x = (obj_bbox[0] + obj_bbox[2]) / 2
                    phone_centroid_y = (obj_bbox[1] + obj_bbox[3]) / 2
                    person_x1, person_y1, person_x2, person_y2 = p_obj_info['bbox']
                    
                    is_inside = (person_x1 <= phone_centroid_x <= person_x2 and
                                 person_y1 <= phone_centroid_y <= person_y2)

                    if iou_with_person > 0.05 or is_inside: 
                        phone_details['associated_person_tracker_id'] = p_obj_id
                        phone_details['associated_person_class'] = p_class 
                        phone_details['associated_person_bbox'] = p_obj_info['bbox']
                        break 
            new_violations_this_frame.append(phone_details)

        if tid_manual_labels_map is not None:
            if obj_id in tid_manual_labels_map:
                manual_label = tid_manual_labels_map[obj_id]
                if manual_label in ['student', 'teacher'] and current_class_by_model in ['student', 'teacher', 'person']:
                    obj_info['class'] = manual_label
            else:
                if frame_idx is not None and bbox_labels_by_video_and_frame is not None and video_prefix is not None:
                    label_from_iou, original_tid_from_label_file = match_label(frame_idx, obj_bbox, bbox_labels_by_video_and_frame, video_prefix) 
                    
                    if label_from_iou and label_from_iou in ['student', 'teacher']:
                        if current_class_by_model in ['student', 'teacher', 'person']:
                            obj_info['class'] = label_from_iou

                            tid_manual_labels_map[obj_id] = label_from_iou 
                            
                            for viol in new_violations_this_frame:
                                if viol['associated_person_tracker_id'] == obj_id:
                                    viol['associated_person_class'] = label_from_iou


    frame_with_tracking = detector.draw_tracking_info(frame.copy(), tracked_objects)
    return frame_with_tracking, tracked_objects, new_violations_this_frame


def sync_yolo_slider():
    st.session_state.extraction_yolo_conf_input = st.session_state.extraction_yolo_conf_slider

def sync_yolo_input():
    st.session_state.extraction_yolo_conf_slider = st.session_state.extraction_yolo_conf_input

def sync_max_crops_slider():
    st.session_state.extraction_max_crops_input = st.session_state.extraction_max_crops_slider

def sync_max_crops_input():
    st.session_state.extraction_max_crops_slider = st.session_state.extraction_max_crops_input

def sync_slider():
    st.session_state.step_input = st.session_state.step_slider

def sync_input():
    st.session_state.step_slider = st.session_state.step_input

def main():
    """
    Основная функция Streamlit-приложения: интерфейс загрузки видео, моделей, разметки и детекции.
    Управляет вкладками, настройками и обработкой видео.
    """
    st.title("🎥 Система обнаружения нарушений на экзамене")
    
    if 'extraction_step' not in st.session_state:
        st.session_state.extraction_step = CONFIG["STEP"]
    if 'extraction_yolo_conf' not in st.session_state:
        st.session_state.extraction_yolo_conf = CONFIG["CONF"]
    if 'extraction_max_crops' not in st.session_state:
        st.session_state.extraction_max_crops = CONFIG["MAX_CROPS_PER_ID"]
    
    if 'detection_yolo_conf' not in st.session_state:
        st.session_state.detection_yolo_conf = 0.3 
    if 'person_classification_conf' not in st.session_state:
        st.session_state.person_classification_conf = 0.5
    if 'phone_detection_conf' not in st.session_state:
        st.session_state.phone_detection_conf = 0.5

    if 'step_slider' not in st.session_state:
        st.session_state.step_slider = 15
    if 'step_input' not in st.session_state:
        st.session_state.step_input = 15
    if 'extraction_yolo_conf_slider' not in st.session_state:
        st.session_state.extraction_yolo_conf_slider = CONFIG["CONF"]
    if 'extraction_yolo_conf_input' not in st.session_state:
        st.session_state.extraction_yolo_conf_input = CONFIG["CONF"]
    if 'extraction_max_crops_slider' not in st.session_state:
        st.session_state.extraction_max_crops_slider = 120
    if 'extraction_max_crops_input' not in st.session_state:
        st.session_state.extraction_max_crops_input = 120

    tab1, tab2, tab3 = st.tabs(["📥 Загрузка и Настройки", "🏷️ Разметка", "🎯 Детекция"])
    
    with tab1:
        st.header("Загрузка видео и моделей")
        col1a, col2a = st.columns(2)
        with col1a:
            st.subheader("📹 Загрузка видео (передний план аудитории)")
            max_video_size = 1024 * 1024 * 1024  # 1 ГБ
            video_file = st.file_uploader("Выберите видеофайл (до 1 ГБ)", type=['mp4', 'avi', 'mov', 'mkv'], key='front_uploader')
            if video_file:
                if video_file.size > max_video_size:
                    st.error("❌ Размер видео превышает 1 ГБ. Пожалуйста, выберите файл меньшего размера.")
                elif not hasattr(st.session_state, 'video_path') or st.session_state.video_path is None or st.session_state.get('uploaded_video_name_front') != video_file.name:
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    tfile.write(video_file.read())
                    st.session_state.video_path = tfile.name
                    st.session_state.uploaded_video_name_front = video_file.name
                    st.session_state.extraction_complete = False 
                    st.session_state.labeling_complete = False 
                    st.success(f"✅ Видео '{video_file.name}' загружено.")
                else:
                    st.info(f"Видео '{st.session_state.uploaded_video_name_front}' уже загружено.")

            st.subheader("📹 Загрузка видео (задний план аудитории)")
            video_file_1 = st.file_uploader("Выберите видеофайл (до 1 ГБ)", type=['mp4', 'avi', 'mov', 'mkv'], key='back_uploader')
            if video_file_1:
                if video_file_1.size > max_video_size:
                    st.error("❌ Размер видео превышает 1 ГБ. Пожалуйста, выберите файл меньшего размера.")
                elif not hasattr(st.session_state, 'video_path_1') or st.session_state.video_path_1 is None or st.session_state.get('uploaded_video_name_back') != video_file_1.name:
                    tfile_1 = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    tfile_1.write(video_file_1.read())
                    st.session_state.video_path_1 = tfile_1.name
                    st.session_state.uploaded_video_name_back = video_file_1.name
                    st.session_state.extraction_complete = False
                    st.session_state.labeling_complete = False
                    st.success(f"✅ Видео '{video_file_1.name}' загружено.")
                else:
                    st.info(f"Видео '{st.session_state.uploaded_video_name_back}' уже загружено.")
        
        with col2a:
            st.subheader("🤖 Загрузка моделей (опционально)")
            person_model_file = st.file_uploader("Модель классификации (Преподаватель/Студент, YOLO .pt)", type=['pt'])
            if person_model_file:
                if not hasattr(st.session_state, 'person_model_path') or st.session_state.person_model_path is None or st.session_state.get('uploaded_person_model_name') != person_model_file.name:
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
                            tmp.write(person_model_file.read())
                            st.session_state.person_model_path = tmp.name
                            st.session_state.uploaded_person_model_name = person_model_file.name
                        st.success(f"✅ Модель классификации '{person_model_file.name}' загружена.")
                    except Exception as e:
                        st.error(f"❌ Ошибка загрузки модели классификации: {str(e)}")
                        st.session_state.person_model_path = None
                else:
                    st.info(f"Модель классификации '{st.session_state.uploaded_person_model_name}' уже загружена.")
            
            phone_model_file = st.file_uploader("Модель детекции телефонов (YOLO .pt)", type=['pt'])
            if phone_model_file:
                if not hasattr(st.session_state, 'phone_model_path') or st.session_state.phone_model_path is None or st.session_state.get('uploaded_phone_model_name') != phone_model_file.name:
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
                            tmp.write(phone_model_file.read())
                            st.session_state.phone_model_path = tmp.name
                            st.session_state.uploaded_phone_model_name = phone_model_file.name
                        st.success(f"✅ Модель детекции телефонов '{phone_model_file.name}' загружена.")
                    except Exception as e:
                        st.error(f"❌ Ошибка загрузки модели телефонов: {str(e)}")
                        st.session_state.phone_model_path = None
                else:
                    st.info(f"Модель детекции телефонов '{st.session_state.uploaded_phone_model_name}' уже загружена.")

        video_paths_to_process = []
        if hasattr(st.session_state, 'video_path') and st.session_state.video_path:
            video_paths_to_process.append(st.session_state.video_path)
        if hasattr(st.session_state, 'video_path_1') and st.session_state.video_path_1:
            video_paths_to_process.append(st.session_state.video_path_1)

        # Показываем настройки только если видео загружено
        if video_paths_to_process:
            st.subheader("⚙️ Настройки извлечения объектов из видео")
            col1, col2 = st.columns([3, 1])
            with col1:
                st.session_state.extraction_step = st.slider(
                    "Шаг между кадрами (Извлечение)",
                    1, 100,
                    key='step_slider',
                    on_change=sync_slider
                )
            with col2:
                st.session_state.extraction_step = st.number_input(
                    " ",
                    min_value=1,
                    max_value=100,
                    key='step_input',
                    on_change=sync_input
                )

            col3, col4 = st.columns([3, 1])
            with col3:
                st.session_state.extraction_yolo_conf = st.slider(
                    "YOLO Уверенность (Извлечение Людей)",
                    0.1, 0.9,
                    key='extraction_yolo_conf_slider',
                    step=0.05,
                    on_change=sync_yolo_slider
                )
            with col4:
                st.session_state.extraction_yolo_conf = st.number_input(
                    " ",
                    min_value=0.1,
                    max_value=0.9,
                    step=0.05,
                    format="%.2f",
                    key='extraction_yolo_conf_input',
                    on_change=sync_yolo_input
                )

            col5, col6 = st.columns([3, 1])
            with col5:
                st.session_state.extraction_max_crops = st.slider(
                    "Макс. кропов на ID (Извлечение)",
                    10, 500,
                    step=10,
                    key='extraction_max_crops_slider',
                    on_change=sync_max_crops_slider
                )
            with col6:
                st.session_state.extraction_max_crops = st.number_input(
                    " ",
                    min_value=10,
                    max_value=500,
                    step=10,
                    key='extraction_max_crops_input',
                    on_change=sync_max_crops_input
                )

            if st.button("Запустить разметку (для всех загруженных видео)"):
                st.session_state.extraction_complete = False
                crops_dir = "dataset_raw"
                meta_csv = 'crops_meta.csv'

                if os.path.exists(crops_dir):
                    shutil.rmtree(crops_dir)
                if os.path.exists(meta_csv):
                    os.remove(meta_csv)
                
                Path(crops_dir).mkdir(exist_ok=True, parents=True)

                all_id2cnt = {}
                all_meta = []
                
                with st.spinner("Извлечение объектов из видео... Это может занять некоторое время."):
                    for i, video_path in enumerate(video_paths_to_process):
                        st.info(f"Начинается обработка видео {i+1}/{len(video_paths_to_process)}: {os.path.basename(video_path)}")
                        tid_prefix = f"v{i+1}"
                        id2cnt, meta = extract_people_from_video(
                            video_path,
                            crops_dir,
                            tid_prefix=tid_prefix,
                            step=int(st.session_state.extraction_step),
                            yolo_conf=float(st.session_state.extraction_yolo_conf),
                            max_crops=int(st.session_state.extraction_max_crops)
                        )
                        all_id2cnt.update(id2cnt)
                        all_meta.extend(meta)
                
                if all_meta:
                    pd.DataFrame(all_meta).to_csv(meta_csv, index=False)
                    st.success(
                        f"✅ Извлечено {sum(all_id2cnt.values())} кропов для {len(all_id2cnt)} человек из {len(video_paths_to_process)} видео.")
                    st.session_state.extraction_complete = True
                    st.session_state.labeling_complete = False
                    st.info("Перейдите на вкладку 'Разметка' для продолжения.")
                else:
                    st.warning("Не удалось извлечь ни одного объекта из видео.")
    
    with tab2:
        st.header("Разметка преподавателей и студентов")
        if not hasattr(st.session_state, 'extraction_complete') or not st.session_state.extraction_complete:
            st.warning("⚠️ Сначала извлеките людей из видео на вкладке 'Загрузка и Настройки'")
        elif os.path.exists("dataset_raw"):
            df_labels_result = label_people("dataset_raw", meta_csv='crops_meta.csv', labels_file='labels.csv')
            
            if df_labels_result is not None and not df_labels_result.empty:
                st.subheader("Статистика разметки")
                st.write(df_labels_result['label'].value_counts())
        else:
            st.info("Директория 'dataset_raw' не найдена. Пожалуйста, выполните извлечение объектов.")
    
    with tab3:
        st.header("Детекция в реальном времени и Отчеты")
        
        st.subheader("⚙️ Настройки Детекции")
        col_det1, col_det2 = st.columns([3, 1])
        with col_det1:
            st.session_state.detection_yolo_conf = st.slider(
                "YOLO Уверенность (Детекция)",
                0.1, 0.9, st.session_state.detection_yolo_conf, 0.05, key='detection_yolo_conf_slider')
        with col_det2:
            st.session_state.detection_yolo_conf = st.number_input(
                " ",
                min_value=0.1,
                max_value=0.9,
                step=0.05,
                format="%.2f",
                key='detection_yolo_conf_input')

        col_det3, col_det4 = st.columns([3, 1])
        with col_det3:
            st.session_state.person_classification_conf = st.slider(
                "Уверенность Классификации Человека (Спец. модель)",
                0.1, 0.99, st.session_state.person_classification_conf, 0.05, key='person_classification_conf_slider')
        with col_det4:
            st.session_state.person_classification_conf = st.number_input(
                "  ",
                min_value=0.1,
                max_value=0.99,
                step=0.05,
                format="%.2f",
                key='person_classification_conf_input')

        col_det5, col_det6 = st.columns([3, 1])
        with col_det5:
            st.session_state.phone_detection_conf = st.slider(
                "Уверенность Детекции Телефона (Спец. модель)",
                0.1, 0.99, st.session_state.phone_detection_conf, 0.05, key='phone_detection_conf_slider')
        with col_det6:
            st.session_state.phone_detection_conf = st.number_input(
                "   ",
                min_value=0.1,
                max_value=0.99,
                step=0.05,
                format="%.2f",
                key='phone_detection_conf_input')

        video_paths_to_detect = []
        if hasattr(st.session_state, 'video_path') and st.session_state.video_path:
            video_paths_to_detect.append(st.session_state.video_path)
        if hasattr(st.session_state, 'video_path_1') and st.session_state.video_path_1:
            video_paths_to_detect.append(st.session_state.video_path_1)

        if not video_paths_to_detect:
            st.warning("⚠️ Сначала загрузите видео на вкладке 'Загрузка и Настройки'")
        else:
            default_phone_model_path = "yolo_training/phone_detection5/weights/best.pt"
            default_person_classifier_path = "yolo_cls_training/person_classifier5/weights/best.pt"
        
            phone_model_to_use = getattr(st.session_state, 'phone_model_path', default_phone_model_path)
            person_model_to_use = getattr(st.session_state, 'person_model_path', default_person_classifier_path)

            detector = initialize_detector(
                person_model_to_use,
                phone_model_to_use,
                yolo_conf_det=st.session_state.detection_yolo_conf,
                person_class_conf=st.session_state.person_classification_conf,
                phone_det_conf=st.session_state.phone_detection_conf
            )

            if detector is None:
                st.error("❌ Не удалось инициализировать детектор. Проверьте модели и настройки.")
            else:
                col1b, col2b = st.columns(2)
                with col1b:
                    start_button = st.button("▶️ Начать детекцию")
                with col2b:
                    stop_button = st.button("⏹️ Остановить")

                placeholders = []
                if len(video_paths_to_detect) == 1:
                    placeholders.append(st.empty())
                else:
                    cols = st.columns(len(video_paths_to_detect))
                    for col in cols:
                        placeholders.append(col.empty())

                if start_button:
                    st.session_state.processing = True
                    st.session_state.violations_log = []
                
                    detectors = [copy.deepcopy(detector) for _ in video_paths_to_detect]
                    st.session_state.tid_manual_labels_maps = [{} for _ in video_paths_to_detect]

                    caps = [cv2.VideoCapture(os.path.abspath(p)) for p in video_paths_to_detect]
                    outs = []
                    output_paths = []
                    
                    for i, cap in enumerate(caps):
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        output_path = f"annotated_video_{i+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                        outs.append(cv2.VideoWriter(output_path, fourcc, fps, (width, height)))
                        output_paths.append(output_path)
                    
                    st.session_state.annotated_video_paths = output_paths

                    if not all(c.isOpened() for c in caps):
                        st.error("Не удалось открыть один или несколько видеофайлов для детекции.")
                        st.session_state.processing = False
                    else:
                        total_frames = [int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps]
                        progress_bars = [st.progress(0) for _ in caps]
                        status_texts = [st.empty() for _ in caps]
                        bbox_labels_by_video_and_frame = load_bbox_labels('labels.csv')
                        frame_indices = [0] * len(caps)

                        while st.session_state.get('processing', False):
                            
                            active_streams = 0
                            for i, cap in enumerate(caps):
                                if not cap.isOpened():
                                    continue

                                ret, frame = cap.read()
                                if not ret:
                                    cap.release()
                                    continue
                                
                                active_streams += 1
                                
                                video_prefix = f"v{i+1}" 
                                
                                frame_with_tracking, _, new_violations = process_video_frame(
                                    frame,
                                    detectors[i], 
                                    frame_idx=frame_indices[i],
                                    bbox_labels_by_video_and_frame=bbox_labels_by_video_and_frame,
                                    video_prefix=video_prefix, 
                                    tid_manual_labels_map=st.session_state.tid_manual_labels_maps[i] 
                                )
                                outs[i].write(frame_with_tracking)
                                st.session_state.violations_log.extend(new_violations)

                                frame_rgb = cv2.cvtColor(frame_with_tracking, cv2.COLOR_BGR2RGB)
                                placeholders[i].image(frame_rgb)
                                
                                frame_indices[i] += 1
                                if total_frames[i] > 0:
                                    progress_bars[i].progress(frame_indices[i] / total_frames[i])
                                    status_texts[i].text(f"Видео {i+1}: кадр {frame_indices[i]}/{total_frames[i]}")
                                else:
                                    status_texts[i].text(f"Видео {i+1}: кадр {frame_indices[i]}")

                            if active_streams == 0:
                                st.session_state.processing = False
                                st.info("Все видео потоки завершены.")
                                break

                        for cap in caps:
                            if cap.isOpened(): cap.release()
                        for out in outs: out.release()
                        st.session_state.processing = False
                        for i in range(len(caps)):
                            status_texts[i].text(f"Обработка видео {i+1} завершена!")

                if stop_button:
                    st.session_state.processing = False
                    st.info("Обработка остановлена пользователем.")

                if not st.session_state.get('processing', True):
                    if hasattr(st.session_state, 'violations_log') and st.session_state.violations_log:
                        st.success(f"Детекция завершена. Зафиксировано {len(st.session_state.violations_log)} потенциальных нарушений.")
                        df_violations = pd.DataFrame(st.session_state.violations_log)
                        txt_data = convert_df_to_html_table(df_violations)
                        st.download_button(
                            label="📥 Скачать отчет о нарушениях (HTML)",
                            data=txt_data,
                            file_name=f"violations_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                            mime="text/html",
                        )
                        st.subheader("Список зафиксированных нарушений:")
                        st.markdown(txt_data.decode('utf-8'), unsafe_allow_html=True)
                    else:
                        st.info("Нарушений не зафиксировано или обработка была прервана до их обнаружения.")

                    if hasattr(st.session_state, 'annotated_video_paths'):
                        st.subheader("Просмотр размеченных видео")
                        
                        paths = st.session_state.annotated_video_paths
                        if paths:
                            cols = st.columns(len(paths))

                            for i, path in enumerate(paths):
                                if os.path.exists(path):
                                    with cols[i]:
                                        st.text(os.path.basename(path))
                                        video_file = open(path, 'rb')
                                        video_bytes = video_file.read()
                                        st.video(video_bytes)
                            
if __name__ == "__main__":
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'extraction_complete' not in st.session_state:
        st.session_state.extraction_complete = False
    if 'labeling_complete' not in st.session_state:
        st.session_state.labeling_complete = False
    if 'tid_manual_labels_maps' not in st.session_state: 
        st.session_state.tid_manual_labels_maps = []
    if 'violations_log' not in st.session_state:
        st.session_state.violations_log = []
    if 'annotated_video_paths' not in st.session_state:
        st.session_state.annotated_video_paths = []
    
    if 'video_path' not in st.session_state:
        st.session_state.video_path = None
        st.session_state.uploaded_video_name_front = None
    if 'video_path_1' not in st.session_state:
        st.session_state.video_path_1 = None
        st.session_state.uploaded_video_name_back = None
    if 'person_model_path' not in st.session_state:
        st.session_state.person_model_path = None
        st.session_state.uploaded_person_model_name = None
    if 'phone_model_path' not in st.session_state: 
        st.session_state.phone_model_path = None
        st.session_state.uploaded_phone_model_name = None

    if 'person_model_path_loaded' not in st.session_state: st.session_state.person_model_path_loaded = None
    if 'phone_model_path_loaded' not in st.session_state: st.session_state.phone_model_path_loaded = None
    if 'yolo_conf_det_loaded' not in st.session_state: st.session_state.yolo_conf_det_loaded = None
    if 'person_class_conf_loaded' not in st.session_state: st.session_state.person_class_conf_loaded = None
    if 'phone_det_conf_loaded' not in st.session_state: st.session_state.phone_det_conf_loaded = None


    main() 