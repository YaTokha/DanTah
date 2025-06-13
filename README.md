
# 📦 FastAPI ML API

Это веб-приложение на FastAPI, предоставляющее REST API для трёх моделей машинного обучения:

- 🔍 Классификация изображений (ResNet)
- 🎙 Распознавание речи (Speech-to-Text)
- ✍️ Анализ тональности текста

---

## 📁 Структура проекта

```
FastAPI_App/
├── main.py                    
├── requirements.txt           
├── README.md                  

├── ImageClassifier/
│   ├── image_classifier.py    
│   ├── imagenet_labels.json   
│   └── data/                  

├── Speech/
│   ├── config.py
│   ├── run.py
│   ├── app/
│   │   ├── main.py
│   │   ├── utils.py
│   │   └── models/
│   │       ├── preprocess.py
│   │       └── transcriber.py
│   ├── data/                 
│   └── processed/            

├── TextTonality/
│   ├── sentiment_analysis.py
│   └── result/               
```

---

## 🚀 Установка и запуск

> Требуется Python 3.8+

1. Установи зависимости:

```bash
pip install -r requirements.txt
```

2. Запусти FastAPI-сервер:

```bash
uvicorn main:app --reload
```

3. Перейти в браузере:

📍 http://127.0.0.1:8000/docs — интерактивная Swagger-документация  
📍 http://127.0.0.1:8000/redoc — альтернативная документация

---

## 🔁 API-маршруты

### 📷 POST `/classify-image/`
Классифицирует изображение (формат PNG/JPG)

**Параметры:**
- `file`: изображение

**Пример ответа:**
```json
{
  "label": "panda"
}
```

---

### 🔊 POST `/transcribe/`
Распознаёт речь из аудиофайла (формат .mp3 или .wav)

**Параметры:**
- `file`: аудиофайл

**Пример ответа:**
```json
{
  "transcription": "Привет, как дела?"
}
```

---

### 🧠 POST `/analyze-tonality/`
Определяет тональность текста (положительная, нейтральная, отрицательная)

**Параметры:**
- `text`: текст для анализа

**Пример тела запроса:**
```json
{
  "text": "Мне очень понравился этот проект!"
}
```

**Пример ответа:**
```json
{
  "sentiment": "positive"
}
```

---

## ⚠️ Возможные ошибки

- `500 Internal Server Error` — проверь, что все зависимости установлены, и входные данные корректны
- Ошибка с `python-multipart` — установи:
  ```bash
  pip install python-multipart
  ```

---

## ✅ Советы

- Для продакшн-сервера:
  ```bash
  uvicorn main:app --host 0.0.0.0 --port 80
  ```

- Деплой: можно использовать Docker, Heroku, Render, Railway
