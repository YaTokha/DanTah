
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from ImageClassifier.image_classifier import ImageClassifier
from Speech.transcriber import Transcriber
from TextTonality import sentiment_analysis
from io import BytesIO
from PIL import Image
import torch
import tempfile

app = FastAPI()

image_model = ImageClassifier()
transcriber = Transcriber(model_name="base")

class TextInput(BaseModel):
    text: str

@app.post("/classify-image/")
async def classify_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        transform = image_model._transform_input()
        input_tensor = transform(image).unsqueeze(0).to(image_model.device)
        with torch.no_grad():
            output = image_model.model(input_tensor)
        predicted_class = output.argmax(dim=1).item()
        label = image_model.labels[predicted_class]
        return {"label": label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename[-4:]) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        result = transcriber.transcribe_audio(tmp_path)
        return {"transcription": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-tonality/")
async def analyze_tonality(input_text: TextInput):
    try:
        # Пример вызова, нужно адаптировать под ваш sentiment_analysis
        # Предположим, sentiment_analysis.analyze(input_text.text) возвращает {"sentiment": "..."}
        result = sentiment_analysis.analyze(input_text.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
