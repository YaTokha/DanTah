from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_classify_image():
    with open("ImageClassifier/data/img_3.png", "rb") as img:
        response = client.post("/classify-image/", files={"file": ("img_3.png", img, "image/png")})
    assert response.status_code == 200
    assert "label" in response.json()

def test_transcribe_audio():
    with open("Speech/data/voice_09-12-2024_00-52-11.mp3", "rb") as audio:
        response = client.post("/transcribe/", files={"file": ("audio.mp3", audio, "audio/mpeg")})
    assert response.status_code == 200
    assert "transcription" in response.json()

def test_analyze_tonality():
    response = client.post("/analyze-tonality/", json={"text": "Мы вас любим, Диана Маратовна!!!"})
    assert response.status_code == 200
    assert response.json()["sentiment"] in ["positive", "neutral", "negative"]
