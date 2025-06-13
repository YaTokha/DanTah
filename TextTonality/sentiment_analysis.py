import os
import nltk
from textblob import TextBlob
from typing import Tuple

class SentimentAnalyzer:
    def __init__(self):
        """Инициализация анализатора тональности."""
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")

    def analyze_sentiment(self, text: str) -> Tuple[str, float]:
        """
        Определяет тональность текста.
        """
        if not text.strip():
            raise ValueError("Текст не может быть пустым.")

        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity

        if polarity > 0:
            sentiment = "positive"
        elif polarity < 0:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return sentiment, polarity

    def save_text_to_file(self, text: str, sentiment: str, polarity: float, output_dir: str):
        """
        Сохраняет результат анализа в текстовый файл.
        """
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, "result.txt")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"Input text: {text}\n")
            f.write(f"Sentiment: {sentiment}\n")
            f.write(f"Polarity: {polarity:.2f}\n")
        print(f"Результат сохранён в файл: {file_path}")


analyzer = SentimentAnalyzer()

def analyze(text: str) -> dict:
    sentiment, polarity = analyzer.analyze_sentiment(text)
    return {"sentiment": sentiment, "polarity": polarity}

if __name__ == "__main__":
    print("Введите текст для анализа тональности:")
    input_text = input("> ").strip()
    try:
        sentiment, polarity = analyzer.analyze_sentiment(input_text)
        print(f"Тональность: {sentiment}, Полярность: {polarity:.2f}")
        analyzer.save_text_to_file(input_text, sentiment, polarity, output_dir="result")
    except ValueError as e:
        print(f"Ошибка: {e}")
