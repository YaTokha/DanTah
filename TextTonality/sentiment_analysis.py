import os
import nltk
from textblob import TextBlob
from typing import Tuple

class SentimentAnalyzer:
    def __init__(self):
        """Инициализация анализатора тональности."""
        # Загрузка необходимых данных для NLTK
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
            sentiment = "позитивный"
        elif polarity < 0:
            sentiment = "негативный"
        else:
            sentiment = "нейтральный"

        return sentiment, polarity

    def save_text_to_file(self, text: str, sentiment: str, polarity: float, output_dir: str):
        """
        Сохраняет результат анализа в текстовый файл.
        """
        os.makedirs(output_dir, exist_ok=True)  # Создание папки, если её нет
        file_name = "result.txt"
        file_path = os.path.join(output_dir, file_name)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"Исходный текст: {text}\n")
            f.write(f"Тональность: {sentiment}\n")
            f.write(f"Полярность: {polarity:.2f}\n")
        print(f"Результат сохранён в файл: {file_path}")


# Пример использования
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()

    # Ввод текста из консоли
    print("Введите текст для анализа тональности:")
    input_text = input("> ").strip()

    try:
        # Анализ тональности
        sentiment, polarity = analyzer.analyze_sentiment(input_text)
        print(f"Тональность: {sentiment}, Полярность: {polarity:.2f}")

        # Сохранение результата в папку result
        analyzer.save_text_to_file(input_text, sentiment, polarity, output_dir="result")
    except ValueError as e:
        print(f"Ошибка: {e}")
