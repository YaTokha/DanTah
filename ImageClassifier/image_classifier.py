import os
import torch
from torchvision import models, transforms
from PIL import Image
from typing import Tuple


class ImageClassifier:
    def __init__(self, model_name: str = "resnet18", device: str = "cpu"):
        """
        Инициализация классификатора изображений.
        """
        self.device = torch.device(device)
        self.model = self._load_model(model_name)
        self.labels = self._load_labels()

    def _load_model(self, model_name: str):
        """
        Загружает предобученную модель.
        """
        model = getattr(models, model_name)(pretrained=True)
        model = model.to(self.device)
        model.eval()  # Перевод в режим оценки
        return model

    def _load_labels(self) -> list:
        """
        Загружает метки классов из ImageNet.
        """
        labels_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
        labels_path = "imagenet_labels.json"
        if not os.path.exists(labels_path):
            import urllib.request
            urllib.request.urlretrieve(labels_url, labels_path)

        import json
        with open(labels_path, "r") as f:
            return json.load(f)

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Предобрабатывает изображение для классификации.
        """
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        image = Image.open(image_path).convert("RGB")
        return transform(image).unsqueeze(0).to(self.device)

    def classify_image(self, image_path: str) -> Tuple[str, float]:
        """
        Классифицирует изображение.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Файл {image_path} не найден.")

        # Предобработка изображения
        image_tensor = self.preprocess_image(image_path)

        # Классификация
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            top_idx = probabilities.argmax().item()
            class_name = self.labels[top_idx]
            confidence = probabilities[top_idx].item()

        return class_name, confidence

    def classify_images_from_folder(self, input_folder: str, output_folder: str):
        """
        Классифицирует все изображения в папке и сохраняет результаты.
        """
        os.makedirs(output_folder, exist_ok=True)

        for filename in os.listdir(input_folder):
            input_path = os.path.join(input_folder, filename)
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                print(f"Пропущен неподдерживаемый файл: {filename}")
                continue

            try:
                class_name, confidence = self.classify_image(input_path)
                output_file = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_result.txt")
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(f"Файл: {filename}\n")
                    f.write(f"Класс: {class_name}\n")
                    f.write(f"Вероятность: {confidence:.2f}\n")
                print(f"Результат сохранён для файла {filename} в {output_file}")
            except Exception as e:
                print(f"Ошибка при обработке {filename}: {e}")

    def _transform_input(self):
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


# Пример использования
if __name__ == "__main__":
    classifier = ImageClassifier(model_name="resnet18", device="cpu")

    # Папки для ввода и вывода
    input_folder = "data"
    output_folder = "result"

    classifier.classify_images_from_folder(input_folder, output_folder)
import os
import torch
from torchvision import models, transforms
from PIL import Image
from typing import Tuple


class ImageClassifier:
    def __init__(self, model_name: str = "resnet18", device: str = "cpu"):
        """
        Инициализация классификатора изображений.
        """
        self.device = torch.device(device)
        self.model = self._load_model(model_name)
        self.labels = self._load_labels()

    def _load_model(self, model_name: str):
        """
        Загружает предобученную модель.
        """
        model = getattr(models, model_name)(pretrained=True)
        model = model.to(self.device)
        model.eval()  # Перевод в режим оценки
        return model

    def _load_labels(self) -> list:
        """
        Загружает метки классов из ImageNet.
        """
        labels_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
        labels_path = "imagenet_labels.json"
        if not os.path.exists(labels_path):
            import urllib.request
            urllib.request.urlretrieve(labels_url, labels_path)

        import json
        with open(labels_path, "r") as f:
            return json.load(f)

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Предобрабатывает изображение для классификации.
        """
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        image = Image.open(image_path).convert("RGB")
        return transform(image).unsqueeze(0).to(self.device)

    def classify_image(self, image_path: str) -> Tuple[str, float]:
        """
        Классифицирует изображение.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Файл {image_path} не найден.")

        # Предобработка изображения
        image_tensor = self.preprocess_image(image_path)

        # Классификация
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            top_idx = probabilities.argmax().item()
            class_name = self.labels[top_idx]
            confidence = probabilities[top_idx].item()

        return class_name, confidence

    def classify_images_from_folder(self, input_folder: str, output_folder: str):
        """
        Классифицирует все изображения в папке и сохраняет результаты.
        """
        os.makedirs(output_folder, exist_ok=True)

        for filename in os.listdir(input_folder):
            input_path = os.path.join(input_folder, filename)
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                print(f"Пропущен неподдерживаемый файл: {filename}")
                continue

            try:
                class_name, confidence = self.classify_image(input_path)
                output_file = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_result.txt")
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(f"Файл: {filename}\n")
                    f.write(f"Класс: {class_name}\n")
                    f.write(f"Вероятность: {confidence:.2f}\n")
                print(f"Результат сохранён для файла {filename} в {output_file}")
            except Exception as e:
                print(f"Ошибка при обработке {filename}: {e}")


# Пример использования
if __name__ == "__main__":
    classifier = ImageClassifier(model_name="resnet18", device="cpu")

    # Папки для ввода и вывода
    input_folder = "data"
    output_folder = "result"

    classifier.classify_images_from_folder(input_folder, output_folder)
