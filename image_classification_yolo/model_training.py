import os
import shutil
import random

import torch.cuda
import torchvision
from ultralytics import YOLO

def train_model(data_path):
    model = YOLO('yolo11n-cls.pt')

    # model.train(
    #     data=data_path,
    #     epochs=300,
    #     batch=48,
    #     imgsz=640,
    #     patience=5,
    #     name='test_run'
    # )
    model.train(
        data=data_path,
        epochs=100,  # Liczba epok treningu
        imgsz=640,  # Rozmiar obrazu wejściowego
        batch=16,  # Rozmiar batcha
        device=0,  # Użycie GPU (0) lub CPU ('cpu')
        workers=8,  # Liczba wątków do ładowania danych
        optimizer='SGD',  # Optymalizator
        lr0=0.01,  # Początkowa wartość learning rate
        momentum=0.937,  # Momentum
        weight_decay=0.0005,  # Współczynnik dekompozycji wag
        patience=50,  # Liczba epok bez poprawy przed wczesnym zatrzymaniem
        cos_lr=True,  # Użycie kosinusowego scheduler'a learning rate
        cache='ram',  # Cache'owanie danych w pamięci RAM
        pretrained=True
    )


def split_dataset(base_dir="data/images", train_ratio=0.8):
    random.seed(42)  # Zapewnia powtarzalność wyników

    # Ścieżki do folderów docelowych
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")

    # Tworzenie folderów docelowych jeśli nie istnieją
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Pobranie wszystkich folderów (foodtype)
    for foodtype in os.listdir(base_dir):
        foodtype_path = os.path.join(base_dir, foodtype)
        if not os.path.isdir(foodtype_path):
            continue  # Pomijamy pliki

        # Ścieżki do folderów treningowych i walidacyjnych dla danej klasy
        train_foodtype_path = os.path.join(train_dir, foodtype)
        val_foodtype_path = os.path.join(val_dir, foodtype)

        os.makedirs(train_foodtype_path, exist_ok=True)
        os.makedirs(val_foodtype_path, exist_ok=True)

        # Pobranie listy plików w danym katalogu
        images = [f for f in os.listdir(foodtype_path) if os.path.isfile(os.path.join(foodtype_path, f))]
        random.shuffle(images)  # Losowe przemieszanie

        # Podział zbioru
        split_index = int(len(images) * train_ratio)
        train_images = images[:split_index]
        val_images = images[split_index:]

        # Przenoszenie plików do odpowiednich folderów
        for img in train_images:
            shutil.move(os.path.join(foodtype_path, img), os.path.join(train_foodtype_path, img))

        for img in val_images:
            shutil.move(os.path.join(foodtype_path, img), os.path.join(val_foodtype_path, img))

        print(f"Podział dla {foodtype}: {len(train_images)} train, {len(val_images)} val")

    print("Podział zakończony!")

if __name__ == '__main__':
    train_model('C:\\Users\\jerem\\Desktop\\data\\images')
    print('Training complete!')
    # split_dataset('C:\\Users\\jerem\\Desktop\\data\\images')
    # print(torch.cuda.is_available())
    # print(torch.cuda.device_count())
    # print(torchvision.ops.nms)