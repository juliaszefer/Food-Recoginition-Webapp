import os
import cv2
from ultralytics import YOLO


def find_faces_with_yolo(base_folder, output_folder, confidence_threshold=0.8):
    # Ładowanie modelu YOLO
    model = YOLO("yolov11n-face.pt")  # Upewnij się, że używasz odpowiedniego modelu twarzy

    # Lista do przechowywania ścieżek do zdjęć z wykrytymi twarzami
    image_paths_with_faces = []

    # Tworzenie folderu na zapis wyników, jeśli nie istnieje
    os.makedirs(output_folder, exist_ok=True)

    # Przechodzenie przez wszystkie pliki w folderze i podfolderach
    for root, dirs, files in os.walk(base_folder):
        for file in files:
            # Sprawdzanie, czy plik jest obrazem
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                file_path = os.path.join(root, file)
                try:
                    # Wczytanie obrazu
                    img = cv2.imread(file_path)

                    # Wykrywanie twarzy za pomocą YOLO z progiem pewności
                    results = model(img, conf=confidence_threshold)

                    # Filtrowanie wyników dla klasy "twarz"
                    face_detected = False
                    for result in results:
                        for box in result.boxes:
                            class_id = int(box.cls)  # Pobranie ID klasy
                            confidence = box.conf  # Pobranie pewności detekcji
                            if class_id == 0 and confidence >= confidence_threshold:
                                face_detected = True
                                break  # Wykryto twarz, nie trzeba przetwarzać dalej

                    # Jeśli wykryto twarze, zapisz obraz i ścieżkę
                    if face_detected:
                        image_paths_with_faces.append(file_path)

                        # Zapisz obraz z detekcjami do folderu wynikowego
                        output_path = os.path.join(output_folder, os.path.basename(file_path))
                        annotated_frame = results[0].plot()  # Narysuj wykryte twarze
                        cv2.imwrite(output_path, annotated_frame)

                except Exception as e:
                    print(f"Błąd podczas przetwarzania {file_path}: {e}")

    return image_paths_with_faces


# Ścieżka do głównego folderu z obrazami
base_folder = 'images'

# Ścieżka do folderu na zapisane obrazy z wykrytymi twarzami
output_folder = 'output_with_faces'

# Znajdowanie obrazów z twarzami
confidence_threshold = 0.8  # Ustaw próg pewności na 80%
images_with_faces = find_faces_with_yolo(base_folder, output_folder, confidence_threshold)

# Wypisywanie ścieżek do znalezionych obrazów
if images_with_faces:
    print("Zdjęcia z twarzami:")
    for path in images_with_faces:
        print(path)
else:
    print("Nie znaleziono zdjęć z twarzami.")
