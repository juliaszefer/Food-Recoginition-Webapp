# 🍽️ Food Recognition Project

## 📌 Overview
This project is an intelligent application designed to recognize various food products and dishes based on images. Using machine learning algorithms, the system analyzes photos and classifies them into appropriate food categories.

The dataset consists of **101 subsets**, each containing **1,000 images (.jpg)** representing a specific food item or dish. The images are stored in **Microsoft OneDrive** and retrieved for machine learning training.  

Since many images contained **human faces**, an additional preprocessing model was developed to detect and filter out faces, ensuring better accuracy in food recognition.  

## 🚀 How to Run the Project
1. **Clone the repository**  
   ```bash
   git clone <repository_url>

2. **Navigate to the project directory**  
   ```bash
   cd image_classification_yolo

3. **Run streamlit app
  ```bash
  streamlit run food_recognition_app.py

## 📷 Example Usage
Once the application is running, upload an image of food, and the model will classify it into one of the predefined categories.

## 🛠 Technologies Used
- **Python**
- **Machine Learning** (YOLO for image classification)
- **OpenCV**
- **Streamlit** (for UI)
- **Microsoft OneDrive** (for dataset storage)

datasource:
https://www.kaggle.com/datasets/kmader/food41/data

inspiration:
https://github.com/Gogul09/image-classification-python/blob/master/train_test.py
