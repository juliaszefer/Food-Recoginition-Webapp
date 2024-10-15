from ultralytics import YOLO

def train_model():
    model = YOLO('yolo11n-cls.pt')

    model.train(
        data='C:\\Users\\jerem\\Desktop\\classification data\\images',
        epochs=2,
        batch=64,
        imgsz=640,
        device=0,
        patience=5
    )

if __name__ == '__main__':
    train_model()