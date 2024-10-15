from ultralytics import YOLO

def train_model(data_path):
    model = YOLO('yolo11n-cls.pt')

    model.train(
        data=data_path,
        epochs=5,
        batch=48,
        imgsz=640,
        device=0,
        patience=5,
        name='test_run'
    )

if __name__ == '__main__':
    train_model('C:\\Users\\jerem\\Desktop\\classification data\\smaller_images')