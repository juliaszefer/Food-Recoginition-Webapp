from ultralytics import YOLO

def predict(img_path):
    model = YOLO('./runs/classify/train5/weights/last.pt')

    results = model.predict(img_path, show=True)
    input("waiting for input...")
    print (results)

if __name__ == '__main__':
    predict('./test_images/03.jpg')