from image_classification_yolo.predict import Predict
from PIL import Image

prediction = Predict()

x = input("Input image number: ")
# predict(f'./test_images/{x}.jpg')
img = Image.open(f'./test_images/{x}.jpg')
res = prediction.predict(img)
Predict.print_results(res)