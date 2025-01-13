from ultralytics import YOLO
from PIL import Image

class Predict:
    model = YOLO('image_classification_yolo/runs/classify/train5/weights/last.pt')

    def predict_top5_results(self, img):
        results = self.model.predict(img)

        dict_result = dict()

        for result in results:
            top5index = result.probs.top5
            top5conf = result.probs.top5conf
            for i in range(5):
                prob = round(top5conf[i].item() * 100, 2)
                if prob >= 0.01:
                    dict_result[result.names[top5index[i]]] = prob

        return dict_result



    def predict_from_path(self, img_path):
        model = YOLO('./runs/classify/train5/weights/last.pt')

        results = model(self,img_path)
        print('----------------')
        for result in results:
            top5index = result.probs.top5
            top5conf = result.probs.top5conf
            for i in range(5):
                print("Name: \t\t\t"+ result.names[top5index[i]] + "\n" +
                      "Probability: \t" + str(top5conf[i].item()))


        print('----------------')

    @staticmethod
    def print_results(results):
        for key in results:
            print(f"Name: {key}\n"
                  f"Probability: {results[key]}")


    if __name__ == '__main__':
        x = input("Input image number: ")
        # predict(f'./test_images/{x}.jpg')
        img = Image.open(f'./test_images/{x}.jpg')
        res = predict_top5_results(img)
        print(res)