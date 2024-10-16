from ultralytics import YOLO

def predict(img_path):
    model = YOLO('./runs/classify/train5/weights/last.pt')

    results = model(img_path)
    print('----------------')
    for result in results:
        top5index = result.probs.top5
        top5conf = result.probs.top5conf
        for i in range(5):
            print("Name: \t\t\t"+ result.names[top5index[i]] + "\n" +
                  "Probability: \t" + str(top5conf[i].item()))


    print('----------------')
def print_results(results):
    print()
if __name__ == '__main__':
    x = input("Input image number: ")
    predict(f'./test_images/{x}.jpg')