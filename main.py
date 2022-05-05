from imageClassifier import predictClass

def main():
    img_path = './data/images/1.jpg'
    ans_class = predictClass(img_path)
    print("prediction = ", ans_class)

if __name__ == "__main__":
    main()