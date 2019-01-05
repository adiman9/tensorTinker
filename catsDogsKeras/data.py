import os
import random
import pickle
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import cv2
from dotenv import load_dotenv
load_dotenv()

DATA_DIR = os.getenv('dataset_dir')
CATEGORIES = ['Cat', 'Dog']
IMG_SIZE = 50

def show_image(img):
    image = np.array(img).reshape(IMG_SIZE, IMG_SIZE)
    plt.imshow(image, cmap = 'gray')
    plt.show()

def create_training_data(save):
    training_data = []
    data_balance = defaultdict(int)
    for label, category in enumerate(CATEGORIES):
        path = os.path.join(DATA_DIR, category)

        for img in os.listdir(path):
            try:
                image_data = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_image = cv2.resize(image_data, (IMG_SIZE, IMG_SIZE))
                show_image(resized_image)
                training_data.append((resized_image, label))
                data_balance[category] += 1
            except Exception as e:
                pass

    random.shuffle(training_data)
    print(data_balance)
    print('There are {} training samples'.format(len(training_data)))

    x_train = []
    y_train = []

    for features, label in training_data:
        x_train.append(features)
        y_train.append(label)

    # 50 x 50px each 1 grayscale value (50 x 50 x 1 tensor)
    x_train = np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    if save:
        with open('x_train.pickle', 'wb') as f:
            pickle.dump(x_train, f)

        with open('y_train.pickle', 'wb') as f:
            pickle.dump(y_train, f)

    return x_train, y_train

def get_training_data():
    with open('x_train.pickle', 'rb') as f:
        x_train = pickle.load(f)

    with open('y_train.pickle', 'rb') as f:
        y_train = pickle.load(f)

    return x_train, y_train

# generate data set from images and save
# x_train, y_train = create_training_data(False)

# load data set from saved pickles
# x_train, y_train = get_training_data()

# show_image(random.choice(x_train))
