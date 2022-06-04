import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from PIL import Image
import keras



PATH_TO_MODEL="Utils/model-3x3.h5"
PATH_TO_IMAGE="uploads/test_docs/children.jpg"
def get_predictions(path_to_model,path_to_image):
    reconstructed_model = keras.models.load_model(path_to_model)
    im=preprocess_image(path_to_image)
    assert im.shape==(32,32,3)
    im = np.expand_dims(im, axis=0)
    print("final shape is,",im.shape)
    scores = reconstructed_model.predict(im)
    print(scores[0].shape)

    prediction = np.argmax(scores)
    print('ClassId:', prediction)

    def label_text(file):
        label_list = []
        r = pd.read_csv(file)
        for name in r['SignName']:
            label_list.append(name)
        return label_list

    labels = label_text('Utils/label_names.csv')

    print('Label:', labels[prediction])

def preprocess_image(path_to_image):
    WIDTH = 32
    HEIGHT = 32
    # Image.open() can also open other image types
    img = Image.open(path_to_image)
    im = np.array(img)
    print("initial size is ,", im.shape)
    # WIDTH and HEIGHT are integers
    resized_img = img.resize((WIDTH, HEIGHT))
    res_im = np.array(resized_img)
    resized_img.save("uploads/test_docs/resized_image1.png")
    print("initial size is ,", res_im.shape)
    rgb_image = resized_img.convert('RGB')
    final=np.array(rgb_image)
    print("rgb  size is ,", final.shape)
    return final


get_predictions(PATH_TO_MODEL,PATH_TO_IMAGE)