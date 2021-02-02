import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from google.colab import files
from io import BytesIO
from PIL import Image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"


model = keras.applications.VGG16()


upload = files.upload()

img=Image.open(BytesIO(upload['1.jpg']))
plt.imshow(img)


img = np.array(img)
x = keras.applications.vgg16.preprocess_input(img)
print(x.shape)
x = np.expand_dims(x ,axis=0)
print(x.shape)


res = model.predict(x)
print(np.argmax(res))