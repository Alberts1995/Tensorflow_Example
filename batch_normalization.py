import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split

# Загрузка обучающей и тестовой выборки
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Нормользация выходных данный

x_train = x_train / 255
x_test = x_test / 255

# Преоброзование выходныйх значений в веторы по категориям
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)



limit = 5000                          # Размер выборки валидаии
x_train_data = x_train[:limit]        # выделения первого наблюдения из обучающей выборки
y_train_data = y_train_cat[:limit]    # в выборку валидации

x_valid = x_train[limit:limit*2]      # выделить последующие наблюдения для обучающей выборки
y_valid = y_train_cat[limit:limit*2]  



# Формирование модели НС и вывод ее структуры в консоль
model = keras.Sequential([
    Flatten(input_shape=(28, 28, 1),),
    Dense(300, activation='relu'),
    BatchNormalization(),                       
    Dense(10, activation="softmax")
])


model.compile(optimizer="adam",
            loss='categorical_crossentropy',
            metrics=['accuracy']
            )


# выборка валидации
his = model.fit(x_train_data, y_train_data, batch_size=32, epochs=50, 
            validation_data=(x_valid, y_valid))

plt.plot(his.history['loss'])
plt.plot(his.history['val_loss'])
plt.show()