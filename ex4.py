import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten
from sklearn.model_selection import train_test_split

# Загрузка обучающей и тестовой выборки
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Нормользация выходных данный

x_train = x_train / 255
x_test = x_test / 255

# Преоброзование выходныйх значений в веторы по категориям
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)



size_val = 10000                        # Размер выборки валидаии
x_val_split = x_train[:size_val]        # выделения первого наблюдения из обучающей выборки
y_val_split = y_train_cat[:size_val]    # в выборку валидации

x_train_split = x_train[size_val:]      # выделить последующие наблюдения для обучающей выборки
y_train_split = y_train_cat[size_val:]  


# Формирование модели НС и вывод ее структуры в консоль
model = keras.Sequential([
    Flatten(input_shape=(28, 28, 1),),
    Dense(128, activation='relu'),
    Dense(10, activation="softmax")
])

# виды оптимизаторов
myAdam = keras.optimizers.Adam(learning_rate=0.1)
myOpt = keras.optimizers.SGD(learning_rate=0.1, momentum=0.0, nesterov=True)

model.compile(optimizer=myAdam,
            loss='categorical_crossentropy',
            metrics=['accuracy']
            )


x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(x_train, y_train_cat, test_size=0.2)

# выборка валидации
model.fit(x_train_split, y_train_split, batch_size=32, epochs=5, 
            validation_data=(x_val_split, y_val_split))


print('Проверка')
model.evaluate(x_test, y_test_cat)