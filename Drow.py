import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

# Загрузка обучающей и тестовой выборки
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Нормользация выходных данный

x_train = x_train / 255
x_test = x_test / 255

# Преоброзование выходныйх значений в веторы по категориям
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

# Отоброжение первых 25 изоброжений из обучающей выборки
plt.figure(figsize=(10, 5))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i], cmap=plt.cm.binary)

plt.show()
# Формирование модели НС и вывод ее структуры в консоль
model = keras.Sequential([
    Flatten(input_shape=(28, 28, 1),),
    Dense(128, activation='relu'),
    Dense(10, activation="softmax")
])

print(model.summary())


# Оптимизаторы
# myAdam = keras.optimizers.Adam(learning_rate=0.1)
# myOpt = keras.optimizers.SGD(learning_rate=0.1, momentum=0.0, nesterov=True)

# Компиляция НС с оптимизацией по Adam и критерием - категориальная кросс-энтропия
model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
            )

# Запуск  процесса обучения:80% - обучающая выборка, 20% - выборка валидации
model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2)


model.evaluate(x_test, y_test_cat)

# Проверка расспознования цифр
n = 150
x = np.expand_dims(x_test[n], axis=0)
res = model.predict(x)
print(res)
print(f"Распознования цифр: {np.argmax(res)}")

plt.imshow(x_test[n], cmap=plt.cm.binary)
plt.show()

# Распознование всей тестовой выборки
pred = model.predict(x_test)
pred = np.argmax(pred, axis=1)

print(pred.shape)
print(pred[:20])
print(y_test[:20])

#Выделение неверных вариантов
mask = pred = y_test
print(mask[:10])

x_false = x_test[~mask]
p_false = pred[~mask]

print(x_false.shape)


# Вывод первых 5 неверных результатов
for i in range(5):
    print(f"Значение сети: "+str(y_test[i]))
    plt.imshow(x_false[i], cmap=plt.cm.binary)
    plt.show()