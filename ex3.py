import numpy as np


def f(x):
    return 2/(1 + np.exp(-x)) - 1

def df(x):
    return 0.5 *(1 + x) * (1 - x)

W1 = np.array([[-0.2, 0.3, -0.4], [0.1, -0.3, -0.4]])
W2 = np.array([0.2, 0.3])

def go_forward(inp):
    sum = np.dot(W1,inp)
    out = np.array([f(x) for x in sum])

    sum = np.dot(W2, out)
    y = f(sum)
    return (y, out)

def train(epoch):
    global W2, W1
    lmd = 0.001          #шаг обучения
    N=10000             # число итерация при обучении
    count = len(epoch)
    for k in range(N):
        x = epoch[np.random.randint(0, count)]  # Случайный выбор входного сигнала из обучающей выборки
        y , out = go_forward(x[0:3])            # прямой поход по НС и вычесление выходных значений нейрона
        e = y - x[-1]                           # Ошибка
        delta = e*df(y)                         # локальный градиент
        W2[0] = W2[0] -lmd * delta * out[0]     # Корриктировка веса первой связи
        W2[1] = W2[1] -lmd * delta * out[1]     # Корриктировка веса второй связи

        delta2 = W2 * delta * df(out)           # вектор из 2-х величин локального градиента

        # корриктировка связей первого слоя
        W1[0, :] = W1[0, :] - np.array(x[0:3]) * delta2[0] * lmd
        W1[1, :] = W1[1, :] - np.array(x[0:3]) * delta2[1] * lmd

epoch = [(-1, -1, -1, -1),
      (-1, -1, 1, 1),
      (-1, 1, -1, -1),
      (-1, 1, 1, 1),
      (1, -1, -1, -1),
      (1, -1, 1, 1),
      (1, 1, -1, -1),
      (1, 1, 1, -1)]

train(epoch)        # Запуск обучаемой сети


# Проверка получения резуьтатов
for x in epoch:
    y, out = go_forward(x[0:3])
    print(f"Выходное значение НС: {y} => {x[-1]}")