import math

import numpy as np

dataset = [
    ([1, 1], 1),
    ([1, 2], 2),
    ([1, 3], 3),
    ([1, 4], 4),
    ([1, 5], 5),
    ([1, 6], 6),
    ([1, 7], 7),
    ([1, 8], 8),
    ([1, 9], 9),
    ([1, 10], 10),
    ([2, 1], 2),
    ([2, 2], 4),
    ([2, 3], 6),
    ([2, 4], 8),
    ([2, 5], 10),
    ([2, 6], 12),
    ([2, 7], 14),
    ([2, 8], 16),
    ([2, 9], 18),
    ([2, 10], 20),
    ([3, 1], 3),
    ([3, 2], 6),
    ([3, 3], 9),
    ([3, 4], 12),
    ([3, 5], 15),
    ([3, 6], 18),
    ([3, 7], 21),
    ([3, 8], 24),
    ([3, 9], 27),
    ([3, 10], 30),
    ([4, 1], 4),
    ([4, 2], 8),
    ([4, 3], 12),
    ([4, 4], 16),
    ([4, 5], 20),
    ([4, 6], 24),
    ([4, 7], 28),
    ([4, 8], 32),
    ([4, 9], 36),
    ([4, 10], 40),
    ([5, 1], 5),
    ([5, 2], 10),
    ([5, 3], 15),
    ([5, 4], 20),
    ([5, 5], 25),
    ([5, 6], 30),
    ([5, 7], 35),
    ([5, 8], 40),
    ([5, 9], 45),
    ([5, 10], 50),
    ([6, 1], 6),
    ([6, 2], 12),
    ([6, 3], 18),
    ([6, 4], 24),
    ([6, 5], 30),
    ([6, 6], 36),
    ([6, 7], 42),
    ([6, 8], 48),
    ([6, 9], 54),
    ([6, 10], 60),
    ([7, 1], 7),
    ([7, 2], 14),
    ([7, 3], 21),
    ([7, 4], 28),
    ([7, 5], 35),
    ([7, 6], 42),
    ([7, 7], 49),
    ([7, 8], 56),
    ([7, 9], 63),
    ([7, 10], 70),
    ([8, 1], 8),
    ([8, 2], 16),
    ([8, 3], 24),
    ([8, 4], 32),
    ([8, 5], 40),
    ([8, 6], 48),
    ([8, 7], 56),
    ([8, 8], 64),
    ([8, 9], 72),
    ([8, 10], 80),
    ([9, 1], 9),
    ([9, 2], 18),
    ([9, 3], 27),
    ([9, 4], 36),
    ([9, 5], 45),
    ([9, 6], 54),
    ([9, 7], 63),
    ([9, 8], 72),
    ([9, 9], 81),
    ([9, 10], 90),
    ([10, 1], 10),
    ([10, 2], 20),
    ([10, 3], 30),
    ([10, 4], 40),
    ([10, 5], 50),
    ([10, 6], 60),
    ([10, 7], 70),
    ([10, 8], 80),
    ([10, 9], 90),
    ([10, 10], 100),
]

lr = 0.1
class Layer:
    def __init__(self,X,input_size,neurons) -> None:
        self.W = np.random.rand(input_size, neurons)
        self.b = np.random.rand(neurons)
        self.X = np.array(X)
        self.input_size = input_size#кол во входов в слой
        self.neurons = neurons#кол во нейронов в слое


    def activation(self):
        z = self.X @ self.W + self.b#сумма нейрона
        self.z = z
        a = 1/(1+(math.e**(-z)))#активация сигмоидой
        self.a = a
        return a

    def Loss(self,y):
        loss = (np.sum(self.a) - y)**2 #функция лосса на весь слой
        self.loss = loss
        return loss

    def der(self,da):
        
        dz = self.a * (1 -self.a) #производная по активации
        dw = np.outer(self.X, da * dz) #производная лосса по весам
        self.dw = dw
        db = da * dz# производная для байасов
        self.db = db
        dX = self.W @ db
        return dw,db,dX

    def update(self):
        self.W -= lr * self.dw
        self.b -= lr * self.db

layer1,layer2,layer3 = Layer([1,1],2,8),Layer(np.zeros(8), 8, 32),Layer(np.zeros(32),32,1)

def forward(layer1,layer2,layer3):
    
    a1 = layer1.activation()
    layer2.X = a1
    a2 = layer2.activation()
    layer3.X = a2
    a3 = layer3.activation()
    return a3 #ответ нейросети
    
def backward(layer1,layer2,layer3,y):
    #dXn это производная по входам в нейрон и используеться дла передачи на след слой так же da n
    loss = layer3.Loss(y)#лосс по последнему слою с данными обучения
    da3 = 2 * (np.sum(layer3.a) - y)#производная функции активации по лоссу последнего слоя  по 3у слою
    dw3,db3,dX3 = layer3.der(da3)#градиент производных для 3е слоя
    dw2,db2,dX2 = layer2.der(dX3)#передача ошибки в пред слой 
    dw, db, dX = layer1.der(dX2)#передача ошибки в самый первый слой 
    #обновление весов 1 2 3 слоев
    layer1.update()
    layer2.update()
    layer3.update()
    return loss


def train(layer1,layer2,layer3):
    epoch = 0
    
    loss = 1
    while loss >= 0.00001 and epoch < 200000:
        total_loss = 0
        for x,y in dataset:
             layer1.X = np.array(x) / 10 #нормализация данных
             y = y / 100 #нормализация данных
             forward(layer1,layer2,layer3)
             total_loss += backward(layer1,layer2,layer3,y)
        epoch += 1
        loss = total_loss/len(dataset)     
        if epoch % 1000 == 0:
            print(f'epoch: {epoch}, loss: {loss}')
             
train(layer1,layer2,layer3)         

def main(layer1,layer2,layer3):
    layer1.X = [float(input('X1: '))/10, float(input('X2:'))/10]
    pred = forward(layer1,layer2,layer3)
    print(f'Ответ нейросети: {pred*100}')

while True:
    main(layer1,layer2,layer3)