#실습코드
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary

import matplotlib.pyplot as plt
import sys

class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.hidden1 = nn.Linear(1, 50)
        self.hidden2 = nn.Linear(50, 50)
        self.hidden3 = nn.Linear(50, 50)
        self.hidden4 = nn.Linear(50, 50)

        self.output = nn.Linear(50, 1)
        
    def forward(self, x):
        x = torch.relu(self.hidden1(x))  
        x = torch.relu(self.hidden2(x)) 
        x = torch.relu(self.hidden3(x))  
        x = torch.relu(self.hidden4(x))   
       
        x = self.output(x) 
        return x
    
class DatasetGenerator:
    def __init__(self, n_points=1000, x_begin=0, x_end=10):
        self.n_points = n_points
        self.x_begin = x_begin
        self.x_end = x_end
        self.noise_level = 0.1

    # y = a*x + b
    def generate_linear(self, a=1, b=2):
        x_values = np.random.rand(self.n_points, 1) * (self.x_end - self.x_begin) + self.x_begin
        noise = np.random.normal(0, self.noise_level, x_values.shape)
        y_values = a * x_values + b + noise
        return torch.tensor(x_values, dtype=torch.float32), torch.tensor(y_values, dtype=torch.float32)

    # y = a + b*x + c*x^2 + d*x^3 
    def generate_polynomial(self, a=27, b=-9, c=-3, d=1):
        x_values = np.random.rand(self.n_points, 1) * (self.x_end - self.x_begin) + self.x_begin
        noise = np.random.normal(0, self.noise_level, x_values.shape)
        y_values = a + b * x_values + c * np.power(x_values, 2) + d * np.power(x_values, 3) + noise
        return torch.tensor(x_values, dtype=torch.float32), torch.tensor(y_values, dtype=torch.float32)

    # y = a*e^(b*x)
    def generate_exponential(self, a=1, b=2):
        x_values = np.random.rand(self.n_points, 1) * (self.x_end - self.x_begin) + self.x_begin
        noise = np.random.normal(0, self.noise_level, x_values.shape)
        y_values = a * np.exp(b * x_values) + noise
        return torch.tensor(x_values, dtype=torch.float32), torch.tensor(y_values, dtype=torch.float32)

    # y = a + b*log(x)
    def generate_logarithmic(self, a=1, b=2):
        x_values = np.random.rand(self.n_points, 1) * (self.x_end - self.x_begin) + self.x_begin
        noise = np.random.normal(0, self.noise_level, x_values.shape)
        y_values = a + b * np.log(x_values) + noise
        return torch.tensor(x_values, dtype=torch.float32), torch.tensor(y_values, dtype=torch.float32)

    # y = 1/(1+e^(a+b*x))
    def generate_sigmoid(self, a=1, b=2):
        x_values = np.random.rand(self.n_points, 1) * (self.x_end - self.x_begin) + self.x_begin
        noise = np.random.normal(0, self.noise_level, x_values.shape)
        y_values = 1 / (1 + np.exp(-(a + b * x_values))) + noise
        return torch.tensor(x_values, dtype=torch.float32), torch.tensor(y_values, dtype=torch.float32)

    # y = a*sin(b*x+c)
    def generate_sine(self, a=1, b=2, c=3):
        x_values = np.random.rand(self.n_points, 1) * (self.x_end - self.x_begin) + self.x_begin
        noise = np.random.normal(0, self.noise_level, x_values.shape)
        y_values = a * np.sin(b * x_values + c) + noise
        return torch.tensor(x_values, dtype=torch.float32), torch.tensor(y_values, dtype=torch.float32)

        

class ModelTrainer:
    def __init__(self, model, n_epochs=200, learning_rate=0.001):
        self.model = model
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        
        self.gradient_clip_val = 1000  # 그래디언트 클리핑 값
        
        self.criterion = nn.MSELoss() #CrossEntropyLoss, BCELoss
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # Adam, Adagrid, RMSprop
        
        self.losses = []

    def train(self, x_train, y_train):
        for epoch in range(self.n_epochs):
            self.model.train() #학습모드 지정
            self.optimizer.zero_grad() #에폭마다 GD값 초기화
            
            outputs = self.model(x_train) #현재 파라미터 세팅에 대한 결과 prediction
            loss = self.criterion(outputs, y_train) #손실값 계산
            loss.backward() #backward propagation, 파라미터별 gd 계산
           
            # 그래디언트 클리핑 적용
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)  
            
            self.optimizer.step() #파라미터 값 업데이트
            
            self.losses.append(loss.item())

            if (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.n_epochs}], Loss: {loss.item():.4f}')


class Visualizer:
    def __init__(self, model_trainer):
        self.model_trainer = model_trainer

    def plot_loss(self):
        plt.plot(self.model_trainer.losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss During Training')
        plt.show()

    def plot_prediction(self, x_train, y_train):
        with torch.no_grad():
            self.model_trainer.model.eval()
            predicted = self.model_trainer.model(x_train).numpy()


        # x_train으로 정렬
        sorted_indices = np.argsort(x_train.numpy().flatten())
        sorted_x_train = x_train.numpy().flatten()[sorted_indices]
        sorted_predicted = predicted.flatten()[sorted_indices]

        
        plt.figure()  # 새로운 그래프 창
        plt.scatter(x_train.numpy(), y_train.numpy(), label='Actual Data')
        plt.plot(sorted_x_train, sorted_predicted, color='red', label='Fitted Line')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Actual vs Predicted')
        plt.legend()
        plt.show()
    
    
if __name__ == '__main__':    
    model = RegressionModel()
    res = summary(model, input_size=(1, 1)) 
    print(res)
    
    generator = DatasetGenerator(n_points=1000, x_begin=-10, x_end=10)

    #x_train, y_train = generator.generate_linear()
    #x_train, y_train = generator.generate_polynomial()
    #x_train, y_train = generator.generate_exponential()
    #x_train, y_train = generator.generate_logarithmic()
    #x_train, y_train = generator.generate_sigmoid()
    x_train, y_train = generator.generate_sine()

    #print(f"x_train={x_train}")
    #print(f"y_train={y_train}")

    #print(f"x_train.shape={x_train.shape}")
    #print(f"y_train.shape={y_train.shape}")
    
    trainer = ModelTrainer(model, n_epochs=10000)
    trainer.train(x_train, y_train)

    visualizer = Visualizer(trainer)
    visualizer.plot_loss()
    visualizer.plot_prediction(x_train, y_train)    
