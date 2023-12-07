import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from torchvision import datasets, transforms

from sklearn.metrics import confusion_matrix

from tqdm import tqdm
import time

import matplotlib.pyplot as plt


class MNISTClassificationModel_FC_only(nn.Module):
    def __init__(self):
        super(MNISTClassificationModel_FC_only, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


class MNISTClassificationModel_CNN(nn.Module):
    def __init__(self):
        super(MNISTClassificationModel_CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1) #1=채널수, 32=출력채널(필터의 수), 필터크기 -> (32, H, W)
        #self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # output: (64, H, W) 

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # output: (H/2, W/2)

        self.fc1 = nn.Linear(32 * 14 * 14, 64)         
        #self.fc1 = nn.Linear(64 * 7 * 7, 64) 

        self.fc2 = nn.Linear(64, 10)  # 출력 계층으로 바로 연결

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x))) # (32, 14, 14)
        #x = self.pool(torch.relu(self.conv2(x))) # (64, 7, 7)

        x = x.view(-1, 32 * 14 * 14) # Flatten        
        #x = x.view(-1, 64 * 7 * 7) # Flatten

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x



class MNISTDataLoader:
    def __init__(self):        

        transform = transforms.Compose([
            transforms.ToTensor(), #픽셀값을 [0,1] 사이로 scaling
            transforms.Normalize((0.5,), (0.5,)) #픽셀값을 mean=0.5, std=0.5로 정규분포로 정규화
        ])

        full_train_dataset = datasets.MNIST(root='./', train=True, download=True, transform=transform)
        train_size = int(0.7 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size 
        train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
        test_dataset = datasets.MNIST(root='./', train=False, download=True, transform=transform)
    
        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        

class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, epochs=10, learning_rate=0.001):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        self.epochs = epochs

        self.train_losses = []
        self.val_losses = []

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            with tqdm(total=len(self.train_loader), desc=f"Epoch {epoch + 1}/{self.epochs}", unit="batch", leave=False) as pbar:
                for features, labels in self.train_loader:
                    outputs = self.model(features)  # __call__()
                    loss = self.criterion(outputs, labels)
                    train_loss += loss.item()

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    pbar.set_postfix({'train_loss': f'{loss.item():.4f}'})
                    pbar.update(1)

            train_loss /= len(self.train_loader)
            val_loss = self.validate()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)            
            
            print(f"Epoch {epoch + 1}/{self.epochs} - Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        self.plot_loss()

    def validate(self):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, labels in self.val_loader:
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
        
        val_loss /= len(self.val_loader)
        
        return val_loss


    def plot_loss(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.legend()
        plt.show()

    
    def evaluate(self):
        self.model.eval()
        test_loss = 0
        all_labels = []
        all_preds = []
        output_list = []
        output_prob_list = []  # 클래스별 확률을 저장할 리스트

        total = 0
        correct = 0
        

        with torch.no_grad():
            for features, labels in self.test_loader:
                #print(f"features={features}")
                #print(f"labels={labels}")                
                outputs = self.model(features.float())
                #print(f"outputs={outputs}")
              
                
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()
                
                output_list.extend(outputs.numpy())
                
                output_prob = F.softmax(outputs, dim=1)
                output_prob_list.extend(output_prob.numpy())

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_labels.extend(labels.numpy())
                all_preds.extend(predicted.numpy())
                #all_labels.extend(labels.cpu().numpy())
                #all_preds.extend(predicted.cpu().numpy())

            for f, l, p, o, prob in zip(features, labels, predicted, outputs, output_prob):
                #if l == p:
                #    continue
                plt.imshow(f.squeeze(), cmap='gray')
                plt.title(f'Predicted: {p}, Actual: {l}')
                plt.show()            
                print(f"{o}->{prob}: {l} -> {p}")


        test_loss /= len(self.test_loader)
        

        print(f"Accuracy = {correct/total: .4f}")

        cm = confusion_matrix(all_labels, all_preds)
        print(f"Test Loss: {test_loss:.4f}")
        print("Confusion Matrix:")
        print(cm)
        

if __name__ == '__main__':
    begin_time = time.time()
    
    #model = MNISTClassificationModel_FC_only()
    model = MNISTClassificationModel_CNN()
    
    dataloader = MNISTDataLoader()
    
    trainer = ModelTrainer(model, dataloader.train_loader, dataloader.val_loader, dataloader.test_loader, epochs=2)
    trainer.train()
    trainer.evaluate()

    end_time = time.time()
    
    print(f"elapsed_time={end_time - begin_time} seconds")
    
    
    
    