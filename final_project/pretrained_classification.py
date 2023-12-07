import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from torchvision import datasets, transforms 
import torchvision.models as models 
from torchinfo import summary

from sklearn.metrics import confusion_matrix

from tqdm import tqdm
import time

import matplotlib.pyplot as plt

# 완전 연결층만 사용하는 신경망 모델 정의
class RestImageClassificationModel_FC_only(nn.Module):
    def __init__(self):
        super(RestImageClassificationModel_FC_only, self).__init__()
        self.fc1 = nn.Linear(3*300*300, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):  # forward 메서드 수정, 누락된 'x' 매개변수 추가
        x = x.view(-1, 3*300*300)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# 컨볼루션층을 사용하는 신경망 모델 정의 (CNN)
class RestImageClassificationModel_CNN(nn.Module):
    def __init__(self):
        super(RestImageClassificationModel_CNN, self).__init__()
        # 여기에 컨볼루션층 및 다른 구성 요소를 정의합니다.

        self.conv1 = nn.Conv2d(3, 32, kernel_size = 3, stride=1, padding=1)
        self.conv2 = nn.COnv2d(3, 64, kernel_size = 3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size = 3, stirde = 2, padding =0)

        self.fc1 = nn.Linear(64 * 75 * 75, 128)

        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        x = x.view(-1, 64 * 75 *75)

        x = self.fc2(x)

        return x 

# 미리 학습된 InceptionV3 기반의 신경망 모델 정의
class RestImageClassificationModel_Pretrained(nn.Module):
    def __init__(self):
        super(RestImageClassificationModel_Pretrained, self).__init__()

        # inceptionV3 모델 로드
        self.base_model = models.inception_v3(pretrained=True, aux_logits=True)

        in_features = self.base_model.fc.in_features

        self.base_model.fc = nn.Identity()

        for param in self.base_model.parameters():
            param.requires_grad = False

        for param in self.base_model.Mixed_7c.parameters():
            param.requires_grad = True

        self.fc1 = nn.Linear(in_features, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        outputs = self.base_model(x)

        if self.training:
            outputs_main, outputs_aux = outputs
        else:
            outputs_main = outputs

        x = torch.relu(self.fc1(outputs_main))
        x = self.fc2(x)

        return x

# 이미지 데이터를 처리하는 DataLoader 클래스 정의
class RestImageDataLoader:
    def __init__(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # ImageFolder 대신에 datasets.ImageFolder 사용, 오타 수정
        dataset = datasets.ImageFolder(root='/ProjectImages', transform=transform)

        total_size = len(dataset)

        train_ratio = 0.7
        val_ratio = 0.1

        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        test_size = total_size - train_size - val_size

        # 데이터셋에 직접 random_split을 사용합니다.
        train_dataset, remaining_dataset = random_split(dataset, [train_size, val_size])
        val_dataset, test_dataset = random_split(remaining_dataset, [val_size, test_size])

        print(f"train_dataset = {len(train_dataset)}")
        print(f"val_dataset = {len(val_dataset)}")
        print(f"test_dataset = {len(test_dataset)}")

        # DataLoader를 사용하여 배치 처리합니다.
        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
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
    
    # model = RestImageClassificationModel_FC_only()
    # model = RestImageClassificationModel_CNN()
    # model = RestImageClassificationModel_Pretrained()

    res = summary(model, input_size = ( 32, 3, 300 ,300))
    print(res)
    
    dataloader = RestImageDataLoader()
    
    trainer = ModelTrainer(model, dataloader.train_loader, dataloader.val_loader, dataloader.test_loader, epochs=2)
    trainer.train()
    trainer.evaluate()

    end_time = time.time()
    
    print(f"elapsed_time={end_time - begin_time} seconds")