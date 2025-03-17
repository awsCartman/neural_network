import os
from collections import Counter
import pandas as pd
from PIL import Image
import zipfile
import os
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from PIL import ImageFile
import time
import torch
import torchvision.models as models
import torch.nn as nn

import os
import random
from tqdm.notebook import tqdm

# Data manipulation and visualization
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import pandas as pd
import numpy as np

# Deep Learning libraries
import torch
import torchvision
from torchinfo import summary
from torch.utils import data
from torchvision import datasets, models, transforms

from torch.utils.data import DataLoader, Dataset
from glob import glob
from itertools import chain
import random

dataset_path = "data"


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset) # общее кол-во примеров в датасете
    sum_loss = 0 # переменная для накопления ошибки за эпоху
    
    model.train()

    # Цикл обучения (итерация по батчам)
    for batch, (X, y) in enumerate(dataloader):
        start_time = time.time()

        pred = model(X) # сеть делает предсказание
        loss = loss_fn(pred, y) # считаем ошибку.
        sum_loss += loss.item() # накапливаем сумму ошибок (для статистики).

        loss.backward() # вычисляет градиенты (как менять веса, чтобы уменьшить ошибку).
        optimizer.step() # обновляет веса модели.
        optimizer.zero_grad() # очищает старые градиенты (иначе они накапливаются).

        end_time = time.time()
        batch_time = end_time - start_time

        if batch % 30 == 0:
            print("Время обработки батча: ",  batch_time)
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    
    model.eval() 

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # correct — переменная для подсчёта количества правильных предсказаний.
    # test_loss — переменная для хранения суммарной потери (ошибки) на тестовых данных.


    # отключаем вычисление градиентов
    # Во время тестирования нам не нужны градиенты, 
    # поскольку мы не обновляем веса модели, а просто оцениваем её производительность.
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    # pred.argmax(1) == y сравнивает предсказанный класс с истинным классом. 
    # argmax(1) находит индекс с максимальной вероятностью 
    # для каждого предсказания (класс, который модель считает наиболее вероятным). 
    # Если предсказанный класс совпадает с истинным, то это правильно классифицированный пример. 

    test_loss /= num_batches # средняя ошибка
    correct /= size # точность
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct, test_loss


def run_training(epochs, train_dataloader, model, criterion, optimizer):
    # списки для хранения потерь
    test_loss = [] 
    train_loss = []
    
    # цикл по эпохам
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        
        # обучение
        sum_loss = train_loop(train_dataloader, model, criterion, optimizer)
        train_loss.append(sum_loss)

        # тест
        acc, avg_loss = test_loop(test_loader, model, criterion)
        test_loss.append(avg_loss)
    print("Done!")
    
    return test_loss

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


for root, _, files in os.walk(dataset_path):
    for file in files:
        file_path = os.path.join(root, file)
        try:
            img = Image.open(file_path)
            img.verify()  # Проверка целостности изображения
        except (IOError, OSError):
            print(f"Найдено поврежденное изображение, удаляем: {file_path}")

            os.remove(file_path)  # Удаление поврежденного файла


train_dir = './data/train/'
test_dir = './data/test/'
val_dir = './data/val/'

mean = [0.2328, 0.2328, 0.2328]
std = [0.0350, 0.0350, 0.0350]

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resizing images
    transforms.ToTensor(),          # Convert image to tensor
    transforms.Normalize(mean=mean, std=std),
])

# Load the full train and test datasets
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)

# Create DataLoaders for each dataset
batch_size=32
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


lr = 0.0001  
epochs = 10
device = torch.device("cpu")

criterion = torch.nn.CrossEntropyLoss()


model = models.efficientnet_b2(weights=None)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


effnet = run_training(epochs, train_loader, model, criterion, optimizer)

torch.save(model, 'effnet_none_weights.pth')