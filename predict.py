import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import datetime
import requests
from bs4 import BeautifulSoup

# data input
data = pd.read_csv('./loa_chart.csv')

# 날짜순 정렬
data = data.set_index("Date")
data = data.sort_index(ascending=True)

data_skt = data[["item1"]]

# Scaling
scaler = MinMaxScaler()
data_skt["item1"] = scaler.fit_transform(data_skt["item1"].values.reshape(-1, 1))

# return
def make_return(data):
    return_list = [0]

    for i in range(len(data) - 1):
        if (data.iloc[i + 1]["item1"] / data.iloc[i]["item1"]) - 1 >= 0:
            return_list.append(1)
#가격이 상승 = 1, 하락 = 0
        else:
            return_list.append(0)

    return return_list


data_skt["return"] = make_return(data)


# sequence data
# 30일치의 여러 Data들을 최근 30일치를 제외하고 생성
def make_data(data, window_size=30):
    feature_list = []
    label_list = []

    if "return" in data.columns:

        for i in range(len(data) - window_size):
            feature_list.append(np.array(data.iloc[i:i + window_size]["item1"]))
            label_list.append(np.array(data.iloc[i + window_size]["return"]))

        data_X = np.array(feature_list)
        data_Y = np.array(label_list)

        return data_X, data_Y

    else:

        for i in range(len(data) - 5):
            feature_list.append(np.array(data.iloc[i:i + window_size]["item1"]))

        data_X = np.array(feature_list)

        return data_X


skt_X, skt_Y = make_data(data_skt)

# 날짜별, Train/Test 데이터 셋 구분
train_data, train_label = skt_X[:-300], skt_Y[:-300]
test_data, test_label = skt_X[-300:], skt_Y[-300:]

train_data = torch.FloatTensor(train_data)
train_data = train_data.view(train_data.shape[0], 1, train_data.shape[1])  ## 배치수, 채널, Row

train_label = torch.FloatTensor(train_label)
train_label = train_label.view(train_label.shape[0], 1)

train_cnn = torch.utils.data.TensorDataset(train_data, train_label)
train_loader = torch.utils.data.DataLoader(dataset=train_cnn, batch_size=train_data.shape[0], shuffle=False)

test_data = torch.FloatTensor(test_data)
test_data = test_data.view(test_data.shape[0], 1, test_data.shape[1])

test_label = torch.FloatTensor(test_label)
test_label = test_label.view(test_label.shape[0], 1)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv1d(1, 5, 3)  ## 1*30-> 5*28
        self.relu1 = nn.ReLU()
        self.max1d1 = nn.MaxPool1d(2, stride=2)  ## 5*28 -> 5*14
        self.conv2 = nn.Conv1d(5, 10, 3)  ## 5*14->10*12
        self.relu2 = nn.ReLU()
        self.max1d2 = nn.MaxPool1d(2, stride=2)  ##10*12->10*6

        self.fc1 = nn.Linear(60, 30)
        self.fc2 = nn.Linear(30, 15)
        self.fc3 = nn.Linear(15, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.max1d1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.max1d2(x)
        x = x.view(-1, 60)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


model = CNN()
loss_fn = nn.L1Loss()
optimizer = torch.optim.RAdam(model.parameters(), lr=0.005)

loss_list = []

for epoch in range(3000):
    for i, (train_, label_) in enumerate(train_loader):
        optimizer.zero_grad()
        y_pred = model(train_)
        loss = loss_fn(y_pred, label_)
        loss.backward()
        optimizer.step()
    if epoch % 30 == 0:
        print(epoch, loss.item())
        loss_list.append(loss.item())

pred_ = torch.round(torch.sigmoid(model(test_data)))
test_pred_ = pred_.detach().numpy()

print("Accuracy:", metrics.accuracy_score(test_label, test_pred_))
plt.plot(loss_list, label="Loss")
plt.legend()
plt.show()