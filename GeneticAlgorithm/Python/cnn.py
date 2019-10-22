
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 16:09:33 2018

@author: zili & haolinl
"""

import numpy as np
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import scipy.io
import time

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
       
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,10,13),
#            nn.Dropout2d(0.5),
            nn.BatchNorm2d(10),
            nn.MaxPool2d(2,2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(10,20,7),
#            nn.Dropout2d(0.5),
            nn.BatchNorm2d(20),
            nn.MaxPool2d(2,2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(20,40,7),
            nn.BatchNorm2d(40),
#            nn.Dropout2d(0.5),
            nn.MaxPool2d(2,2),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(40,80,5),
            nn.BatchNorm2d(80),
#            nn.Dropout2d(0.5),
            nn.MaxPool2d(2,2),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(720,400),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Linear(400,1)
    
    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        f4 = self.conv4(f3)
        f4_flat = f4.view(f4.size(0), -1)
        f5 = self.fc1(f4_flat)
        output = self.fc2(f5)
        return output 

if __name__ == "__main__":
    train_data_path = 'all_data_1000.mat'
    neural_net_savePath = 'CNN.pkl' # Path to save CNN's parameters
    
    data = scipy.io.loadmat(train_data_path)
    data_x, data_y, rNorm = data['data_x'], data['data_y'], data['Normalization_Factor']
    num_data = np.shape(data_x)[0]
    train_x, train_y = data_x[:int(0.9*num_data)], data_y[:int(0.9*num_data)]
    valid_x,valid_y = data_x[int(0.9*num_data):], data_y[int(0.9*num_data):]
    #test_x, test_y = data_x[int(0.9*num_data):], data_y[int(0.9*num_data):]
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #set up parameters
    batch_size = 32
    learning_rate = 0.00005
    num_epochs = 200
    
    train_x = torch.from_numpy(train_x).float()
    train_y = torch.from_numpy(train_y).float()
    valid_x = torch.from_numpy(valid_x).float()
    valid_y = torch.from_numpy(valid_y).float()
    #test_x = torch.from_numpy(test_x).float()
    #test_y = torch.from_numpy(test_y).float()
    
    train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
    valid_dataset = torch.utils.data.TensorDataset(valid_x, valid_y)
    #test_dataset = torch.utils.data.TensorDataset(test_x, test_y)
    
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    valid_dataloader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    #test_dataloader = torch.utils.data.DataLoader(
    #    dataset=test_dataset
    #)
    neural_net = Net1().to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(neural_net.parameters(),learning_rate)
    lossList = []
    accList = []
    valid_lossList = []
    valid_accList = []
    
    for epoch in range(num_epochs):
        loss_sum_train = 0
        loss_sum_valid = 0
        acc_sum = 0
        
        for iteration, (images, labels) in enumerate(train_dataloader):
            x_batch = torch.autograd.Variable(images)
            x_batch = x_batch.reshape(-1,1,128,128)
            y_batch = torch.autograd.Variable(labels)
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            output = neural_net(x_batch)
            loss = criterion(output,y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc = abs((output.cpu() - labels)/labels).mean()
            acc_sum += acc
            loss_sum_train += loss.cpu().data.numpy()
        loss_sum_train = batch_size * loss_sum_train /len(train_dataloader.dataset)
        lossList.append(loss_sum_train*rNorm[0][0])
        
        for iteration, (images, labels) in enumerate(valid_dataloader):
            x_batch = torch.autograd.Variable(images)
            x_batch = x_batch.reshape(-1,1,128,128)
            y_batch = torch.autograd.Variable(labels)
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            output_valid = neural_net(x_batch)
            loss = criterion(output_valid,y_batch)
            loss_sum_valid += loss.cpu().data.numpy()
            
        loss_sum_valid = batch_size * loss_sum_valid/len(valid_dataloader.dataset)
        valid_lossList.append(loss_sum_valid*rNorm[0][0])
        print('Epoch: ', epoch, '| train loss: %.6f | valid loss: %.6f  ' % (loss_sum_train*rNorm[0][0], loss_sum_valid*rNorm[0][0]))
        
        if ((epoch+1)%20 == 0):
            CNN_savePath_temp = 'CNN_' + str(int((epoch+1)/20)) + '.pkl'
            torch.save(neural_net.state_dict(), CNN_savePath_temp) # Save model every 20 epochs
    
    np.save('valid_loss.npy', valid_lossList)
    np.save('train_loss.npy', lossList)
    torch.save(neural_net.state_dict(), neural_net_savePath) # Save the final CNN model parameters

#__name__ = "__main__"