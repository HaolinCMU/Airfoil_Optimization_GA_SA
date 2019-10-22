# -*- coding: utf-8 -*-
import numpy as np
import time
import matplotlib.pyplot as plt

valid_lossList = np.load('valid_loss.npy')
lossList = np.load('train_loss.npy')
pre = np.load('pre_y.npy' )
test = np.load('test_y.npy')


pre = np.array([pre]).T.squeeze()
test = np.array([test]).T.squeeze()
coordinate = np.vstack((pre,test))
mean = np.mean(coordinate,axis=1).reshape(2,1)
std = np.std(coordinate,axis=1).reshape(2,1)
data = ((coordinate - mean)/std).T


cov = np.cov(data)
eig_val,eig_vec = np.linalg.eig(cov)
data_temp = eig_vec[:,:3].reshape(-1,3).T @ data


variance1 = np.var(data_temp[:1])
variance2 = np.var(data_temp[1:2])
variance3 = np.var(data_temp[2:3])
print('.................................')
print('eig_value:',np.shape(eig_val),'\n')
print('eig_vector:',np.shape(eig_vec), '\n')
print('variance1',variance1 )
print('variance2',variance2 )
print('variance3',variance3 )
#print('.................................')
#print('The mean is',mean)
#print('The std is',std)