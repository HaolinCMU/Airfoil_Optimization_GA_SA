# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 11:04:13 2019

@author: haolinl
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab

fig = pylab.figure(figsize=(12.8,12.8))
a = np.array([1,2,3,4,5])
b = np.array([1,2,3,4,5])
#fig, ax = plt.subplots(nrows=1, ncols=1)
#ax.plot(a,b,c='k')
pylab.plot(a,b,c='k')
#ax.axis('off')
pylab.axis('off')
fig.savefig('test.png')
fig.show()
pylab.close(fig)

img_temp = GDF.genFig(GA.pop[0], save_contour_path, save_fig_path)
img_temp = img_temp.reshape(1,-1)
DataLoader_temp = torch.utils.data.TensorDataset(torch.from_numpy(img_temp).float())
img_matrix_temp = torch.autograd.Variable(DataLoader_temp[:][0])
img_matrix_temp = img_matrix_temp.reshape(-1,1,128,128)
value_temp = neural_network(img_matrix_temp)




# Test Bezier curve fitting




# test Bayesian curve fitting