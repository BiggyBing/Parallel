import pandas as pd
import numpy as np

from keras.datasets import mnist
(train_feature, train_label), (X_test, y_test) = mnist.load_data()
#train = pd.read_csv('mnist_train.csv')
#test = pd.read_csv('mnist_test.csv')
#train = np.array(train)
#test = np.array(test)
#train_feature = train[:,1:785]
#train_label = train[:,0]
train_target = np.zeros(len(train_label))

for i in range(len(train_label)):
    if train_label[i]%2 == 0:
        train_target[i] = 1
    else:
        train_target[i] = -1
        
        
train_feature = np.array(train_feature).reshape(60000,-1)