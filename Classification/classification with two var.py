

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt 


path = 'D:\AI projects\Course\Classification\data\data with two varible.csv'


data = pd.read_csv(path, header=None, names= ['Exam1', 'Exam2', 'Admitted'])



positive = data[data['Admitted'].isin([1])]
nigative = data[data['Admitted'].isin([0])]

fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(positive['Exam1'], positive['Exam2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(nigative['Exam1'], nigative['Exam2'], s=50, c='r', marker='x', label='Not Admitted')
ax.legend()

ax.set_xlabel('Exam1 score')
ax.set_ylabel('Exam2 score')

def sigmoid(z):
    return 1 / (1+np.exp(-z))


nums = np.arange(-10,10,1)

fig, ax = plt.subplots(figsize = (5,5))
ax.plot(nums, sigmoid(nums), 'r.-')


def cost(theta, x, y):
    
    theta = np.matrix(theta)
    x= np.matrix(x)
    y= np.matrix(y)
    
    
    first = np.multiply(-y, np.log(sigmoid(x * theta.T)))
    second = np.multiply((1-y), np.log(1-sigmoid(x * theta.T)))
    
    return np.sum((first - second) / len(x))

data.insert(0, 'ones', 1)

cols = data.shape[1]

x = data.iloc[:, :cols-1]

y = data.iloc[:, cols-1:cols]

x = np.array(x.values)

y = np.array(y.values)

theta = np.zeros(cols-1)




def gradint(theta, x, y):
    
    theta = np.matrix(theta)
    
    x = np.matrix(x)
    y = np.matrix(y)
    
    
    parametrs = int(theta.ravel().shape[1])
    grad = np.zeros(parametrs)
    
    error = sigmoid(x*theta.T) - y
    
    for i in range(parametrs):
        term = np.multiply(error, x[:,i])
        grad[i] = np.sum(term)/len(x)

    return grad



result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradint, args=(x,y))

cost_after_optimize = cost(result[0], x, y)

print('cost after optimize = ' , cost_after_optimize)

def predict(theta, x):
    
    probability = sigmoid(x*theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]

theta_min = np.matrix(result[0])
print ('min theta:',theta_min )
prediction = predict(theta_min, x)

correct = [1 if (a==1 and b==1) or (a==0 and b==0) else 0 for (a,b) in zip(prediction,y)]
accuracy = sum(map(int,correct))%len(correct)

print ('accuracy = {0}%'.format(accuracy))
                   
