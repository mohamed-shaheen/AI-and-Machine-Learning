

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt




path = 'D:\\AI projects\\Course\\Classification\\data\\two varible data with regularization.csv'


data = pd.read_csv(path, header=None, names=['Test 1', 'Test 2', 'Accepted'])


postitive = data[data['Accepted'].isin([1])]
negative = data[data['Accepted'].isin([0])]


fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(postitive['Test 1'], postitive['Test 2'], s=50, c='b', marker='o', label = 'Accepted')
ax.scatter(negative['Test 1'], negative['Test 2'], s=50, c='r', marker = 'x', label = 'Not accepted')
ax.legend()
ax.set_xlabel('Test 1 score')
ax.set_ylabel('Test 2 score')



x1 = data['Test 1']
x2 = data['Test 2']

data.insert(3, 'ones', 1)

degree = 5

for i in range(1, degree):
    for j in range(0, i):
        data['F' + str(i) + str(j)] = np.power(x1, i-j) * np.power(x2, j)
        
        
data.drop('Test 1', axis=1, inplace=True)
data.drop('Test 2', axis=1, inplace= True)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def costReg(theta, x, y, lr):
    theta = np.matrix(theta)
    x = np.matrix(x)
    y = np.matrix(y)
    
    first = np.multiply(-y, np.log(sigmoid(x * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(x * theta.T)))
    
    reg = (lr / 2 * len(x)) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))
    
    return np.sum(first - second) / len(x) + reg

def gradintReg(theta, x, y, lr):
    
    theta = np.matrix(theta)
    
    x = np.matrix(x)
    y = np.matrix(y)
    
    
    parametrs = int(theta.ravel().shape[1])
    grad = np.zeros(parametrs)
    
    error = sigmoid(x*theta.T) - y
    
    for i in range(parametrs):
        term = np.multiply(error, x[:,i])
        if (i==0):
            grad[i] = np.sum(term)/len(x)
        else:
            grad[i] = np.sum(term)/len(x) + (lr/len(x)) * theta[:,i]

    return grad

cols = data.shape[1]
X2 = data.iloc[:, 1:cols]
Y2 = data.iloc[:, 0:1]

X2 = np.array(X2.values)
Y2 = np.array(Y2.values)
theta = np.zeros(X2.shape[1])

learningRate = 0.00001 

rcost = costReg(theta, X2, Y2, learningRate)     

result = opt.fmin_tnc(func=costReg , x0=theta, fprime=gradintReg, args=(X2,Y2, learningRate))

cost_after_optimize = costReg(result[0], X2, Y2, learningRate)

print('cost after optimize = ' , cost_after_optimize)


def predict(theta, x):
    
    probability = sigmoid(x*theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]

theta_min = np.matrix(result[0])
print ('min theta:',theta_min )
prediction = predict(theta_min, X2)

correct = [1 if (a==1 and b==1) or (a==0 and b==0) else 0 for (a,b) in zip(prediction,Y2)]
accuracy = sum(map(int,correct))%len(correct)

print ('accuracy = {0}%'.format(accuracy)) 
        
