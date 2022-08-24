import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize



path = 'D:\\AI projects\\Course\\Linear regression\\data\\one variable data.csv'

data = pd.read_csv(path, header=None, names=['Population', 'Profit'])


data.plot(kind='scatter', x='Population', y='Profit', figsize=(5,5))


data.insert(0, 'ones', 1)

cols = data.shape[1]

X= data.iloc[:,0:cols-1]

Y = data.iloc[:,cols-1:cols]

X = np.matrix(X.values)

Y = np.matrix(Y.values)

theta = np.matrix(np.array([0,0]))



def compuCost(x, y, theta):
    z = np.power(((x*theta.T)-y), 2)
    
    return np.sum(z)/(2*len(x))
    




def gradientDescent(x, y, theta, alpha, iters):
    
    temp = np.matrix(np.zeros(theta.shape))
    parameters= int(theta.ravel().shape[1])
    
    cost = np.zeros(iters)
    
    for i in range(iters):
        
        error = (x * theta.T) - y
        
        for j in range(parameters):
            
            term = np.multiply(error, x[:,j])
            temp[0,j] = theta[0,j] - ((alpha/len(x)) * np.sum(term))
            theta = temp
            cost[i] = compuCost(x, y, theta)
       
    return theta, cost

    

alpha = 0.01
iters = 3000

g, cost = gradientDescent(X, Y, theta, alpha, iters)

print('theta 0&1:', g)
print('\n cost:', cost[iters-1:iters])


x = np.linspace(data.Population.min(), data.Population.max(), 100)

hx = g[0,0]+(g[0,1]*x)

fig, ax = plt.subplots(figsize=(5,5))

ax.plot(x, hx, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')



fig2, ax2 = plt.subplots(figsize=(5,5))
ax2.plot(np.arange(iters), cost, 'r-.')
ax2.set_xlabel('Iterations')
ax2.set_ylabel('Cost')
ax2.set_title('Error vs. Training Epoch')




    