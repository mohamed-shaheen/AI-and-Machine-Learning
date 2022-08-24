
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



path = 'D:\\AI projects\\Course\\Linear regression\\data\\multy variable data.csv'

data = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])


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

#rescaling data
data = (data - data.mean()) / data.std()

data.insert(0, 'ones', 1)

cols = data.shape[1]
X = data.iloc[:,0:cols-1]
Y = data.iloc[:, cols-1:cols]

X = np.matrix(X.values)
Y= np.matrix(Y.values)
Theta = np.matrix(np.array([0,0,0]))


Alpha = 0.1
Iters = 100


g , cost = gradientDescent(X, Y, Theta, Alpha, Iters)


print('theta  : ', g)
print('Cost : ', cost[Iters-1:Iters])


size_x = np.linspace(data.Size.min(), data.Size.max(), 100)
hx = g[0,0] + (g[0,1] * size_x)

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(size_x, hx, 'r', label='Prediction' )
ax.scatter(data.Size, data.Price, label='Traing Data')
ax.legend(loc=2)
ax.set_xlabel('Size')
ax.set_ylabel('Price')
ax.set_title('Size vs. Price')


rooms_x = np.linspace(data.Bedrooms.min(), data.Bedrooms.max(), 100)

hx_rooms = g[0, 0] + (g[0, 2] * rooms_x)



fig, ax = plt.subplots(figsize=(5,5))
ax.plot(rooms_x , hx_rooms, 'r', label='Prediction')
ax.scatter(data.Bedrooms, data.Price, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Bedrooms')
ax.set_ylabel('Price')
ax.set_title('Size vs. Price')




fig, ax = plt.subplots(figsize=(5,5))
ax.plot(np.arange(Iters), cost, 'r.-')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')










