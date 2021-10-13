#Simple Neural Network
#Oregon State University AI Club
#October 12, 2021

import numpy as np

#Sigmoid
#Keeping the output value 0 or 1
def sigmoid(x, deriv=False):
    if deriv:
        return sigmoid(x)*(1-sigmoid(x))
    return 1.0 / (1 + np.exp(-x))

# Make the array as the input
X = np.array ([
    [1, 2, 3]
], dtype=float)

# Set up the prediction
Y = np.array([
    [0.67]
], dtype=float)

# Set up the weight
W = np.array([
    [0.01],
    [0.005],
    [0.0010]
], dtype=float)
#w1 = np.random.randn(3, 13)    #Another method with random
#w2 = np.random.randn(13, 1)

# Change the bias here
bias = 0

# Change the learning rate here
lr = 0.1

# Preactivation
for i in range(100):
    preactivation = X.T[0]*W[0] + X.T[1]*W[1] + X.T[2]*W[2]+bias #Method 1
    #preactivation = np.dot(X, W)	                             #Method 2
    #preactivation = X @ W	                                     #Method 3
    print("Preactivation", preactivation)

    # Activation
    activation = sigmoid(preactivation)
    output = activation

    #Finding the loss
    loss = (output - Y)**2
    print("loss", loss)

    #Finding the gradient loss
    #Dl/Da
    loss_gradient = 2*(output - Y)

    #Activation gradient
    #Da/Dp
    activation_gradient = sigmoid(preactivation, True)

    #Update the bias
    updateBias = activation_gradient*loss_gradient*1

    #Preactivation gradient
    #Dp/w
    preactivation_gradient = X

    #Minimize the loss
    #Dl/w
    error_gradient = (loss_gradient * activation_gradient * preactivation_gradient).T

    #Update the weights
    W -= lr*error_gradient
    bias -= updateBias

print(output)