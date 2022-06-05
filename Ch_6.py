"""
Python Script
"""


import numpy as np
np.random.seed(0)


# activation functions
def relu(x):
    return (x > 0) * x

def relu2deriv(output):
    return (output > 0)




#input data
streetlights = np.array([
    [1, 0, 1],
    [0, 1, 1],
    [0, 0, 1],
    [1, 1, 1]
])

#alpha value
alpha = 0.2


# goal_prediction values
walk_vs_stop = np.array([[1, 1, 0, 0]]).T


# weight values
weights_layer01 = 2*np.random.random((3, 4)) - 1
weights_layer12 = 2*np.random.random((4, 1)) - 1


# training loop here.
for i in range(60):
    output_layer_error = 0
    for j in range(len(streetlights)):
        # define the layers/predictions for each layer
        layer_0 = streetlights[j:j+1]
        layer_1 = relu(np.dot(layer_0, weights_layer01))
        output_layer = layer_1.dot(weights_layer12)

        # update the error value
        output_layer_error += np.sum((output_layer - walk_vs_stop[j:j+1]) ** 2)

        # find the delta values.
        delta_output = (output_layer - walk_vs_stop[j:j+1])
        delta_layer1 = delta_output.dot(weights_layer12.T)

        # apply gradient descent and activation functions
        d_delta_layer1 = delta_layer1 * relu2deriv(layer_1)

        # update the weights
        weights_layer12 -= (alpha * layer_1.T.dot(delta_output))
        weights_layer01 -= (alpha * layer_0.T.dot(d_delta_layer1))
         

    if i % 10 == 9: 
        print(output_layer_error)






     


