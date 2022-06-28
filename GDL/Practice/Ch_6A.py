import numpy as np

np.random.seed(1)


# define the activation functions
def relu(x):
    return (x > 0) * x

def relu2deriv(output):
    return (output > 0)


# define the values and inputs
inputs = np.array([
    [1, 0, 0], 
    [0, 1, 1], 
    [0, 0, 1], 
    [1, 1, 1]
])
goal_predictions = np.array([[1, 1, 0, 0]]).T
alpha  = 0.2

weights_0_1 = 2*np.random.random((3, 4)) - 1
weights_1_2 = 2*np.random.random((4, 1)) - 1


# define the training loop
for i in range(60):
    output_error = 0
    for j in range(len(inputs)):
        # define the predictions and layers
        layer_0 = inputs[j:j+1]
        layer_1 = relu(np.dot(layer_0, weights_0_1))
        layer_2 = np.dot(layer_1,weights_1_2)

        # define the error value.
        output_error += np.sum((layer_2 - goal_predictions[j:j+1]) ** 2)

        # calculate the delta value
        delta_layer_2 = (layer_2 - goal_predictions[j:j+1])
        delta_layer_1 = delta_layer_2.dot(weights_1_2.T)

        # apply activation function for gradient descent.
        # delta * input
        d_delta_layer_1 = delta_layer_1*relu2deriv(layer_1)      

        # adjust the weights.
        weights_1_2 -= (alpha * layer_1.T.dot(delta_layer_2))
        weights_0_1 -= (alpha * layer_0.T.dot(d_delta_layer_1))
       

    if i % 10 == 9: 
        print(output_error)


        
