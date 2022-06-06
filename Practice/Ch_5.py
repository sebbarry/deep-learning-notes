import numpy as np
from matplotlib.pyplot import plot


np.random.seed(0)

weights = np.array([0.5, 0.48, -0.7])
alpha = 0.1

# a matrix of inputs (what we have noted in the real world)
streetlights = np.array([
    [1, 0, 1],
    [0, 1, 1],
    [0, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
    [1, 0, 1]
])

# our goals (a matrix of goal outputs)
walk_vs_stop = np.array([[0, 1, 0, 1, 1, 0]]).T

def relu(x):
    return (x > 0) * x

input = streetlights[0]

goal_prediction = walk_vs_stop[0]

for i in range(20):
    prediction = input.dot(weights)
    error = (prediction - goal_prediction) ** 2
    delta = prediction - goal_prediction # delta on the curve.
    delta_d = delta * input # derivative
    weights -= (alpha * delta_d)



def request(url):
    import urllib.request, gzip, os, hashlib
    filename = hashlib.md5(url.encode('utf-8')).hexdigest()
    if not os.path.isfile(filename): 
        data = urllib.request.urlopen(url).read()
        with open(filename, 'wb') as f: 
            f.write(data)
    else: 
        with open(filename, 'rb') as f: 
            data = f.read()
    d = gzip.decompress(data)




request("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")

