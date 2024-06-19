# -*- coding: utf-8 -*-
import jax.numpy as jnp
from jax import grad, random, jit
from jax.example_libraries.optimizers import adam
from jax.nn import sigmoid
from jax.nn import relu
import numpy as np
import jax.scipy as jsc
from tqdm import tqdm
import matplotlib.pyplot as plt

# Throughout the file we are mostly using JAX
# jax.numpy (jnp) is equivalent to numpy (np) except for np.random which is separately handled by jax.random
# jax.scipy (jsc) is equivalent to scipy
# A key should be provided everytime you sample with jax.random
# Before you sample, you may need to use `rng_key, key = random.split(rng_key)` to generate a key. See the implementation for details
# The types of jax variables may not be compatible with other libraries, and you can use converter functions like float() to solve it

size_data = 28
d_data = size_data * size_data  # the dimension of a datapoint x
d_latent = 16  # the dimension of the hidden variables in the encoder/decoder
d_z = 2  # the dimension of the latent variable z
min_prob = 1e-6
max_prob = 1 - min_prob
epochs = 100


def loaddata():
    train = []
    test = []
    anomaly = []
    train_y = []
    test_y = []
    with open('data/train.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            ar = line.split()
            ar = [float(a) for a in ar]
            train.append(jnp.array(ar))
    with open('data/train_label.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            train_y.append(int(line))
    with open('data/test.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            ar = line.split()
            ar = [float(a) for a in ar]
            test.append(jnp.array(ar))
    with open('data/test_label.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            test_y.append(int(line))
    with open('data/anomaly.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            ar = line.split()
            ar = [float(a) for a in ar]
            anomaly.append(jnp.array(ar))
    return train, train_y, test, test_y, anomaly


rng_key = random.PRNGKey(0)
ranges = [d_z * d_latent, d_latent, d_latent * d_data, d_data, d_latent * d_data, d_latent, d_z * d_latent, d_z]
ranges = np.cumsum(ranges)
d_parameters = ranges[-1]


# we packed all parameters into one vector v (dimension is d_parameters) for automatic differentiation
# v2par translates v into the parameters
def v2par(v):
    w1 = jnp.reshape(v[:ranges[0]], (d_latent, d_z))
    b1 = jnp.reshape(v[ranges[0]:ranges[1]], (d_latent))
    w2 = jnp.reshape(v[ranges[1]:ranges[2]], (d_data, d_latent))
    b2 = jnp.reshape(v[ranges[2]:ranges[3]], (d_data))
    w3 = jnp.reshape(v[ranges[3]:ranges[4]], (d_latent, d_data))
    b3 = jnp.reshape(v[ranges[4]:ranges[5]], (d_latent))
    w4 = jnp.reshape(v[ranges[5]:ranges[6]], (d_z, d_latent))
    b4 = jnp.reshape(v[ranges[6]:], (d_z))
    return w1, b1, w2, b2, w3, b3, w4, b4


# par2v packs all the parameters into a vector v
def par2v(w1, b1, w2, b2, w3, b3, w4, b4):
    res = jnp.append(w1, b1)
    res = jnp.append(res, w2)
    res = jnp.append(res, b2)
    res = jnp.append(res, w3)
    res = jnp.append(res, b3)
    res = jnp.append(res, w4)
    return jnp.append(res, b4)


# estimate the elbo given x with a single sample of \epsilon
# You need to implement something here
def f(z, w1, b1, w2, b2):
    ##### WRITE YOUR CODE HERE #####

    ##### END YOUR CODE #####
    pass


def g(x, w3, b3, w4, b4):
    ##### WRITE YOUR CODE HERE #####

    ##### END YOUR CODE #####
    pass


def elbo_estimator(v, epsilon, x):
    w1, b1, w2, b2, w3, b3, w4, b4 = v2par(v)

    # You may use the following line to ensure numerical stability
    # x = jnp.clip(x,min_prob,max_prob)
    ##### WRITE YOUR CODE HERE #####

    ##### END YOUR CODE #####
    pass


elbo_gradient = jit(grad(elbo_estimator, argnums=(0)))  # the auto-diff command in JAX

train, train_y, test, test_y, anomaly = loaddata()

# Question 3
# elbo_gradient is a function that has the same input as elbo_estimator but returns the gradients with respect to v
# Set the parameters, v and x first and call elbo_gradient

# You need to implement something here
# What is the gradient for the ELBO w.r.t. b_4 for the first datapoint in training set?
##### WRITE YOUR CODE HERE #####

##### END YOUR CODE #####

# Question 4
# The codes here are the minimal codes for training, to answer question 4, you need to add codes after each epoch
rng_key, key = random.split(rng_key)
v = 0.1 * random.normal(key, (d_parameters,))  # random initialization
init, update, get_params = adam(1e-3)
state = init(v)  # the initial state of training, which includes not only v, but also some parameters for the optimizer
step = 0
elbo_per_epoch_train = np.zeros(epochs)
elbo_per_epoch_test = np.zeros(epochs)
for i in range(epochs):
    # train
    elbo_train_sum = 0
    for x in tqdm(train):  # tqdm is a library that makes a progress bar for the loop
        rng_key, key = random.split(rng_key)
        epsilon = random.normal(key, (d_z,))  # generate epsilon for this datapoint
        grad_v = elbo_gradient(v, epsilon, x)  # evaluate the gradients
        state = update(step, -grad_v,
                       state)  # update the parameters. Since we are maximizing the ELBO, we reverse the gradients here
        step += 1
        v = get_params(state)  # get_params is for retrieving the parameters from state

        ##### WRITE YOUR CODE HERE #####

        ##### END YOUR CODE #####

    # test
    v = get_params(state)
    elbo_test_sum = 0
    for sample in test:
        rng_key, key = random.split(rng_key)
        epsilon = random.normal(key, (d_z,))

        ##### WRITE YOUR CODE HERE #####

        ##### END YOUR CODE #####

    elbo_per_epoch_train[i] = elbo_train_sum / len(train)
    elbo_per_epoch_test[i] = elbo_test_sum / len(test)

# Plot Solution
plt.plot(range(epochs), elbo_per_epoch_train)
plt.plot(range(epochs), elbo_per_epoch_test)
plt.title("Average ELBO")
plt.legend(["Train", "Test"])
plt.xlabel("Epoch")
plt.ylabel("Average ELBO Value")
plt.show()
plt.clf()

# With the trained parameters, you can finish Question 5~8
v = get_params(state)
w1, b1, w2, b2, w3, b3, w4, b4 = v2par(v)


# Question 5
def plotCompressedData(compdata, complabels):
    compressedData = np.zeros((len(compdata), 2))
    ##### WRITE YOUR CODE HERE #####

    ##### END YOUR CODE #####
    plt.scatter(compressedData[:, 0], compressedData[:, 1], c=complabels)
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.title("Compressed Dimension of q(z|x)")
    plt.show()
    plt.clf()


plotCompressedData(test, test_y)

# Question 6

for im in range(10):
    rng_key, key = random.split(rng_key)

    ##### WRITE YOUR CODE HERE #####

    ##### END YOUR CODE #####

    plt.imshow(x.reshape(28, 28))
    plt.show()
    plt.clf()