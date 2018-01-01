#Training a 1-hidden layer neural network using the Tensorflow library.


import  numpy as np 
import matplotlib.pyplot as plt 
from scipy.linalg import qr
import numdifftools as nd 
import tensorflow as tf 

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


#PARAMETERS

n = 600 #number of datapoints
dimx = 13 #dimension of input
dimy = 4 #dimension of output
bs = 30 #batch size (default 30)
h = 4 #size of the hidden layer (default 4)
epochs = 200 #number of epochs (default 50)
lr = 1e-3 #learning rate (default 1e-5)
alpha = 0.9 #momentum parameter (default 0.9)
sigma = 0.5
mu = 0
ev = 1e8 #max eigenvalue (default 1e0)


#FUNCTIONS

def random_normal_init(shape):
    return np.random.normal(size=shape, scale=0.05)

def xavier_init(shape):
    epsilon = np.sqrt(6/sum(shape))
    return np.random.uniform(low=-epsilon, high=epsilon, size=shape)

def newton_function(M):
    preds = np.matmul(x_enlarged,M)
    return np.mean((preds-y.reshape((n*dimy)))*(preds-y.reshape((n*dimy))))

def build(x):

    x = tf.layers.dense(x, h, activation=None)

    x = tf.layers.dense(x, dimy, activation=None)

    return x


#DATA

x = np.random.rand(n,dimx) #data
m = np.mean(x)
s = np.std(x)

Aux1 = np.random.randn(dimx,dimx)
U, R1 = qr(Aux1)
Aux2 = np.random.randn(dimy,dimy)
V, R2 = qr(Aux2)
diag = sigma*np.random.randn(min(dimx,dimy))+mu
D = np.zeros((dimx,dimy))
for i in range(min(dimx,dimy)):
    D[i,i] = diag[i]
D[0,0] += ev #introducing bad conditioning

A = np.matmul(np.matmul(U, D), V.T) #true relation that we try to learn
print('Condition number of the A matrix: {}'.format(np.linalg.cond(A)))
print(A.shape)

y = np.matmul(x,A) #truth (labels)


#1-STOCHASTIC GRADIENT DESCENT WITH MOMENTUM

print('SGD with momentum using the Tensorflow package')

losses_gd = []

global_step = tf.Variable(0, trainable=False)
inc_global_step = tf.assign(global_step, global_step+1)

n_input = tf.placeholder(tf.float32, shape=(bs, dimx))
n_label = tf.placeholder(tf.float32, shape=(bs, dimy))
n_output = build(n_input)

loss = tf.nn.l2_loss(n_output-n_label)
optim = tf.train.GradientDescentOptimizer(learning_rate = lr)
train_op = optim.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)

    batch = sess.run(global_step)

    avg_loss = []

    while (batch < int((n*epochs)/bs)):

        for j in range(int(n/bs)):

            data_batch = x[j*bs:(j+1)*bs]
            labels = y[j*bs:(j+1)*bs]

            current_loss, _ = sess.run((loss, (train_op, inc_global_step)), feed_dict={n_input: data_batch, n_label: labels})
            avg_loss.append(current_loss)
            batch += 1

            if batch > 0 and batch % int(n/bs) == 0:
                print('Epoch number: {}'.format(int(batch/int(n/bs))))
                print('loss on this epoch: {}'.format(np.mean(np.array(avg_loss))))
                losses_gd.append(np.mean(np.array(avg_loss)))
                avg_loss = []
        
#PLOTTING THE LOSS

plt.plot(np.log(np.array(losses_gd)))
plt.show()


#2-NEWTON'S METHOD

print('Newton method')

losses_newton = []

W1 = random_normal_init((dimx,h))
W2 = random_normal_init((h,dimy))
x_enlarged = np.zeros((n*dimy,dimx*dimy))
for i in range(n*dimy):
    x_enlarged[i,(i % dimy)*dimx:((i % dimy) +1)*dimx] = x[int(i/dimy),:]
W = np.matmul(W1,W2)
W = W.reshape((dimx*dimy))

for i in range(epochs):

    print('Epoch number: {}'.format(i))

    gd = nd.Gradient(newton_function)
    
    Hess = nd.Hessian(newton_function)
    Hinv = np.linalg.solve(Hess(gd(W)),np.identity(dimx*dimy))

    W = W-np.matmul(Hinv, gd(W))

    loss = newton_function(W)
    print('loss on this epoch: {}'.format(loss))
    losses_newton.append(loss)

#PLOTTING THE LOSS

plt.plot(np.log(np.array(losses_newton)))
#plt.ylim(ymin=-50, ymax=50)
plt.show()

