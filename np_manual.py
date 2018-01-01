#Training a 1-hidden layer neural network with backpropagation done manually.


import  numpy as np 
import matplotlib.pyplot as plt 
from scipy.linalg import qr
import numdifftools as nd 


#PARAMETERS

n = 600 #number of datapoints
dimx = 13 #dimension of input
dimy = 4 #dimension of output
bs = 30 #batch size (default 30)
h = 4 #size of the hidden layer (default 4)
epochs = 20 #number of epochs (default 50)
lr = 1e-6 #learning rate (default 1e-5)
alpha = 0.9 #momentum parameter (default 0.9)
sigma = 0.5
mu = 0
ev = 0 #max eigenvalue (default 1e0)


#FUNCTIONS

def l2_loss(ypred,ytrue):
	return np.mean((ypred-ytrue)*(ypred-ytrue))

def random_normal_init(shape):
	return np.random.normal(size=shape, scale=0.05)

def xavier_init(shape):
	epsilon = np.sqrt(6/sum(shape))
	return np.random.uniform(low=-epsilon, high=epsilon, size=shape)

def newton_function(M):
	preds = np.matmul(x_enlarged,M)
	return np.mean((preds-y.reshape((n*dimy)))*(preds-y.reshape((n*dimy))))
	

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

print('SGD with momentum')

W1 = xavier_init((dimx,h))
W2 = xavier_init((h,dimy))

losses_gd = []
delta1 = 0
delta2 = 0

for i in range(epochs):

	print('Epoch: {}'.format(i))
	avg_loss = []
	np.random.shuffle(x)

	for j in range(int(n/bs)):

		batch = x[j*bs:(j+1)*bs]
		labels = y[j*bs:(j+1)*bs]

		h_layer = np.matmul(batch, W1)
		preds = np.matmul(h_layer, W2)
		batch_loss = l2_loss(preds, labels)
		#print(batch_loss)
		avg_loss.append(batch_loss)
		#losses_gd.append(batch_loss)

		W1_derivatives = np.zeros((dimx,h))
		W2_derivatives = np.zeros((h,dimy))

		#backprop on W2 weights
		for a in range(h):
			for b in range(dimy):
				W2_derivatives[a,b] = np.mean((preds[:,b]-labels[:,b])*(preds[:,b]*(1-preds[:,b]))*h_layer[:,a], axis=0)

		#update W2
		delta2 = alpha*delta2 - lr*W2_derivatives 
		W2 = W2 + delta2

		#backprop on W1 weights
		for a in range(dimx):
			for b in range(h):
				partial_sum = 0
				for c in range(dimy):
					partial_sum += W2[b,c]*(preds[:,c]-labels[:,c])*preds[:,c]*(1-preds[:,c])
				W1_derivatives[a,b] = np.mean(partial_sum*h_layer[:,b]*(1-h_layer[:,b])*batch[:,a], axis=0)

		#update W1
		delta1 = alpha*delta1 - lr*W1_derivatives
		W1 = W1 + delta1

	#print avg loss on the epoch
	print('loss on this epoch: {}'.format(np.mean(np.array(avg_loss))))
	losses_gd.append(np.mean(np.array(avg_loss)))

#PLOTTING THE LOSS

plt.plot(np.log(np.array(losses_gd)))
#plt.show()


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

