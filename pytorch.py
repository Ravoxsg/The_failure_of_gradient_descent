import  numpy as np 
import matplotlib.pyplot as plt 
from scipy.linalg import qr
import numdifftools as nd 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


#PARAMETERS

n = 600 #number of datapoints
dimx = 13 #dimension of input
dimy = 4 #dimension of output
bs = 30 #batch size (default 30)
h = 4 #size of the hidden layer (default 4)
epochs = 20 #number of epochs (default 50)
lr = 1e-5 #learning rate (default 1e-5)
alpha = 0.9 #momentum parameter (default 0.9)
sigma = 0.5
mu = 0
ev = 0 #max eigenvalue (default 1e0)
criterion = nn.MSELoss() #loss


#FUNCTIONS

def random_normal_init(shape):
    return np.random.normal(size=shape, scale=0.05)

def xavier_init(shape):
    epsilon = np.sqrt(6/sum(shape))
    return np.random.uniform(low=-epsilon, high=epsilon, size=shape)

def newton_function(M):
    preds = np.matmul(x_enlarged,M)
    return np.mean((preds-y.reshape((n*dimy)))*(preds-y.reshape((n*dimy))))

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(dimx, h)
        self.fc2 = nn.Linear(h, dimy)

    def forward(self, x):
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

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

print('SGD with momentum using the Pytorch package')

net = Net()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=alpha)

losses_gd = []
delta1 = 0
delta2 = 0

for i in range(epochs):

    print('Epoch: {}'.format(i))
    avg_loss = []
    np.random.shuffle(x)

    for j in range(int(n/bs)):

        batch = torch.from_numpy(x[j*bs:(j+1)*bs])
        batch = batch.type(torch.FloatTensor)
        labels = torch.from_numpy(y[j*bs:(j+1)*bs])
        labels = labels.type(torch.FloatTensor)

        batch, labels = Variable(batch), Variable(labels)

        outputs = net(batch)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        avg_loss.append(loss.data[0])
        
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

