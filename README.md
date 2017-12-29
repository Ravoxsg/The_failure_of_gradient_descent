# The-failure-of-gradient-descent
Playing around with gradient descent and its limits.

Following Ali Rahimi's talk at NIPS 2017 (https://www.youtube.com/watch?v=Qi1Yry33TQE), I decided to explore the limits of gradient descent (GD) on ill-conditioned problems.

I take the same use case that he mentions: a simple one hidden layer neural network, with no non-linearities. This is equivalent to two consecutive matrix multiplications.

Data is generated randomly. There are 600 datapoints, each of dimension 13. Labels are in dimension 4. The (13,4) matrix (named A in the code) mapping datapoints to labels that we are trying to learn is also generated randomly, with a Gaussian distribution. A parameter controls named "ev" controls the weight that we add to the first singular value of this matrix. This weight represents the ill-condition of the problem. A heavy such weigth will lead to GD needing a smaller and smaller learning rate to converge, if it converges at all. 

The trick here to play on singulat values is to write A in its singular value decomposition: A = U.D.V^{T}, where D is a diagonal with Gaussian values, U and V are orthogonal. Generating U and V can be done by taking the first matrix in the QR decomposition of 2 random matrices. 

I compare here GD versus Newton's method, with 3 different ways to do backpropagation:\
_manually\
_with Pytorch\
_with Tensorflow

The 3 scripts are respectively np_manual.py, pytorch.py and tensorf.py. 

I compare GD with Newton Raphson method. Since dimension is not too big here, the Hessian can easily be calculated and inversed. 

Let's take an example with Pytorch and a ev value of 1e2, which is the order of magnitude of the condition number of A. That is still quite small. 
The following shows gradient descent vs Newton's method for 2 different learning rates: 1e-6 and 1e-7. 

