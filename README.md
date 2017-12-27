# The-failure-of-gradient-descent
Playing around with gradient descent and its limits.

Following Ali Rahimi's talk at NIPS 2017 (https://www.youtube.com/watch?v=Qi1Yry33TQE), I decided to explore the limits of gradient descent (GD) on ill-conditioned problems.

I take the same use case that he mentions: a simple one hidden layer neural network, with no non-linearities. This is equivalent to two consecutive matrix multiplications.

Data is generated randomly. There are 600 datapoints, each of dimension 13. Labels are in dimension 4. The (13,4) matrix mapping datapoints to labels that we are trying to learn is also generated randomly, with a Gaussian distribution.

I compare here GD versus Newton's method, with 3 different ways to do backpropagation:\
_manually\
_with Pytorch\
_with Tensorflow
