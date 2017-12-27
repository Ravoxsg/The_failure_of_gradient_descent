# The-failure-of-gradient-descent
Playing around with gradient descent and its limits.

Following Ali Rahimi's talk at NIPS 2017 (https://www.youtube.com/watch?v=Qi1Yry33TQE), I decided to explore the limits of gradient descent (GD) on ill-conditioned problems.
I take the same use case that he mentions: a simple one hidden layer neural network, with no non-linearities. This is equivalent to two consecutive matrix multiplications.
I compare here GD versus Newton's method, with 3 different ways to do backpropagation:
_Manually
_with Pytorch
_with Tensorflow
