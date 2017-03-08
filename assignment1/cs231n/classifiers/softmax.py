import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[1]
  num_class = W.shape[0]
  for i in xrange(num_train):
    scores=W.dot(X[:,i])
    scores-=np.max(scores)
    normalization_score = np.exp(scores)/np.sum(np.exp(scores))
    loss+=-np.log(normalization_score[y[i]])

  #calculate the dW
    for j in xrange(num_class):
      dW[j,:]+=(normalization_score[j]-(y[i]==j))*X[:,i]

  loss/=num_train
  loss+=0.5*reg*np.sum(W * W)

  dW/=num_train
  dW+=reg*W



  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_train=X.shape[1]
  num_class=W.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  f=W.dot(X)
  log_c=np.max(f)
  fixed_f=f-log_c #avoid the numerical overflow
  exp_f=np.exp(fixed_f)
  sum_f=np.sum(exp_f,axis=0)
  normalization_score=exp_f/sum_f
  L=-np.log(normalization_score[y,xrange(num_train)]) # the loss of each sample Xi is stored in L
  loss=np.sum(L)
  loss/=num_train
  loss+=0.5*reg*np.sum(W*W)  

  # calculate the gradient dW
  template=np.zeros((num_class,num_train))
  template[y,xrange(num_train)]=1
  dW=(normalization_score-template).dot(X.T)
  dW/=num_train
  dW+=reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
