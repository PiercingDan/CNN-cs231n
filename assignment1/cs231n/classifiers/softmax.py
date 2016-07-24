import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dloss = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1] 
  num_pixels = W.shape[0] 
  scores = X.dot(W) #dim = (N, C) 
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #numerical stability so I will choose 'C', see notes, as e^-max(scores for that training example), shift all scores by the max in their row
  #print(scores)
  max_score = np.max(scores, axis=1)
  #print(max_score)
  scores += -max_score[:, np.newaxis]
  
  num_train = X.shape[0]
  num_classes = W.shape[1] 
  num_pixels = W.shape[0] 
  scores = X.dot(W) #dim = (N, C) 

  for i in range(num_train):
    bot_sum = 0 #bottom sum
    for j in range(num_classes):
        if j == y[i]: #correct class
            loss += -scores[i, j]
        bot_sum += np.e**scores[i, j]
    loss += np.log(bot_sum)

    for j in range(num_classes):
        dloss[:, j] += np.e**scores[i, j]/bot_sum * X[i]
    dloss[:, y[i]] += -X[i]

  loss /= num_train
  dloss /= num_train

  loss += 0.5*reg*np.sum(W * W)
  dloss += reg*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dloss


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0

  num_train = X.shape[0]
  num_classes = W.shape[1] 
  num_pixels = W.shape[0] 
  scores = X.dot(W) #dim = (N, C) 
  scores -= np.max(scores, axis=1)[:, np.newaxis] #normalize to avoid large numbers
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  loss = np.sum(np.log(np.sum(np.e**scores, axis=1)) - scores[range(num_train), y])/num_train + 0.5*reg*np.sum(W*W)

  #X_3d = np.tile(X, (1, 1, num_classes)) #(N, D, C) X_3d identical along D axis
  
  #scores_3d identical along N axis
  bot_sum = np.sum(np.e**scores, axis=1) #(N)
  B = np.e**scores.T/bot_sum#intemediary product I'll call B (C, N)
  B[y, range(num_train)] += -1 #subtracting 1 from correct class entries
  dloss_3d = X * B[:, :, np.newaxis] #(C, N, D) broadcast
  dloss = np.sum(dloss_3d, axis=1).T/num_train + reg*W #(D, C) 
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dloss

