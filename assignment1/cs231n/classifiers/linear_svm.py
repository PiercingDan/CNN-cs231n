import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  #dW = np.zeros(W.shape) # initialize the gradient as zero
      
  #This doesn't account for the bias so I modified the code to manually add the constant
  #X = np.hstack((X, np.ones((X.shape[0], 1))))
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  dloss = np.zeros(W.shape) #gradient of loss, it is real-value function of matrix W so its gradient is put in the shape of a matrix. Note this does not mean that loss is a mult-valued function, we simply adopt a matrix structure to the gradient
  #loss is composed of many loss_i for each training exampleso we add dloss together in same way when we iterate over i
  
  #For some strange reason W has the transpose shape of what it says in the notes so we modify the columns rather than the rows one at a time
  for i in xrange(num_train):  
    dloss_i = np.zeros(W.shape) #dloss is sum of all i's 
    scores = X[i].dot(W)# + W[-1] #modified
    #print(scores.shape)
    correct_class_score = scores[y[i]]
    indicator_index = (scores - correct_class_score + 1 > 0)
    #print (indicator_index)
    for j in xrange(num_classes):
      if j == y[i]:
        #First case  
        dloss_i[:, j] = -(np.sum(indicator_index) - 1)*X[i] #jth row of dloss_i, TEMP FIX: I OMIT THE j != y_i by subtracting 1!!
        continue #exits for loop with num_classes
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        #Second case, indicator met
        dloss_i[:, j] = X[i]
   
    dloss += dloss_i
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dloss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W) #element-wise multiplication 
  dloss += reg * W
  
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss, dloss 


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  #loss = 0.0

  #X = np.hstack((X, np.ones((X.shape[0], 1)))) #adding the bias "1" term
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  num_pixels = X.shape[1] # = W.shape[1]
  scores = X.dot(W) #big matrix
  correct_class_score = scores[range(num_train), y] #column vector (500, 1)
  #print(correct_class_score.shape)
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  #loss has two double sums, represented here by 2d sum of a booean matrix, margin_matrix
  margin_matrix = scores.T - correct_class_score + 1
  
  margin_matrix[y, range(num_train)] = 0 #Setting the correct match to 0 instead of one
  margin_index = (margin_matrix > 0).astype(float)
  loss = np.sum(margin_matrix[margin_matrix > 0])/num_train + 0.5 * reg * np.sum(W * W)

  #Broadcasting into 3D is still pretty slow but it seems to be the only way
  #dloss = np.zeros((num_classes, num_train, num_pixels))
  margin_index[y, range(num_train)] = -np.sum(margin_index, axis=0)
  #print(margin_index.sum())
  #print(margin_index)
  dloss = X * (margin_index)[:, :, np.newaxis] #this is part is slow, about 0.10s

  #dloss[y, range(num_train), :] = -np.sum(margin_index, axis=0)[:, np.newaxis] * X
  
  
  dloss = np.sum(dloss, axis=1).T/num_train + reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dloss
