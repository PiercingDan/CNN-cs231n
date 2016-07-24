import numpy as np
import matplotlib.pyplot as plt

#sigmoid
#f = lambda x: x [x<0] = 0#1.0/(1.0 + np.exp(-x))
def f(x):
    #ReLU
    x[x<0] = 0
    return x

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape
    _, H = W1.shape

    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    #X_1 = X , X_2 = X notation, I compute it at this stage because I will need it later for grad check
    X_2 = X.dot(W1) + b1 #(N, H)
    #print (X_2.shape)
    scores = f(X_2).dot(W2) + b2 #(N, C)
    #Numerically stabilizing scores
    scores -= np.max(scores, axis=1)[:, np.newaxis] #(N,C) - N
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    # Softmax loss
    loss = None
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss. So that your results match ours, multiply the            #
    # regularization loss by 0.5                                                #
    #############################################################################
    loss = np.sum(np.log(np.sum(np.e**scores, axis=1)) - scores[range(N), y])/N + 0.5*reg*np.sum(W1*W1) + 0.5*reg*np.sum(W2*W2)
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    #copy from softmax.py svm_loss_vectorized, but this biasand weights are separate here!
    #instead of scores, use intermediate result X_2, hidden later
    # REMEMBER D-->H
    # X_2 (N, H)
    bot_sum = np.sum(np.e**scores, axis=1) #(N)
    B = np.e**scores.T/bot_sum#intemediary product I'll call B (C, N)
    # print (B.shape)
    B[y, range(N)] += -1 #subtracting 1 from correct class entries
    dloss_3d = X_2 * B[:, :, np.newaxis] #(C, N, H) broadcast
    #print (dloss_3d.shape)
    grads['W2'] = np.sum(dloss_3d, axis=1).T/N + reg*W2 #(H, C) 
    grads['b2'] = np.sum(B, axis=1)/N #(C, N)
    """
    grads['W1'] = np.zeros(W1.shape)
    grads['b1'] = np.zeros(b1.shape)
    #This is pretty complicated so I'll do for loop first
    for i in range(N): #iterate over training examples
        grad_W1 = np.zeros(W1.shape) # (D, H)
        grad_b1 = np.zeros(b1.shape) # (H)
        for j in range(H): #iterate over columns of W_1
            #split it into 3 parts as described in my notes
            d1 = (X_2[i, j] > 0) * X[i]#(D) 
            d2 = W2.T[:, j]#(C) 
            d3 = np.exp(scores[i, :])/bot_sum[i]#(C)
            d3[y[i]] -= 1. #subtract 1 for correct class

            grad_W1[:, j] = d3.dot(d2)*(d1) #(1, D) Setting the columns of grad
            grad_b1[j] = d3.dot(d2)*(X_2[i, j] > 0)
      
        d0 = (X_2[i] > 0) #(H)
        d1 = (X_2[i] > 0)*X[i][:, np.newaxis] #(H)*(D, 1)=(D, H) 
        d2 = W2 #(H,C)
        d3 = np.exp(scores[i, :])/bot_sum[i] #(C)
        #np.exp(scores)/bot_sum[:, np.newaxis]# (N, C) / (N, 1) = (N, C)
        d3[y[i]] -= 1
        

        #print(d1.shape, d2.shape, d3.shape)
        grad_W1 = d1*(d2.dot(d3)) #(D, H)*((H, C)dot(C)) = (D, H)
        grad_b1 = d0*d2.dot(d3) #H*((H, C)dot(C))
        #grad_b1 = 0 (N, H)
        grads['W1'] += grad_W1
        grads['b1'] += grad_b1

        #if i % 1000 == 0:
        #    print(float(i)/N*100)
    """
    #No Loop (will involve broadcasting and an axis sum
    d0 = (X_2 > 0) #(N, H)

    #My memory sucks 4gb free ram with everything loaded
    d2 = W2.T #(H, C).T = (C, H)
    d3 = B.T #(C, N)

    try: 
        d1 = d0 * X.T[:, :, np.newaxis] #(N, H) * (D, N, 1) = (D, N, H) #Memory breaking operation
        grads['W1'] = np.sum(d1 * d3.dot(d2), axis=1)/N + reg * W1 #(D, N, H)* (N, C)dot(C, H) = (D, N, H) -sum axis 1 -> (D, H)

    except MemoryError:
        grads['W1'] = np.empty(W1.shape)
        step = 50
        #Partition along d variable, only affects the computation at one place 
        for d in range(0, D, step): 
            d1 = d0 * X.T[d: d + step, :, np.newaxis]
           # print((np.sum(d1 * d3.dot(d2), axis=1)).shape)
            grads['W1'][d: d + step, :] = np.sum(d1 * d3.dot(d2), axis=1)
        
        grads['W1'] /= N 
        grads['W1'] += reg * W1 

    grads['b1'] = np.sum(d0 * d3.dot(d2), axis=0)/N

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      sample_index = np.random.choice(range(num_train), size=batch_size)
      X_batch = X[sample_index]
      y_batch = y[sample_index]

                 #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)
      if loss > 1e7: #time to quit
        #print (loss)
         print('it = ', it)
         print ('I quit: loss over 10 million')
         break

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      self.params['W1'] -= learning_rate * grads['W1']
      self.params['b1'] -= learning_rate * grads['b1']
      self.params['W2'] -= learning_rate * grads['W2']
      self.params['b2'] -= learning_rate * grads['b2']
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################
      if verbose and it % 100 == 0:
        print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

    # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
      # Check accuracy
        train_acc = np.mean((self.predict(X_batch) == y_batch))
        val_acc = np.mean((self.predict(X_val) == y_val))
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

      # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    y_pred = np.argmax(f(X.dot(W1) + b1).dot(W2) + b2, axis=1)

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


