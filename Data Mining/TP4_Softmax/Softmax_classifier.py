from __future__ import print_function

import numpy as np
import sys



if sys.version_info >= (3, 0):
    def xrange(*args, **kwargs):
        return iter(range(*args, **kwargs))


class SoftmaxClassifier(object):

    def __init__(self):
        self.W = None
   
    def softmax(self, X, y, W):
        """
        softmax function
        
        Inputs:
        - X: A numpy array of shape (N, D) containing a minibatch of N
          data points; each point has dimension D.
        - y: labels of shape (N, 1)
        - W: initial random weights
        Returns:
        - the softmax function
        """
      #############################################################################
      # TODO: Compute the softmax function
      #############################################################################

        nom = np.exp(np.dot(X, W))
        denom = np.sum(np.exp(np.dot(X, W)), axis=1)
        denom = np.repeat(denom, len(np.unique(y))).reshape((X.shape[0], len(np.unique(y))))
        softmax = nom/denom


      #############################################################################
      #                          END OF YOUR CODE                                 #
      #############################################################################
        return softmax
    
    def softmax_loss(self, X, y, W, reg):
        """
        Compute the loss function of the softmax classifier with L2 norm regulariser.
        Inputs:
        - X: A numpy array of shape (N, D) containing a minibatch of N
          data points; each point has dimension D.
        - y: A numpy array of shape (N,) containing labels for the minibatch.
        - reg: (float) the parameter of the regularizer.
        Returns:
        - loss with l2 regulariser as a single float
        - when reg= 0 returns the loss function of the softmax classifier without regulariser 
        """
      #############################################################################
      # TODO: Compute the softmax function
      #############################################################################

        num_classes = 3
        nom = np.exp(np.dot(X, W))
        denom = np.sum(np.exp(np.dot(X, W)), axis=1)
        denom = np.repeat(denom, W.shape[1]).reshape((X.shape[0], W.shape[1]))
        labels = np.repeat(y, num_classes).reshape(len(y), num_classes)
        classes = np.tile(np.arange(num_classes), (len(y), 1))
        indicator = (labels == classes)
        calc_loss = (-1/X.shape[0])*np.sum(indicator*np.log(nom/denom), axis=1).sum()

    #############################################################################
    #                          END OF YOUR CODE                                 #
    ############################################################################
      
        return calc_loss
    
    def softmax_loss_gradient(self, X, y, W, reg):
        """
        Compute the gradient of the loss function of the softmax classifier with L2 norm regulariser.
        Inputs:
        - X: A numpy array of shape (N, D) containing a minibatch of N
          data points; each point has dimension D.
        - y: A numpy array of shape (N,) containing labels for the minibatch.
        - reg: (float) the parameter of the regularizer.
        Returns:
        - gradient with respect to self.W; an array of the same shape as W
        """
      #############################################################################
      # TODO: Compute the softmax function
      #############################################################################

        num_classes = 3
        nom = np.exp(np.dot(X, W))
        denom = np.sum(np.exp(np.dot(X, W)), axis=1)
        denom = np.repeat(denom, W.shape[1]).reshape((X.shape[0], W.shape[1]))
        labels = np.repeat(y, num_classes).reshape(len(y), num_classes)
        classes = np.tile(np.arange(num_classes), (len(y), 1))
        indicator = (labels == classes)


        grad = np.zeros((W.shape[0], W.shape[1]))
        const = (-1/X.shape[0])

        for i in range(0, X.shape[1]):
            grad[i] = const*np.sum(np.repeat(X[:,i], num_classes).reshape((X.shape[0], num_classes))*(indicator - nom/denom), axis=0)
        

        grad = grad + reg*W
    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
        return grad





    
    def train(self, X_train, y_train, learning_rate=0.001, reg=0, num_iters=1,
              batch_size=128):
        """
        Train softmax classifier using stochastic gradient descent.
        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c
          means that X[i] has label 0 <= c < C for C classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) the parameter of the regularizer.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - initial weights W: array of (N, D) with initial random weights
        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        num_train, dim = X_train.shape
        num_classes = np.max(y_train) + 1  
        
        if self.W is None:
            # initialize W
            self.W = 0.001 * np.random.randn(dim, num_classes)
        

        # Run stochastic gradient descent to optimize W
        loss_history = []
        for it in xrange(num_iters):


            #########################################################################
            # TODO:                                                                 #
            # Sample batch_size elements from the training data and their           #
            # corresponding labels to use in this round of gradient descent.        #
            # Store the data in X_batch and their corresponding labels in           #
            # y_batch; after sampling X_batch should have shape (dim, batch_size)   #
            # and y_batch should have shape (batch_size,)                           #
            #                                                                       #
            # Hint: Use np.random.choice to generate indices. Sampling with         #
            # replacement is faster than sampling without replacement.              #
            #########################################################################

            ind = np.random.choice(X_train.shape[0], batch_size, replace=True)
            X_batch = X_train[ind]
            y_batch = y_train[ind]





            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            # evaluate loss and gradient
            loss = self.softmax_loss(X_batch, y_batch, self.W, reg)
            grad = self.softmax_loss_gradient(X_batch, y_batch, self.W, reg)
            loss_history.append(loss)


            # 
            #########################################################################
            # TODO:                                                                 #
            # perform parameter update using the gradient and the learning rate.   #
            #########################################################################
            
            #Stopping criterion
            #if ((loss_history[it] - loss_history[it-1]) < 10**-5) and it >= 2:
            #    print("STOPPED")
            #    break

            self.W -= learning_rate*grad
            #print(self.W)
            

            

            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################


        return loss_history, self.W

    def predict(self, X_train, y_train, W_opt):
        """
    Use the trained weights of this linear classifier to predict labels for
    data points.
    Inputs:
    - X: A numpy array of shape (N, D) containing training data; there are N
      training samples each of dimension D.
    - y_train: vector of lables of shape (N, 1)
    Returns:
    - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
      array of length N, and each element is an integer giving the predicted
      class.
    """
        #y_pred = np.zeros(X_train.shape[0])
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################


        y_pred = np.argmax(self.softmax(X_train, y_train, W_opt), axis=1)

        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred

 


