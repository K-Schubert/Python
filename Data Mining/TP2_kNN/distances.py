#TP2_SCHUBERT_Kieran

import numpy as np


def compute_euclidean_dist_two_loops(x_train, x_test):
    """
    Compute the distance between each test point in x_test and each training point
    in x_train using a nested loop over both the training data and the
    test data.

    Inputs:
    - x_train: A numpy array of shape (num_train, D) containing test data.
    - x_test: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """
    num_test = x_test.shape[0]
    num_train = x_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
        for j in range(num_train):
        #####################################################################
        # TODO:                                                             #
        # Compute the l2 distance between the ith test point and the jth    #
        # training point, and store the result in dists[i, j]. You should   #
        # not use a loop over dimension.                                    #
        #####################################################################
        # Your code
            dists[i, j] = np.sqrt(np.square(x_train[j,:] - x_test[i,:]).sum())
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
    return dists




def compute_euclidean_dist_one_loop(x_train, x_test):
    """
    Compute the distance between each test point in x_test and each training point
    in x_train using a single loop over the test data.

    Input / Output: Same as compute_euclidean_dist_two_loops
    """
    num_test = x_test.shape[0]
    num_train = x_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
      #######################################################################
      # TODO:                                                               #
      # Compute the l2 distance between the ith test point and all training #
      # points, and store the result in dists[i, :].                        #
      #######################################################################
      # Your code
        dists[i, :] = np.sqrt(np.sum(np.square(x_train[:,:] - x_test[i,:]), axis=1))



      #######################################################################
      #                         END OF YOUR CODE                            #
      #######################################################################
    return dists

def compute_euclidean_dist_no_loops(x_train, x_test):
    """
    Compute the distance between each test point in x_test and each training point
    in x_train using no explicit loops.

    Input / Output: Same as compute_euclidean_dist_two_loops
    """
    num_test = x_test.shape[0]
    num_train = x_train.shape[0]
    dists = np.zeros((num_test, num_train))
    #########################################################################
    # TODO:                                                                 #
    # Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    #                                                                       #
    # You should implement this function using only basic array operations; #
    # in particular you should not use functions from scipy.                #
    #                                                                       #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################
    # Your code

    dists = -2 * np.dot(x_test, x_train.T) + np.sum(x_train**2, axis=1) + np.sum(x_test**2, axis=1)[:, np.newaxis]
    dists = np.sqrt(dists)

    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    return dists

def compute_mahalanobis_dist(x_train, x_test, sigma):
    """
    Compute the Mahalanobis distance between each test point in x_test and each training point
    in x_train (please feel free to choose the implementation).

    Inputs:
    - x_train: A numpy array of shape (num_train, D) containing test data.
    - x_test: A numpy array of shape (num_test, D) containing test data.
    - sigma: A numpy array of shape (D,D) containing a covariance matrix.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Mahalanobis distance between the ith test point and the jth training
      point.
    """
    num_test = x_test.shape[0]
    num_train = x_train.shape[0]
    dists = np.zeros((num_test, num_train))
    #########################################################################
    # TODO:                                                                 #
    # Compute the Mahalanobis distance between all test points and all      #
    # training points (please feel free to choose the implementation), and store the       #
    # result in dists.                                                      #
    #                                                                       #
    #########################################################################
    # Your code

    a = (x_test[:, np.newaxis] - x_train).reshape(-1, x_test.shape[1])
    sigma_inv = np.linalg.inv(sigma)
    b = np.einsum('ij,jk->ik', a, sigma_inv)
    dists = np.sqrt(np.diag(np.dot(a, b.T))).reshape((x_test.shape[0], x_train.shape[0]))


    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    return dists





def define_sigma(X, method):
    """
    Define a covariance  matrix using 3 difference approaches:
    """
    d = X.shape[1]

    #########################################################################
    # TODO:                                                                 #
    # Computre Σ as a diagonal matrix that has at its diagonal the average  #
    # variance of  the different features,
    #  i.e. all diagonal entries Σ_ii will be the same                      #
    #                                                                       #
    #########################################################################
    # Your code
    if method == 'diag_average_cov':
        sigma = (np.var(X, axis=0)).mean()*np.identity(X.shape[1])



    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################


    #########################################################################
    # TODO:                                                                 #
    # Computre  Σ as a diagonal matrix that has at its diagonal             #
    # he variance of eachfeature, i.e.σ_k
    #                                                                       #
    #########################################################################
    # Your code
    elif method == 'diag_cov':
        sigma = np.var(X, axis=0)*np.identity(X.shape[1])



    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################


    #########################################################################
    # TODO:                                                                 #
    # Computre Σ as the full covariance matrix between all pairs of features#
    #                                                                       #
    #########################################################################

    # Your code
    elif method == 'full_cov':
        sigma = np.cov(X, rowvar=False)


    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    return sigma
