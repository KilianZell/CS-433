import numpy as np

def build_poly(x, degree):
    """polynomial basis functions for input data x
    
    Args:
        x: shape=(N,D)
        degree: int
    Returns:
        the polynomial basis functions for input data x of shape=(N,D*(degree+1)).
    """
    N = x.shape[0]
    try:
        D = x.shape[1]
    except IndexError as e:
        D = 1
    res = np.zeros((N,D*degree+1))
    res[:,-1] = 1
    res[:,:D] = x            
    for i in range(1,degree):
        res[:,i*D:(i+1)*D] = res[:,(i-1)*D:i*D]*x
    return res


def add_fct(x,tx,features,fct):
    '''
    adds as a feature a function of the list of features (features) from x to tx
    
    Args:
        features: list of int: columns of the features to exp
        x:numpy array of shape=(N,D): raw data
        tx:numpy array of shape=(N,D'): augmented data
        fct: function that takes a (N,) array and returns a transformed (N,) array
    Returns:
        res: numpy array of shape=(N,D'+len(features)): (tx|fct(x[features]))
        
    ex: add_fct(tx_tr, tx_tr_poly, [0,4,8], np.exp)
        >>> numpy array (tx_tr_poly | np.exp(tx_tr[features]) of shape=(N,D'+3)
    '''
    (N,D_dash) = tx.shape
    res = np.zeros((N,D_dash+len(features)))
    res[:,:D_dash] = tx
    for i in range(len(features)):
        res[:,D_dash+i] = fct(x[:,features[i]])
    return res

def compute_loss(y, tx, w,loss_type='MSE'):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.
        loss_type: str. 'MSE', 'MAE' or 'RMSE' (=sqrt(2*MSE))
    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    if loss_type == 'MSE':
        prediction = np.dot(tx,w)
        loss = np.dot(y-prediction,y-prediction)/(2*y.shape[0])
    elif loss_type == 'MAE':
        loss = 1/y.shape[0]*np.sum(np.abs(y-np.dot(tx,w)))
    elif loss_type == 'RMSE':
        prediction = np.dot(tx,w)
        loss = np.sqrt(2/y.shape[0]*np.dot(y-prediction,y-prediction))
    else:
        raise NotImplementedError
    return loss

### GD ###
def compute_gradient(y, tx, w):
    """Computes the gradient at w.
        
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.
        
    Returns:
        An numpy array of shape (D, ) (same shape as w), containing the gradient of the loss at w.
    """
    e = y-np.dot(tx,w)
    return -1/y.shape[0]*np.dot(tx.transpose(),e)


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.
        
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
        
    Returns:
        w: array of size (D, ) . Weights of the last upadating of GD
        loss: (scalar) loss of the last GD iteration
    """
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient and loss
        grad = compute_gradient(y,tx,w)
        w = w - gamma*grad
        loss = compute_loss(y,tx,w)
        print("GD iter. {bi}/{ti}: loss={l}, w={w}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w=w))
    return w, loss

def LS_GD(y, tx, initial_w, L, max_iters):
    '''
    Line search gradient descent algorithm
    
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        L: initial lipschitz constant guess (L = 1/step-size)
        max_iters: a scalar denoting the total number of iterations of GD
        
    Returns:
        w: array of size (D, ) . Weights of the last upadating of GD
        loss: (scalar) loss of the last GD iteration
    '''
    w = initial_w
    for i in range(max_iters):
        grad = compute_gradient(y,tx,w)
        L = L/2
        while compute_loss(y,tx, w-grad/L) > compute_loss(y,tx,w) -np.dot(grad,grad)/(2*L):
            L = 2*L
        gamma = 1/L
        w = w -gamma*grad
        loss = compute_loss(y,tx,w)
        print('Iteration {nb}/{tot}. loss:{loss}, L:{L}'.format(nb=i,tot=max_iters,loss=loss, L=L))
    return w,loss

### Grid search ###
def generate_w(num_intervals):
    """Generate a grid of values for w0 and w1."""
    w0 = np.linspace(-100, 200, num_intervals)
    w1 = np.linspace(-150, 150, num_intervals)
    return w0, w1


def get_best_parameters(w0, w1, losses):
    """Get the best w from the result of grid search."""
    min_row, min_col = np.unravel_index(np.argmin(losses), losses.shape)
    return [w0[min_row], w1[min_col]], losses[min_row, min_col] 


def grid_search(y, tx, grid_w0, grid_w1,loss_type='MSE'):
    """Algorithm for grid search.
        
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        grid_w0: numpy array of shape=(num_grid_pts_w0, ). A 1D array containing num_grid_pts_w0 values of parameter w0 to be tested in the grid search.
        grid_w1: numpy array of shape=(num_grid_pts_w1, ). A 1D array containing num_grid_pts_w1 values of parameter w1 to be tested in the grid search.
        
    Returns:
        losses: numpy array of shape=(num_grid_pts_w0, num_grid_pts_w1). A 2D array containing the loss value for each combination of w0 and w1
    """

    losses = np.zeros((len(grid_w0), len(grid_w1)))
    # compute loss for each combination of w0 and w1.
    for i in range(len(grid_w0)):
        for j in range(len(grid_w1)):
            losses[i,j] = compute_loss(y,tx,[grid_w0[i],grid_w1[j]],loss_type=loss_type)
    return losses


### SGD ###
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w from just few examples n and their corresponding y_n labels.
        
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        w: numpy array of shape=(D, ). The vector of model parameters.
        
    Returns:
        A numpy array of shape (D, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """
   
    e = y-np.dot(tx,w)
    return -1/y.shape[0]*np.dot(tx.transpose(),e)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma, batch_size=1):
    """The Stochastic Gradient Descent algorithm (SGD).
            
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize
        
    Returns:
        w: array of size (D, ) . Weights of the last upadating of SGD
        loss: (scalar) loss of the last SGD iteration 
    """
    
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y,minibatch_tx in batch_iter(y,tx,batch_size=batch_size):
            grad = compute_stoch_gradient(minibatch_y,minibatch_tx,w)
            w = w-gamma*grad
        loss = compute_loss(y,tx,w)
        print("SGD iter. {bi}/{ti}: loss={l}, w={w}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w=w))
    return w, loss

def LS_SGD(y, tx, initial_w, L, max_iters,batch_size=1):
    '''
    Line search stochastic gradient descent algorithm
    
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        L: initial lipschitz constant guess (L = 1/step-size)
        max_iters: a scalar denoting the total number of iterations of GD
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
    Returns:
        w: array of size (D, ) . Weights of the last upadating of GD
        loss: (scalar) loss of the last GD iteration
    '''
    w = initial_w
    for i in range(max_iters):
        for minibatch_y,minibatch_tx in batch_iter(y,tx,batch_size=batch_size):
            grad = compute_gradient(minibatch_y,minibatch_tx,w)
            L = L/2
            while compute_loss(minibatch_y,minibatch_tx, w-grad/L) > compute_loss(minibatch_y,minibatch_tx,w) -np.dot(grad,grad)/(2*L):
                L = 2*L
        gamma = 1/L
        w = w -gamma*grad
        loss = compute_loss(y,tx,w)
        print('Iteration {nb}/{tot}. loss:{loss}, L:{L}'.format(nb=i,tot=max_iters,loss=loss, L=L))
    return w,loss

### SubGD ###
def compute_subgradient_mae(y, tx, w):
    """Compute a subgradient of the MAE at w.
        
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        w: numpy array of shape=(D, ). The vector of model parameters.
        
    Returns:
        A numpy array of shape (D, ) (same shape as w), containing the subgradient of the MAE at w.
    """
    e = y-np.dot(tx,w)
    return -1/y.shape[0]*np.dot(tx.transpose(),np.sign(e))

### Least Squares ###
def least_squares(y, tx):
    """Calculate the least squares solution, using normal equations.
       returns mse, and optimal weights.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """
    w = np.linalg.solve(np.dot(tx.transpose(),tx), np.dot(tx.transpose(),y))
    mse = compute_loss(y,tx,w,loss_type='MSE')
    return w,mse

### Ridge Regression ###
def ridge_regression(y, tx, lambda_):
    """implement ridge regression, using normal equations.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.

    """
    N,D = tx.shape[0],tx.shape[1]
    w = np.linalg.solve(np.dot(tx.transpose(),tx)+lambda_*2*N*np.eye(D),np.dot(tx.transpose(),y))
    loss = compute_loss(y,tx,w)
    return w,loss


### Logistic regression ###
def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array
    """
    return 1/(1+np.exp(-t))

def compute_loss_logistic(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N,). Binary variable in {0,1}
        tx: shape=(N, D)
        w:  shape=(D,) 

    Returns:
        res: the negative log of the logistic loss
    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    tx_w = np.dot(tx,w)
    log = np.log(1+np.exp(tx_w)).sum()
    res = 1/y.shape[0]*(-np.dot(y.transpose(),tx_w)+np.log(1+np.exp(tx_w)).sum())
    return res

def compute_gradient_logistic(y, tx, w):
    """compute the gradient of loss.
    
    Args:
        y:  shape=(N,). Binary variable in {0,1}
        tx: shape=(N, D)
        w:  shape=(D,) 

    Returns:
        a vector of shape (D, 1)
    """

    sigma = sigmoid(np.dot(tx,w))
    return 1/y.shape[0]*np.dot(tx.transpose(),sigma-y)

def logistic_regression(y, tx, initial_w, max_iters, gamma, stochastic = False, batch_size = 1):
    """Calculate the logistic regression solution, using (stochastic) gradient descent.
       returns mse, and optimal weights.

    Args:
        y:  shape=(N,). Binary variable in {0,1}
        tx: shape=(N, D)
        initial_w:  shape=(D,). Initial guess of w.
        gamma: float. Learning rate.
        stochastic: bool. True for SGD, False for GD.
        batch_size: int. The size of the batch, if stochastic is True.

    Returns:
        loss: scalar number
        w: shape=(D,) 
    """
    w = initial_w
    if stochastic:
        for n_iter in range(max_iters):
            for minibatch_y,minibatch_tx in batch_iter(y,tx,batch_size=batch_size):
                grad = compute_gradient_logistic(minibatch_y,minibatch_tx,w)
                w = w-gamma*grad
            loss = compute_loss(y,tx,w)
            print("SGD iter. {bi}/{ti}: loss={l}, w={w}".format(
                  bi=n_iter, ti=max_iters - 1, l=loss, w=w))
    else:
        for n_iter in range(max_iters):
            grad = compute_gradient_logistic(y,tx,w)
            w = w-gamma*grad
            loss = compute_loss(y,tx,w)
            print("GD iter. {bi}/{ti}: loss={l}, w={w}".format(
                  bi=n_iter, ti=max_iters - 1, l=loss, w=w))
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, stochastic = False, batch_size = 1):
    """Calculate the regularized logistic regression solution, using (stochastic) gradient descent.
       returns mse, and optimal weights.

    Args:
        y:  shape=(N,). Binary variable in {0,1}
        tx: shape=(N, D)
        lambda_: float. Regularization parameter.
        initial_w:  shape=(D,). Initial guess of w.
        gamma: float. Learning rate.
        stochastic: bool. True for SGD, False for GD.
        batch_size: int. The size of the batch, if stochastic is True.

    Returns:
        loss: scalar number
        w: shape=(D,) 
    """
    w = initial_w
    if stochastic:
        for n_iter in range(max_iters):
            for minibatch_y,minibatch_tx in batch_iter(y,tx,batch_size=batch_size):
                grad = compute_gradient_logistic(minibatch_y,minibatch_tx,w)+2*lambda_*w
                w = w-gamma*grad
            loss = compute_loss(y,tx,w)
            print("SGD iter. {bi}/{ti}: loss={l}, w={w}".format(
                  bi=n_iter, ti=max_iters - 1, l=loss, w=w))
    else:
        for n_iter in range(max_iters):
            grad = compute_gradient_logistic(y,tx,w)+2*lambda_*w
            w = w-gamma*grad
            loss = compute_loss(y,tx,w)
            print("GD iter. {bi}/{ti}: loss={l}, w={w}".format(
                  bi=n_iter, ti=max_iters - 1, l=loss, w=w))
    return w, loss
