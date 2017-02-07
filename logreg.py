def logLikelihood(theta, X, y,regTerm=0.0):
    """
    calculates the cost function
    param theta:initial guess
    param X : a feature matrix
    param y: corresponding labels
    param regTerm : regularization parameter
    returns loglikelihood for a binary classifier
    """
    m = X.shape[0]  
    p = sigmoid(X,theta)
    
    log_loss = -np.mean(y*np.log(p) + (1-y)*np.log(1-p))
    J = log_loss + (regTerm * np.linalg.norm(theta[1:]) ** 2 / (2*m))
    return J    
    #return(-(np.sum(y*np.log(sigmoid(X, theta)) + (1-y)*(np.log(1-sigmoid(X, theta))))))

def gradient(theta, X, y,regTerm=0.0):
    """
    calculates jacobian of the cost function
    param theta:initial guess
    param X : a feature matrix
    param y: corresponding labels
    param regTerm : regularization parameter
    returns gradient vector of the cost function
    """
    (m,n) = X.shape
    grad1 = (1.0/m)*np.dot(X[:,0].T,sigmoid(X , theta)-y)
    grad2 = (1.0/m)*(np.dot(X[:,1:].T,sigmoid(X , theta)-y) + (regTerm*theta[1:]))
    out = np.zeros(n)
    out[0] = grad1
    out[1:] = grad2
    return out

def hessian(theta,X,y,regTerm=0.0):
    """
    calculates hessian of the cost function
    param theta:initial guess
    param X : a feature matrix
    param y: corresponding labels
    param regTerm : regularization parameter
    returns Hessain matrix
    """
    (m,n) = X.shape
    tmp = (regTerm/m)*np.identity(n)
    tmp[0][0] = 0.      #no regularization for first term
    mat1 = np.dot(X.T,np.diag(np.multiply(sigmoid(X,theta),1-sigmoid(X,theta))))
    mat2 = (1.0/m)*np.dot(mat1,X)
    return mat2 + tmp

def train_model(theta, X, y, converge_change,regTerm=0.0):
    """train the model using Newton-Raphson algorithm
        param theta:initial guess
        param X : a feature matrix
        param y: corresponding labels
        param converge_change
        param regTerm : regularization parameter
        returns trained model""" 
          
    cost = logLikelihood(theta, X, y,regTerm)
    G = 1
    while G > converge_change:
        old_cost = cost
        theta = theta - (np.dot(np.linalg.inv(hessian(theta,X,y,regTerm)), gradient(theta, X, y,regTerm)))
        cost = logLikelihood(theta, X, y,regTerm)
        G = old_cost - cost
    return theta  


def test_model(X_test, model):
    """ test a model 
        param X_test: a feature matrix
        param model: trained logistic regression model
        return : predicted labels for test data
    """
    probability = sigmoid(X_test,model)

    return [1 if x >= 0.5 else 0 for x in probability]

if __name__ == "__main__":

    # prepare training data
    m= X_train.shape[0]  #number of examples
    n= X_train.shape[1]   #number of features
    X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)

    #intercept term
    X_train = np.hstack((np.ones((m, 1)), X_train))

    initial_theta = np.zeros(n+1)
    regTerm = 1.0
    converge_change = 0.000001
    sol_matrix = train_model(initial_theta,X_train,y_train,converge_change,regTerm)
    # predict
    X_test = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)  
    X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
    y_test = test_model(X_test, sol_matrix)
