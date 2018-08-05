from parallel_base import Parallel_base
import numpy as np
import time

class SVM(Parallel_base):
    
    optimizer_check_list = ['BGD','SGD','ASGD']
    
    def __init__(self, optimizer = 'SGD', learning_rate = 0.0001, train_type = 'Parallel', iteration = 1000):
        
        optimizer_check_list = ['BGD','SGD','ASGD']
        super().__init__(learning_rate = learning_rate, train_type = train_type, iteration = iteration)
        if optimizer not in optimizer_check_list:
            raise ValueError('The value of optimizer is supposed to be BGD,SGD or ASGD, {} is given!'.format(optimizer))
        
        self.optimizer = optimizer
    
    
    def accuracy(self, X, y):
        error = 0
        for i in range(X.shape[0]):
            if y[i]*np.dot(self.w,X[i]) < 0:
                error +=1
        correct = X.shape[0] - error
        return correct/X.shape[0]
        
    def compute_loss(self, X, y, eta = 1):
        '''
        Computing loss according to Hinge loss
        
        Args:
            X: Features, shape = (# of samples, # of features)
            y: Target, shape = (1, # of samples)
            eta: coefficient

        Returns: float

        '''
        
        loss = 0
        for i in range(X.shape[0]):
            if (y[i] * np.dot(X[i], self.w)) < 1:
                loss += 0.5*eta*np.linalg.norm(self.w) + (1-y[i]*np.dot(self.w,X[i]))
            else:
                loss += 0.5*eta*np.linalg.norm(self.w)
        return loss   
    
    def SGD(self, X, y, C = 1):
 
        for i, x in enumerate(X):
            
            if (y[i]*np.dot(X[i], self.w)) < 1:
                
                self.w = self.w + C * ( (X[i] * y[i]) + (-2 * self.learning_rate * self.w) )
                
            else:
                
                self.w = self.w + C * (-2 * self.learning_rate * self.w)

                    
    def BGD(self, X, y, C = 1):
        
        grad = 0
        for i, x in enumerate(X):
            if (y[i]*np.dot(X[i], self.w)) < 1:
                grad += ((2 * self.learning_rate * self.w) - (X[i] * y[i]))
            else:
                grad += (2 * self.learning_rate * self.w)
        self.w = self.w - C * grad
        
    
    def compute_gradients(self, X, y):
        '''
        Compute gradients give a parameter list.
        '''
        grad = 0
        for i, x in enumerate(X):
            if (y[i]*np.dot(X[i], self.w)) < 1:
                grad += self.w - C * y[i] * X[i]
            else:
                grad += self.w
        return grad

    def fit(self, X, y):
        '''
        Implement SVM, loss = hinge, optimizer: BGD,SGD, adaptive SGD

        
        Args:
            X: Features, shape = (# of samples, # of features)
            y: Target, shape = (1, # of samples)
            optimizer: 'SGD','BGD','ASGD'

        Returns: object

        '''
        tic = time.time()
        
        if self.train_type == 'Parallel':
            
            self.w = self.w_hat #updata local w by aggregator 
            
            count = 0
            for i in range(self.torque): 
                loss = self.compute_loss(X, y)
                accuracy = self.accuracy(X, y)
                self.history.append([loss,accuracy,str(self.t)])#record loss history from t=1, t=0(w intialization) not included 
                self.t += 1
                if self.optimizer == 'SGD':
                    self.SGD(X, y)
                elif self.optimizer == 'BGD':
                    self.BGD(X, y)
                elif self.optimizer == 'ASGD':
                    self.w = ASGD(X, y, self.w, eta = 1)

                
                count += 1
                if count < self.torque:
                    self.w_hat = self.w
                elif count == self.torque:
                    '''
                    #self.grad saved for belta computation. 
                    #It denotes in time t(update time), the gradient of local loss of local parameters
                    
                    '''
                    self.grad = compute_gradients(X,y)  

        else:
            self.w = np.zeros(X.shape[1]) #updata local w by aggregator 
            for i in range(self.iteration):
                loss = self.compute_loss(X, y)
                accuracy = self.accuracy(X, y)
                self.history.append([loss,accuracy])#record loss history from t=1, t=0(w intialization) not included 
                if self.optimizer == 'SGD':
                    self.SGD(X, y)
                elif self.optimizer == 'BGD':
                    self.BGD(X, y)
                elif self.optimizer == 'ASGD':
                    self.w = ASGD(X, y, self.w, eta = 1)


            
                
                
        toc = time.time()
        self.resource = toc - tic
        
        return self