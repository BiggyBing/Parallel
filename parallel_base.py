import numpy as np
import time


class Parallel_base():

    def __init__(self, learning_rate = 0.0001, train_type = 'Parallel', iteration = 1000):
        
        '''
        Initialization of object
        
        Args:
            train_type: Two training type of 'Parallel' or 'Center', other value would raise ValueError.
                        In 'Parallel' mode, aggregator would initialize and sometimes update parameters.
                        In 'Center' mode, parameters are initialized and updated locally.
            iteration: Iteration means iteration number, it works when train_type is 'Center'.
            
            learning_rate: means learning rate

        Returns: No return
        
        Raises:
          ValueError: If train_type is neither 'Parallel' nor 'Center'

        '''
        if train_type != 'Parallel' and train_type != 'Center':
            raise ValueError('train_type must be \'Parallel\' or \'Center\', \'%s\' is given!'%train_type)
        
        self.iteration = iteration
        self.train_type = train_type
        self.t = 0
        self.t_0 = 0
        self.grad = 0
        self.learning_rate = learning_rate
        self.torque = 0
        self.belta = 0
        self.resource = 0
        self.grad_t0 = 0
        self.history = []
        self.w = 0
        self.w_hat = 0
        self.w_t0 = 0
        
        
    
#    def compute_loss(self,feature, target):
#        
#        m = len(target)
#        sum_of_square_errors = np.square(np.dot(feature, self.w)-target).sum()
#        cost = sum_of_square_errors/(2*m)
#    
#        return cost
        
    def Rec_from_Agg(self, w_global, torque_global):
        self.w_t0 = w_global
        self.w_hat = w_global
        self.torque = torque_global
        
    def Snd_to_Agg(self):
        if self.t_0 > 0:
            return w,self.resource, self.belta, self.grad_t0
        else:
            return w,self.resource
    
    def Est_Resource(self):
        return self.resource

    def get_coef(self):
        return self.w
    
    def set_coef(self, w_global):
        self.w = w_global
        
    def Est_Belta(self ,X, y):
        grad_global_parameter = np.dot((np.dot(X ,self.w_hat)-y), X)# In time t, the gradient of local loss of global parameters
        self.grad_t0 = grad_global_parameter
        self.belta = np.linalg.norm(self.grad - grad_global_parameter)/np.linalg.norm(self.w - self.w_hat)
        
    def time_record(self):
        self.t_0 = self.t
        return self
    
    def predict(self, X):
        
        return np.dot(X, self.w)