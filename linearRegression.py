from parallel_base import Parallel_base

class LinearRegression(Parallel_base):
    
    def compute_loss(self, X, y):
        '''
        Computing loss according to OLS
        
        Args:
            X: Features, shape = (# of samples, # of features)
            y: Target, shape = (1, # of samples)

        Returns: loss

        '''
        
        m = len(target)
        sum_of_square_errors = np.square(np.dot(feature, self.w)-target).sum()
        cost = sum_of_square_errors/(2*m)
    
        return cost
        

    def fit(self, X, y):
        '''
        Implement l2-linear regression by Gradient descent.
        In 'Parallel' mode, aggregator would initialize and sometimes update parameters.(w is derived from w_hat)
        In 'Center' mode, parameters are initialized and updated locally.
        
        Args:
            X: Features, shape = (# of samples, # of features)
            y: Target, shape = (1, # of samples)

        Returns: object

        '''
        tic = time.time()
        
        if self.train_type == 'Parallel':
            self.w = self.w_hat #updata local w by aggregator 
            
            count = 0
            tic = time.time()
            for i in range(self.torque): 
                count += 1
                loss = self.compute_loss(X, y)
                self.history.append([loss,str(self.t)])#record loss history from t=1, t=0(w intialization) not included             
                self.t += 1
                grad = np.dot((np.dot(X,self.w)-y), X) + 0.1 * self.w
                self.w = self.w - self.learning_rate * grad
                if count < self.torque:
                    self.w_hat = self.w
                elif count == self.torque:
                    '''
                    self.grad saved for belta computation. 
                    It denotes that in time t(update time), the gradient of local loss of local parameters
                    '''
                    self.grad = grad  
        else:
            self.w = np.zeros(X.shape[1])
            for i in range(self.iteration):                 
                loss = self.compute_loss(X, y)
                self.history.append(loss)#record loss history from t=1, t=0(w intialization) not included                
                grad = np.dot((np.dot(X,self.w)-y), X) + 0.1*self.w
                self.w = self.w - self.learning_rate * grad
            print('Training Finished!')
            
                
                
        toc = time.time()
        self.resource = toc - tic
        
        return self
