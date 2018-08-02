import time
import numpy as np
from scipy import stats
from parallel_base import Parallel_base
from sklearn.metrics.pairwise import euclidean_distances

class Kmeans(Parallel_base):
    
    def __init__(self, cluster_num, learning_rate = 0.0001, train_type = 'Parallel', iteration = 1000):
        '''
        Initialization of object
        
        Args:
            cluster_num: the number of cluster

        Returns: no
        
        Raise: ValueError raised when cluster is not integer.

        '''  
        if isinstance(cluster_num,int) == False:
            raise ValueError('The value of cluster must be integer, {} is given'.format(cluster_num))
            
        super().__init__(learning_rate = learning_rate, train_type = train_type, iteration = iteration)
        
        self.cluster = cluster_num
        
    def parameter_init(self, X):
        '''
        Initialize parameter: parameter selected from normal distribution
        
        Args:
            X: Features, shape = (# of samples, # of features)
            y: Target, shape = (1, # of samples)

        Returns: np.array, shape = (cluster_num, # of features)

        '''
        
        statistic = stats.describe(X)
        w_init = np.random.normal(loc = statistic[2], scale = statistic[3], size = (self.cluster,X.shape[1]))
        
        return w_init
    
    def Est_Belta(self ,X):
        #estimate clusters given global parameters
        distance_matrix = euclidean_distances(X, self.w_hat)
        cluster_id = self.partition(distance_matrix)
        grad_global_parameter = self.grad_kmeans(self.w_hat, X, cluster_id)
        self.grad_t0 = grad_global_parameter
        self.belta = np.linalg.norm(self.grad - grad_global_parameter)/np.linalg.norm(self.w - self.w_hat)
    
    def compute_loss(self, X, y):
        '''
        Computing loss according to OLS
        
        Args:
            X: Features, shape = (# of samples, # of features)
            y: Target, shape = (1, # of samples)

        Returns: loss

        '''
        
        m = len(y)
        sum_of_square_errors = np.square(np.dot(X, self.w)-y).sum()
        cost = sum_of_square_errors/(2*m)
    
        return cost
    
    def partition(self, distance_matrix):
        cluster = [[] for i in range(distance_matrix.shape[1])]
        for i in range(len(distance_matrix)):
            cluster_label = np.argmin(distance_matrix[i])
            cluster[cluster_label].append(i)
        return cluster
    
    def grad_kmeans(self, X, cluster_id):
        shape = self.w.shape
        grad = np.zeros(shape = shape)
        for m in range(shape[0]):
            for n in range(shape[1]):
                grad[m,n] = self.w[m,n]*len(cluster_id[m]) - X[cluster_id[m],n].sum()
        return grad
        

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
            for i in range(self.torque): 
                self.t += 1
                distance_matrix = euclidean_distances(X, self.w)
                cluster_id = self.partition(distance_matrix)
                grad = self.grad_kmeans(X, cluster_id)
                self.w = self.w - self.learning_rate * grad
                
                count += 1
                if count < self.torque:
                    self.w_hat = self.w
                elif count == self.torque:
                    self.grad = grad  #self.grad save the gradient(t) when t has a global aggregation

        else:
            self.w = self.parameter_init( X)
            for i in range(self.iteration):                 
                self.t += 1
                distance_matrix = euclidean_distances(X, self.w)
                cluster_id = self.partition(distance_matrix)
                grad = self.grad_kmeans(X, cluster_id)
                self.w = self.w - self.learning_rate * grad
                
            print('Training Finished!')
            
                
                
        toc = time.time()
        self.resource = toc - tic
        
        return self
