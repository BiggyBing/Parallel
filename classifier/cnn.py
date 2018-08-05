import time
import numpy as np
import tensorflow as tf
from parallel_base import Parallel_base
from tensorflow.python.training.adam import AdamOptimizer


class CNN(Parallel_base):
    
    
    def __init__(self,layer_size, batch_size = 50, learning_rate = 0.0001, train_type = 'Parallel', iteration = 1000):
    
        super().__init__(learning_rate = learning_rate, train_type = train_type, iteration = iteration)
        
        
        self.pred = 0
        self.loss = 0
        self.optimizer_gradient = 0
        self.optimizer = 0
        self.shape_list = 0
        self.init_op = 0
        
        self.batch_size = batch_size
        #Parameter size in each layer
        #Layer_1 convolutionary layer and pool layer
        self.layer1_size = layer_size[0]#[1, 32, [5, 5], [2, 2]]
        self.layer2_size = layer_size[1]#[32, 64, [5, 5], [2, 2]]
        self.flatten1_size = layer_size[2] #1000
        self.flatten2_size = layer_size[3] #10

    def Est_Belta(self, dataset):
        w_global = flatten_matrix(self.w_hat)
        w_local = flatten_matrix(self.w)
        self.grad_t0 = batch_gradient_collector(self.compute_gradient(dataset)[1])
        grad_global_parameter = flatten_matrix(self.grad_t0)
        self.belta = np.linalg.norm(flatten_matrix(self.grad) - grad_global_parameter)/np.linalg.norm(w_global - w_local)   
        
        
        
    def compute_loss(self, X, y, eta = 1):
        '''
        Computing loss according to cross entropy loss
        
        Args:
            X: Features, shape = (# of samples, # of features)
            y: Target, shape = (1, # of samples)
            eta: coefficient

        Returns: float

        '''
        return
    #The padding type and stride is in consistent with pooling strided defined in create_new_conv_layer()
    def conv_output_size(self, input_size, filter_size, stride=1, padding = 'Same'):
        if padding == 'Same':
            output_size = input_size
        else:
            output_size = int((input_size - filter_size)/stride) + 1
        return output_size
    #The strides of pooling is defaulted to be 2 in consistent with pooling strided defined in create_new_conv_layer()
    def pooling_output_size(self, input_size, filter_size, stride=2):
        return int((input_size - filter_size)/stride) + 1
    
    
    def flattend_size(self):
        '''
        Used to calculate output size after conv and pooling layers
        Input_size of 28 needed to be adjust when other datasets are used
        
        '''
        return self.pooling_output_size(self.conv_output_size(
            self.pooling_output_size(self.conv_output_size(28, self.layer1_size[2][0]),
                                     self.layer1_size[3][0]), self.layer2_size[2][0]),self.layer2_size[3][0])
 

    def compute_gradient(self, dataset): 
        w_global = self.w_hat
    
        x = tf.placeholder(tf.float32, [None, 784])
        # dynamically reshape the input
        x_shaped = tf.reshape(x, [-1, 28, 28, 1])
        # now declare the output data placeholder - 10 digits
        y = tf.placeholder(tf.float32, [None, 10])
        # create some convolutional layers
        layer1 = create_new_conv_layer(x_shaped,w_global[:2],self.layer1_size[0], 
                                       self.layer1_size[1], self.layer1_size[2], self.layer1_size[3], name='layer1')
    
        layer2 = create_new_conv_layer(layer1,w_global[2:4], self.layer2_size[0], 
                                       self.layer2_size[1], self.layer2_size[2], self.layer2_size[3], name='layer2')
        
        
        flattened_parameter_size = self.flattend_size()**2 * self.layer2_size[1]
        flattened = tf.reshape(layer2, [-1, flattened_parameter_size])
        
        # setup some weights and bias values for this layer, then activate with ReLU
     
        wd1 = tf.Variable(w_global[4].reshape(flattened_parameter_size, self.flatten1_size), name='wd1')
        bd1 = tf.Variable(w_global[5], name='bd1')
        dense_layer1 = tf.matmul(flattened, wd1) + bd1
        dense_layer1 = tf.nn.relu(dense_layer1)

        # another layer with softmax activations
        
        wd2 = tf.Variable(w_global[6].reshape(self.flatten1_size, self.flatten2_size), name='wd2')
        bd2 = tf.Variable(w_global[7], name='bd2')
        dense_layer2 = tf.matmul(dense_layer1, wd2) + bd2
        
        y_ = tf.nn.softmax(dense_layer2)
        #loss is cross_entropy loss
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer2, labels=y))

        #Optimizer initialization
        optimizer_gradient = AdamOptimizer_Bing(learning_rate=self.learning_rate).minimize(cross_entropy)

        # setup the initialisation operator        
        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            # initialise the variables
            sess.run(init_op)
            total_batch = len(dataset.train.labels)
            for epoch in range(1):
                batch_x, batch_y = dataset.train.next_batch(batch_size=total_batch)
                g,c = sess.run([optimizer_gradient,cross_entropy], feed_dict={x: batch_x, y: batch_y})

        return g
        
        
    def fit(self, dataset):        
        self.w = self.w_hat
        
        if self.train_type == 'Center':
            self.torque = self.iteration
            self.w = 0
        
        x = tf.placeholder(tf.float32, [None, 784])
        # dynamically reshape the input
        x_shaped = tf.reshape(x, [-1, 28, 28, 1])
        # now declare the output data placeholder - 10 digits
        y = tf.placeholder(tf.float32, [None, 10])
        # create some convolutional layers
        layer1 = create_new_conv_layer(x_shaped,self.w[:2],self.layer1_size[0], 
                                       self.layer1_size[1], self.layer1_size[2], self.layer1_size[3], name='layer1')

        layer2 = create_new_conv_layer(layer1,self.w[2:4], self.layer2_size[0], 
                                       self.layer2_size[1], self.layer2_size[2], self.layer2_size[3], name='layer2')
        
        
        flattened_parameter_size = self.flattend_size()**2 * self.layer2_size[1]
        flattened = tf.reshape(layer2, [-1, flattened_parameter_size])
        
        # setup some weights and bias values for this layer, then activate with ReLU
     
        wd1 = tf.Variable(self.w[4].reshape(flattened_parameter_size, self.flatten1_size), name='wd1')
        bd1 = tf.Variable(self.w[5], name='bd1')
        dense_layer1 = tf.matmul(flattened, wd1) + bd1
        dense_layer1 = tf.nn.relu(dense_layer1)

        # another layer with softmax activations
        
        wd2 = tf.Variable(self.w[6].reshape(self.flatten1_size, self.flatten2_size), name='wd2')
        bd2 = tf.Variable(self.w[7], name='bd2')
        dense_layer2 = tf.matmul(dense_layer1, wd2) + bd2
        
        y_ = tf.nn.softmax(dense_layer2)
        #loss is cross_entropy loss
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer2, labels=y))
        
            
            
        #metrics 
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #Optimizer initialization
        optimizer_gradient = AdamOptimizer_Bing(learning_rate=self.learning_rate).minimize(cross_entropy)
        optimizer = AdamOptimizer(learning_rate=self.learning_rate).minimize(cross_entropy)
        


        # setup the initialisation operator        
        init_op = tf.global_variables_initializer()
        grad = []
        with tf.Session() as sess:
            # initialise the variables
            sess.run(init_op)
            total_batch = int(len(dataset.train.labels) / self.batch_size)
            count = 0
            for epoch in range(self.torque):
                avg_cost = 0
                self.t += 1
                count += 1

                if count < self.torque:
                    for i in range(total_batch):
                        batch_x, batch_y = dataset.train.next_batch(batch_size=self.batch_size)
                        _,c = sess.run([optimizer,cross_entropy], feed_dict={x: batch_x, y: batch_y})
                        avg_cost += c / total_batch

                elif count == self.torque:
                    '''
                    #self.grad saved for belta computation. 
                    #It denotes in time t(update time), the gradient of local loss of local parameters

                    '''

                    for i in range(total_batch):
                        batch_x, batch_y = dataset.train.next_batch(batch_size=self.batch_size)
                        g,c = sess.run([optimizer_gradient,cross_entropy], feed_dict={x: batch_x, y: batch_y})
                        #g[1] is grad_var list
                        gradient_temp = batch_gradient_collector(g[1])
                        grad.append(gradient_temp)
                        avg_cost += c / total_batch
                    
                    self.w = batch_parameter_collector(g[1])
                    #Sum up gradients from each batch
                    self.grad = np.array(grad).sum(axis = 0)
                
                
                test_acc = sess.run(accuracy, feed_dict={x: dataset.test.images, y: dataset.test.labels})   
                self.history.append([avg_cost,test_acc, str(self.t)])


            return self
    
def create_new_conv_layer(input_data, parameters, num_input_channels, num_filters, filter_shape, pool_shape, name):
    # setup the filter input shape for tf.nn.conv_2d
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels,
                      num_filters]
    weights = parameters[0].reshape(conv_filt_shape)
    bias = parameters[1].reshape(num_filters)
    
    # initialise weights and bias for the filter
    weights_cnn = tf.Variable(weights,
                                      name=name+'_W')
    bias_cnn = tf.Variable(bias, name=name+'_b')

    # setup the convolutional layer operation
    out_layer = tf.nn.conv2d(input_data, weights_cnn, [1, 1, 1, 1], padding='SAME')

    # add the bias
    out_layer += bias_cnn

    # apply a ReLU non-linear activation
    out_layer = tf.nn.relu(out_layer)

    # now perform max pooling
    ksize = [1, pool_shape[0], pool_shape[1], 1]
    strides = [1, 2, 2, 1]
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, 
                               padding='SAME')

    return out_layer


def batch_parameter_collector(paramter):
    '''
    Collect the variables from grad_var list of compute_gradients() function
    Input: the return value of optimizer.compute_gradients (grad_var list)
    Output: the sum of gradients within one epoch as a vector
    
    '''
    paramter = np.array(paramter)
    #shape_list record parameter shapes of each layer
    parameter_vector = []
    for i in range(paramter.shape[0]):
        parameter_temp = paramter[i,1].flatten()
        parameter_vector.append(parameter_temp)
        
    return np.array(parameter_vector)

def batch_gradient_collector(gradient):
    '''
    Collect the gradient from grad_var list of compute_gradients() function
    Input: the return value of optimizer.compute_gradients (grad_var list)
    Output: the sum of gradients within one epoch as a vector
    
    '''
    gradient = np.array(gradient)
    #shape_list record parameter shapes of each layer
    gradient_vector = []
    for i in range(gradient.shape[0]):
        grad_temp = gradient[i,0].flatten()
        gradient_vector.append(grad_temp)
        
    return np.array(gradient_vector)


def parameter_shape(gradient):
    '''
    Record the parameters original shape
    Input: the return value of optimizer.compute_gradients (many matrices)
    Output: A list of shape
    
    '''
    gradient = np.array(gradient)
    #shape_list record parameter shapes of each layer
    shape_list = []
    for i in range(gradient.shape[0]):
        grad_temp = gradient[i,0].flatten()
        shape_list.append(['Layer_'+str(i+1), gradient[i,0].shape, grad_temp.shape[0]])
        
    return shape_list

class AdamOptimizer_Bing(AdamOptimizer):
     
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
               use_locking=False, name="Adam"):

        super(AdamOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None
        self._epsilon_t = None

        # Created in SparseApply if needed.
        self._updated_lr = None
    
    def minimize(self, loss, global_step=None, var_list=None,
               gate_gradients=1, aggregation_method=None,
               colocate_gradients_with_ops=False, name=None,
               grad_loss=None):
        """
        The same as function minimize, but return the result of compute_gradients
        Created by: Big Bing in 7/28 
        Purpose: To realize parallel computing(communicate gradient)
        """
        grads_and_vars = self.compute_gradients(
            loss, var_list=var_list, gate_gradients=gate_gradients,
            aggregation_method=aggregation_method,
            colocate_gradients_with_ops=colocate_gradients_with_ops,
            grad_loss=grad_loss)[-8:]

        vars_with_grad = [v for g, v in grads_and_vars if g is not None]
        if not vars_with_grad:
          raise ValueError(
              "No gradients provided for any variable, check your graph for ops"
              " that do not support gradients, between variables %s and loss %s." %
              ([str(v) for _, v in grads_and_vars], loss))
        #self.apply_gradients(grads_and_vars, global_step=global_step, name=name)

        return self.apply_gradients(grads_and_vars, global_step=global_step, name=name),grads_and_vars
        
def flatten_matrix(matrix):
    '''
    Flatten parameter matrix recieve from classifier objects.
    Parameter recieved from cnn object have a shape (#number of parameter type, # of paramter each matrix)
    var:
        matrix: paramter and gradient matrix of cnn
    output: a vector 
    '''
    temp = []
    for i in range(len(matrix)):
        temp.extend(matrix[i])
    return np.array(temp)      

def wrangle_matrix(vector, parameter_length_list):
    '''
    Reverse function of flatten_matrix.
    vector to matrix
    Var:
        vector: A vector like parameter list
        parameter_length_list: A list that saves parameter length of each layer
    '''
    matrix = []
    flag = 0
    for i in range(len(parameter_length_list)):
            temp = vector[flag:parameter_length_list[i]]
            matrix.append(temp)
            flag += parameter_length_list[i]
    return np.array(matrix)