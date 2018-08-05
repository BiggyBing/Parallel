# -*- coding: utf-8 -*-
"""
Functions used for CNN algorithm


"""
import numpy as np
import tensorflow as tf
from tensorflow.python.training.adam import AdamOptimizer

def cnn_parameter_initial(parameter_length_list):   
    '''
    Initialize parameter that send to CNN layers
    Adjustment of parameter_length_list neede when change data 
    '''
    parameter = []
    for i in range(len(parameter_length_list)):
        temp = np.random.normal(0,0.03,parameter_length_list[i]).astype(np.float32)
        parameter.append(temp)
    return np.array(parameter)


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


def batch_gradient_collector(gradient):
    '''
    Collect the gradient of each batch
    Input: the return value of optimizer.compute_gradients (many matrices)
    Output: the sum of gradients within one epoch as a vector
    
    '''
    gradient = np.array(gradient)
    #shape_list record parameter shapes of each layer
    gradient_vector = []
    for i in range(gradient.shape[0]):
        grad_temp = gradient[i,0].flatten()
        gradient_vector.append(grad_temp)
        
    return np.array(gradient_vector)   


def batch_parameter_collector(gradient):
    '''
    Collect the gradient of each batch
    Input: the return value of optimizer.compute_gradients (many matrices)
    Output: the sum of gradients within one epoch as a vector
    
    '''
    gradient = np.array(gradient)
    #shape_list record parameter shapes of each layer
    parameter_vector = []
    for i in range(gradient.shape[0]):
        grad_temp = gradient[i,1].flatten()
        parameter_vector.append(grad_temp)
        
    return np.array(parameter_vector)

def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
    # setup the filter input shape for tf.nn.conv_2d
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels,
                      num_filters]

    # initialise weights and bias for the filter
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03),
                                      name=name+'_W')
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')

    # setup the convolutional layer operation
    out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')

    # add the bias
    out_layer += bias

    # apply a ReLU non-linear activation
    out_layer = tf.nn.relu(out_layer)

    # now perform max pooling
    ksize = [1, pool_shape[0], pool_shape[1], 1]
    strides = [1, 2, 2, 1]
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, 
                               padding='SAME')

    return out_layer

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
    
    
#The padding type is in consistent with pooling strided defined in create_new_conv_layer()
def conv_output_size(input_size, filter_size, stride, padding = 'Same'):
    if padding == 'Same':
        output_size = input_size
    else:
        output_size = int((input_size - filter_size)/stride) + 1
    return output_size

#The strides of pooling is defaulted to be 2 in consistent with pooling strided defined in create_new_conv_layer()
def pooling_output_size(input_size, filter_size, stride=2):
    return int((input_size - filter_size)/stride) + 1

def flatten_matrix(matrix):
    '''
    Flatten parameter matrix recieve from classifier objects.
    Parameter recieved from cnn object have a shape (#number of parameter matrix, # of paramter each matrix)
    
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