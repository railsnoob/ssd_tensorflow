import tensorflow as tf
from tensorflow.contrib.layers import flatten
import numpy as np
import pickle
import matplotlib.image as mpimg


class IllegalArgumentError(ValueError):
    pass

class BaseNet:
    def __init__(self,num_default_boxes,num_classes):
        self.num_default_boxes = num_default_boxes
        self.num_classes = num_classes
        pass

    def conv_layer_optional_pooling(self, x_tensor,n_outputs,n_ksize,n_strides,name,
                                    padding_type="VALID",pool_ksize=None,pool_strides=None,pool_name=None):
        # Convolution layer with Relu 
        """
        x_tensor : input tensor
        n_outputs: number of outputs of the convolutional layer
        n_ksize: kernel size 2-d tuple 
        n_strides: 2-d tuple for convlution
        padding_type: Type of padding. default is "SAME"
        pool_ksize: kernel size 2-d tuple for max pool
        pool_strides: stride 2-d tuple for max pool
        
        returns A tensor that is hte output of convolution, relu & max-pooling (optional)
        """
    
        # check for dumb input errors
        # if (len(n_ksize) != 2) or (len(n_strides) !=2) or \
        # (pool_ksize!=None and len(pool_ksize)!=2) or (pool_strides!=None and len(pool_strides!=2)):
        #    raise IllegalArgumentError
        
        num_channels = int(x_tensor.shape[-1])
    
        filter_weight = tf.Variable(tf.truncated_normal(list(n_ksize)+[num_channels,n_outputs],mean=0,stddev=0.001))
        filter_bias = tf.Variable(tf.zeros(n_outputs))
    
        conv_layer = tf.nn.conv2d(x_tensor,filter_weight,[1]+list(n_strides)+[1],padding_type,name=name)
        conv_layer = tf.nn.bias_add(conv_layer,filter_bias)
        conv_layer = tf.nn.relu(conv_layer)

        print(name,conv_layer.shape)
    
        if pool_ksize!= None and pool_strides !=None:
            pooled = tf.nn.max_pool(conv_layer,
                                    ksize=[1] + list(pool_ksize) + [1],
                                    strides = [1] + list(pool_strides) + [1],
                                    padding='SAME',name=pool_name)
            print("After pool ",pool_name,pooled.shape)
            return pooled
        
        return conv_layer

    def flatten(self, x_tensor):
        batch_size = x_tensor.shape[0]
        mult = 1
        for a in range(1,len(x_tensor.shape)):
            mult = mult * int(x_tensor.shape[a])
        return tf.reshape(x_tensor,[-1,mult])

    def fully_conn(self, x_tensor,num_outputs):
        num_inputs = int(x_tensor.shape[1])
        weight= tf.Variable(tf.random_normal([num_inputs,num_outputs],mean=0,stddev=0.001))
        bias = tf.Variable(tf.zeros(shape=num_outputs))
    
        layer = tf.add(tf.matmul(x_tensor,weight),bias)
        layer = tf.nn.relu(layer)
        return layer
    
    def convolve_and_collect(self,fmap,name,y_box_coords,y_class):
        # Apply 2 convolutions and get predictions for coordinates and classes and store them 
        # Get both of these guys and convolve
        b = self.conv_layer_optional_pooling(fmap,4*self.num_default_boxes,(3,3),(1,1),name+"box_coords",padding_type="SAME")
        print("   =====> ",name+"box_coords",self.flatten(b))
        y_box_coords.insert(0,flatten(b))
    
        c = self.conv_layer_optional_pooling(fmap,self.num_classes*self.num_default_boxes,(3,3),(1,1),name+"class",padding_type="SAME")
        print("   =====> ",name+"class",self.flatten(c))
        y_class.insert(0,flatten(c))


