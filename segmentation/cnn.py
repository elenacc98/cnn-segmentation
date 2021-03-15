"""
The cnn submodule implements some classes to define CNN-based
models in a dynamic way.
"""
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import Model

from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Conv1D, Conv2D, Conv3D, Conv3DTranspose
from keras.layers.pooling import AveragePooling2D, AveragePooling3D, GlobalAveragePooling3D, MaxPool3D
from keras.layers import Input, Concatenate, Lambda, Dropout, Concatenate, Multiply, Softmax, Add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from segmentation.utils import ASPP, PEE, RA, MINI_MTL, CFF, build_MINI_MTL
from tensorflow.keras.applications import *
import math


class UNet(object):
    """
    This class provides a simple interface to create
    a U-Net network with custom parameters. 
    Args:
        input_size: input size for the network
        kernel_size: size of the kernel to be used in the convolutional layers of the U-Net
        strides: stride shape to be used in the convolutional layers of the U-Net
        deconv_strides: stride shape to be used in the deconvolutional layers of the U-Net
        deconv_kernel_size: kernel size shape to be used in the deconvolutional layers of the U-Net
        pool_size: size of the pool size to be used in MaxPooling layers
        pool_strides: size of the strides to be used in MaxPooling layers
        depth: depth of the U-Net model
        activation: activation function used in the U-Net layers
        padding: padding used for the input data in all the U-Net layers
        n_inital_filters: number of feature maps in the first layer of the U-Net
        add_batch_normalization: boolean flag to determine if batch normalization should be applied after convolutional layers
        kernel_regularizer: kernel regularizer to be applied to the convolutional layers of the U-Net
        bias_regularizer: bias regularizer to be applied to the convolutional layers of the U-Net
        n_classes: number of classes in labels
    """

    def __init__(self, input_size = (128,128,128,1), 
                        kernel_size = (3,3,3),
                        strides = (1,1,1),
                        deconv_kernel_size = (2,2,2),
                        deconv_strides = (2,2,2),
                        pool_size = (2,2,2),
                        pool_strides = (2,2,2),
                        depth = 5,
                        activation = 'relu',
                        padding = 'same',
                        n_initial_filters = 8,
                        add_batch_normalization = True,
                        kernel_regularizer = regularizers.l2(0.001),
                        bias_regularizer = regularizers.l2(0.001),
                        n_classes = 3):

        self.input_size = input_size
        self.n_dim = len(input_size) # Number of dimensions of the input data
        self.kernel_size = kernel_size
        self.strides = strides
        self.deconv_kernel_size = deconv_kernel_size
        self.deconv_strides = deconv_strides
        self.depth = depth
        self.activation = activation
        self.padding = padding
        self.n_initial_filters = n_initial_filters
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.add_batch_normalization = add_batch_normalization
        self.pool_size = pool_size
        self.pool_strides = pool_strides
        self.n_classes = n_classes

    def create_model(self):
        '''
        This function creates a U-Net network based
        on the configuration.
        '''
        # Check if 2D or 3D convolution must be used
        if (self.n_dim == 3):
            conv_layer = layers.Conv2D
            max_pool_layer = layers.MaxPooling2D
            conv_transpose_layer = layers.Conv2DTranspose
            softmax_kernel_size = (1,1)
        elif (self.n_dim == 4):
            conv_layer = layers.Conv3D
            max_pool_layer = layers.MaxPooling3D
            conv_transpose_layer = layers.Conv3DTranspose
            softmax_kernel_size = (1,1,1)
        else:
            print("Could not handle input dimensions.")
            return

        # Input layer
        temp_layer = layers.Input(shape=self.input_size)
        input_tensor = temp_layer
        
        # Variables holding the layers so that they can be concatenated
        downsampling_layers = []
        upsampling_layers = []
        # Down sampling branch
        for i in range(self.depth):
            for j in range(2):
                # Convolution
                temp_layer = conv_layer(self.n_initial_filters*pow(2,i), kernel_size = self.kernel_size, 
                                            strides = self.strides, 
                                            padding = self.padding,
                                            activation = 'linear', 
                                            kernel_regularizer = self.kernel_regularizer, 
                                            bias_regularizer = self.bias_regularizer)(temp_layer)
                # batch normalization
                if (self.add_batch_normalization):
                    temp_layer = layers.BatchNormalization(axis = -1)(temp_layer)
                # activation
                temp_layer = layers.Activation(self.activation)(temp_layer)
            downsampling_layers.append(temp_layer)
            temp_layer = max_pool_layer(pool_size = self.pool_size,
                                         strides=self.pool_strides, 
                                         padding=self.padding)(temp_layer)
        for j in range(2): 
            # Bottleneck
            temp_layer = conv_layer(self.n_initial_filters*pow(2,self.depth), kernel_size = self.kernel_size, 
                                        strides = self.strides, 
                                        padding = self.padding,
                                        activation = 'linear',
                                        kernel_regularizer = self.kernel_regularizer,
                                        bias_regularizer = self.bias_regularizer)(temp_layer)
            if (self.add_batch_normalization):
                temp_layer = temp_layer = layers.BatchNormalization(axis = -1)(temp_layer)
            # activation
            temp_layer = layers.Activation(self.activation)(temp_layer)
        
        # Up sampling branch
        for i in range(self.depth):
            temp_layer = conv_transpose_layer(self.n_initial_filters*pow(2,(self.depth-1)-i),
                                                    kernel_size=self.deconv_kernel_size,
                                                    strides=self.deconv_strides,
                                                    activation='linear',
                                                    padding=self.padding,
                                                    kernel_regularizer=self.kernel_regularizer,
                                                    bias_regularizer=self.bias_regularizer)(temp_layer)
            # activation
            temp_layer = layers.Activation(self.activation)(temp_layer)
            # concatenation
            temp_layer = layers.Concatenate(axis=self.n_dim)([downsampling_layers[(self.depth-1)-i], temp_layer])
            # convolution
            for j in range(2):
                # Convolution
                temp_layer = conv_layer(self.n_initial_filters*pow(2,(self.depth-1)-i),
                                            kernel_size = self.kernel_size, 
                                            strides = self.strides, 
                                            padding = self.padding,
                                            activation = 'linear', 
                                            kernel_regularizer = self.kernel_regularizer, 
                                            bias_regularizer = self.bias_regularizer)(temp_layer)
                # activation
                temp_layer = layers.Activation(self.activation)(temp_layer)
  

        # Convolution 1 filter sigmoidal (to make size converge to final one)
        temp_layer = conv_layer(self.n_classes, kernel_size = softmax_kernel_size,
                                                       strides = self.strides,
                                                       padding = 'same', 
                                                       activation = 'linear',
                                                       kernel_regularizer = self.kernel_regularizer, 
                                                       bias_regularizer = self.bias_regularizer)(temp_layer)

        output_tensor = layers.Softmax(axis = -1)(temp_layer)
        self.model = Model(inputs = [input_tensor], outputs = [output_tensor])

    def set_initial_weights(self, weights):
        '''
        Set the initial weights of the U-Net, in case
        training was stopped and then resumed. An exception
        is raised in case the model currently configured
        has different properties than the one whose weights
        were stored.
        '''
        try:
            self.model.load_weights(weights)
        except:
            raise
    
    def get_n_parameters(self):
        '''
        Get the total number of parameters of the model
        '''
        return self.model.count_params()

    def summary(self):
        '''
        Print out summary of the model.
        '''
        print(self.model.summary())


class CEL_UNet(object):
    """
    This class provides a simple interface to create
    a U-Net network with custom parameters.
    Args:
        input_size: input size for the network
        kernel_size: size of the kernel to be used in the convolutional layers of the U-Net
        strides: stride shape to be used in the convolutional layers of the U-Net
        deconv_strides: stride shape to be used in the deconvolutional layers of the U-Net
        deconv_kernel_size: kernel size shape to be used in the deconvolutional layers of the U-Net
        pool_size: size of the pool size to be used in MaxPooling layers
        pool_strides: size of the strides to be used in MaxPooling layers
        depth: depth of the U-Net model
        activation: activation function used in the U-Net layers
        padding: padding used for the input data in all the U-Net layers
        n_inital_filters: number of feature maps in the first layer of the U-Net
        add_batch_normalization: boolean flag to determine if batch normalization should be applied after convolutional layers
        kernel_regularizer: kernel regularizer to be applied to the convolutional layers of the U-Net
        bias_regularizer: bias regularizer to be applied to the convolutional layers of the U-Net
        n_classes: number of classes in labels
    """

    def __init__(self, input_size=(128, 128, 128, 1),
                 kernel_size=(3, 3, 3),
                 strides=(1, 1, 1),
                 deconv_kernel_size=(2, 2, 2),
                 deconv_strides=(2, 2, 2),
                 pool_size=(2, 2, 2),
                 pool_strides=(2, 2, 2),
                 depth=5,
                 activation='relu',
                 padding='same',
                 n_initial_filters=8,
                 add_batch_normalization=True,
                 kernel_regularizer=regularizers.l2(0.001),
                 bias_regularizer=regularizers.l2(0.001),
                 n_classes=3):

        self.input_size = input_size
        self.n_dim = len(input_size)  # Number of dimensions of the input data
        self.kernel_size = kernel_size
        self.strides = strides
        self.deconv_kernel_size = deconv_kernel_size
        self.deconv_strides = deconv_strides
        self.depth = depth
        self.activation = activation
        self.padding = padding
        self.n_initial_filters = n_initial_filters
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.add_batch_normalization = add_batch_normalization
        self.pool_size = pool_size
        self.pool_strides = pool_strides
        self.n_classes = n_classes

    def create_model(self):
        '''
        This function creates a U-Net network based
        on the configuration.
        '''
        # Check if 2D or 3D convolution must be used
        if (self.n_dim == 3):
            conv_layer = layers.Conv2D
            max_pool_layer = layers.MaxPooling2D
            conv_transpose_layer = layers.Conv2DTranspose
            softmax_kernel_size = (1, 1)
        elif (self.n_dim == 4):
            conv_layer = layers.Conv3D
            max_pool_layer = layers.MaxPooling3D
            conv_transpose_layer = layers.Conv3DTranspose
            softmax_kernel_size = (1, 1, 1)
        else:
            print("Could not handle input dimensions.")
            return

        # Input layer
        temp_layer = layers.Input(shape=self.input_size)
        input_tensor = temp_layer

        # Variables holding the layers so that they can be concatenated
        downsampling_layers = []
        upsampling_layers = []
        # Down sampling branch
        for i in range(self.depth):
            for j in range(2):
                # Convolution
                temp_layer = conv_layer(self.n_initial_filters * pow(2, i), kernel_size=self.kernel_size,
                                        strides=self.strides,
                                        padding=self.padding,
                                        activation='linear',
                                        kernel_regularizer=self.kernel_regularizer,
                                        bias_regularizer=self.bias_regularizer)(temp_layer)
                # batch normalization
                if self.add_batch_normalization:
                    temp_layer = layers.BatchNormalization(axis=-1)(temp_layer)
                # activation
                temp_layer = layers.Activation(self.activation)(temp_layer)
            downsampling_layers.append(temp_layer)
            temp_layer = max_pool_layer(pool_size=self.pool_size,
                                        strides=self.pool_strides,
                                        padding=self.padding)(temp_layer)
        for j in range(2):
            # Bottleneck
            temp_layer = conv_layer(self.n_initial_filters * pow(2, self.depth), kernel_size=self.kernel_size,
                                    strides=self.strides,
                                    padding=self.padding,
                                    activation='linear',
                                    kernel_regularizer=self.kernel_regularizer,
                                    bias_regularizer=self.bias_regularizer)(temp_layer)
            if self.add_batch_normalization:
                temp_layer = layers.BatchNormalization(axis=-1)(temp_layer)
            # activation
            temp_layer = layers.Activation(self.activation)(temp_layer)

        # Up sampling branch
        temp_layer_edge = temp_layer
        temp_layer_merge = temp_layer

        for i in range(self.depth):
            # EDGE PATH
            temp_layer_edge = conv_transpose_layer(self.n_initial_filters * pow(2, (self.depth - 1) - i),
                                                   kernel_size=self.deconv_kernel_size,
                                                   strides=self.deconv_strides,
                                                   activation='linear',
                                                   padding=self.padding,
                                                   kernel_regularizer=self.kernel_regularizer,
                                                   bias_regularizer=self.bias_regularizer)(temp_layer_edge)
            temp_layer_edge = layers.Activation(self.activation)(temp_layer_edge)

            # MASK PATH
            temp_layer_mask = conv_transpose_layer(self.n_initial_filters * pow(2, (self.depth - 1) - i),
                                                   kernel_size=self.deconv_kernel_size,
                                                   strides=self.deconv_strides,
                                                   activation='linear',
                                                   padding=self.padding,
                                                   kernel_regularizer=self.kernel_regularizer,
                                                   bias_regularizer=self.bias_regularizer)(temp_layer_merge)
            temp_layer_mask = layers.Activation(self.activation)(temp_layer_mask)

            # Concatenation
            temp_layer_edge = layers.Concatenate(axis=self.n_dim)([downsampling_layers[(self.depth - 1) - i], temp_layer_edge])
            temp_layer_mask = layers.Concatenate(axis=self.n_dim)([downsampling_layers[(self.depth - 1) - i], temp_layer_mask])

            for j in range(2):
                temp_layer_edge = conv_layer(self.n_initial_filters * pow(2, (self.depth - 1) - i),
                                             kernel_size=self.kernel_size,
                                             strides=self.strides,
                                             padding=self.padding,
                                             activation='linear',
                                             kernel_regularizer=self.kernel_regularizer,
                                             bias_regularizer=self.bias_regularizer)(temp_layer_edge)
                if self.add_batch_normalization:
                    temp_layer_edge = layers.BatchNormalization(axis=-1)(temp_layer_edge)
                temp_layer_edge = layers.Activation(self.activation)(temp_layer_edge)

                temp_layer_mask = conv_layer(self.n_initial_filters * pow(2, (self.depth - 1) - i),
                                             kernel_size=self.kernel_size,
                                             strides=self.strides,
                                             padding=self.padding,
                                             activation='linear',
                                             kernel_regularizer=self.kernel_regularizer,
                                             bias_regularizer=self.bias_regularizer)(temp_layer_mask)
                if self.add_batch_normalization:
                    temp_layer_mask = layers.BatchNormalization(axis=-1)(temp_layer_mask)
                temp_layer_mask = layers.Activation(self.activation)(temp_layer_mask)

            # if i % 2 != 1:
            temp_layer_edge = PEE(temp_layer_edge, self.n_initial_filters * pow(2, (self.depth - 1) - i))

            temp_layer_merge = Concatenate()([temp_layer_edge, temp_layer_mask])
            temp_layer_merge = conv_layer(self.n_initial_filters * pow(2, (self.depth - 1) - i),
                                          kernel_size=self.kernel_size,
                                          strides=self.strides,
                                          padding=self.padding,
                                          activation='linear',
                                          kernel_regularizer=self.kernel_regularizer,
                                          bias_regularizer=self.bias_regularizer)(temp_layer_merge)
            # else:
            #     temp_layer_merge = temp_layer_mask

        # Convolution 1 filter sigmoidal (to make size converge to final one)
        temp_layer_mask = conv_layer(self.n_classes, kernel_size=softmax_kernel_size,
                                     strides=self.strides,
                                     padding='same',
                                     activation='linear',
                                     kernel_regularizer=self.kernel_regularizer,
                                     bias_regularizer=self.bias_regularizer)(temp_layer_merge)

        temp_layer_edge = conv_layer(self.n_classes, kernel_size=softmax_kernel_size,
                                     strides=self.strides,
                                     padding='same',
                                     activation='linear',
                                     kernel_regularizer=self.kernel_regularizer,
                                     bias_regularizer=self.bias_regularizer)(temp_layer_edge)

        output_tensor_edge = layers.Softmax(axis=-1, dtype='float32', name='out_edge')(temp_layer_edge)
        output_tensor_mask = layers.Softmax(axis=-1, dtype='float32',  name='out_mask')(temp_layer_mask)
        self.model = Model(inputs=[input_tensor], outputs=[output_tensor_edge, output_tensor_mask])

    def set_initial_weights(self, weights):
        '''
        Set the initial weights of the U-Net, in case
        training was stopped and then resumed. An exception
        is raised in case the model currently configured
        has different properties than the one whose weights
        were stored.
        '''
        try:
            self.model.load_weights(weights)
        except:
            raise

    def get_n_parameters(self):
        '''
        Get the total number of parameters of the model
        '''
        return self.model.count_params()

    def summary(self):
        '''
        Print out summary of the model.
        '''
        print(self.model.summary())


class Chen_UNet(object):
    """
    This class provides a simple interface to create
    a U-Net network with custom parameters.
    Args:
        input_size: input size for the network
        kernel_size: size of the kernel to be used in the convolutional layers of the U-Net
        strides: stride shape to be used in the convolutional layers of the U-Net
        deconv_strides: stride shape to be used in the deconvolutional layers of the U-Net
        deconv_kernel_size: kernel size shape to be used in the deconvolutional layers of the U-Net
        pool_size: size of the pool size to be used in MaxPooling layers
        pool_strides: size of the strides to be used in MaxPooling layers
        depth: depth of the U-Net model
        activation: activation function used in the U-Net layers
        padding: padding used for the input data in all the U-Net layers
        n_inital_filters: number of feature maps in the first layer of the U-Net
        add_batch_normalization: boolean flag to determine if batch normalization should be applied after convolutional layers
        kernel_regularizer: kernel regularizer to be applied to the convolutional layers of the U-Net
        bias_regularizer: bias regularizer to be applied to the convolutional layers of the U-Net
        n_classes: number of classes in labels
    """

    def __init__(self, input_size=(128, 128, 128, 1),
                 kernel_size=(3, 3, 3),
                 strides=(1, 1, 1),
                 deconv_kernel_size=(2, 2, 2),
                 deconv_strides=(2, 2, 2),
                 pool_size=(2, 2, 2),
                 pool_strides=(2, 2, 2),
                 depth=5,
                 activation='relu',
                 padding='same',
                 n_initial_filters=8,
                 add_batch_normalization=True,
                 kernel_regularizer=regularizers.l2(0.001),
                 bias_regularizer=regularizers.l2(0.001),
                 n_classes=3):

        self.input_size = input_size
        self.n_dim = len(input_size)  # Number of dimensions of the input data
        self.kernel_size = kernel_size
        self.strides = strides
        self.deconv_kernel_size = deconv_kernel_size
        self.deconv_strides = deconv_strides
        self.depth = depth
        self.activation = activation
        self.padding = padding
        self.n_initial_filters = n_initial_filters
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.add_batch_normalization = add_batch_normalization
        self.pool_size = pool_size
        self.pool_strides = pool_strides
        self.n_classes = n_classes

    def create_model(self):
        '''
        This function creates a U-Net network based
        on the configuration.
        '''
        # Check if 2D or 3D convolution must be used
        if (self.n_dim == 3):
            conv_layer = layers.Conv2D
            max_pool_layer = layers.MaxPooling2D
            conv_transpose_layer = layers.Conv2DTranspose
            softmax_kernel_size = (1, 1)
        elif (self.n_dim == 4):
            conv_layer = layers.Conv3D
            max_pool_layer = layers.MaxPooling3D
            conv_transpose_layer = layers.Conv3DTranspose
            softmax_kernel_size = (1, 1, 1)
        else:
            print("Could not handle input dimensions.")
            return

        # Input layer
        temp_layer = layers.Input(shape=self.input_size)
        input_tensor = temp_layer

        # Variables holding the layers so that they can be concatenated
        downsampling_layers = []
        upsampling_layers = []
        # Down sampling branch
        for i in range(self.depth):
            for j in range(2):
                # Convolution
                temp_layer = conv_layer(self.n_initial_filters * pow(2, i), kernel_size=self.kernel_size,
                                        strides=self.strides,
                                        padding=self.padding,
                                        activation='linear',
                                        kernel_regularizer=self.kernel_regularizer,
                                        bias_regularizer=self.bias_regularizer)(temp_layer)
                # batch normalization
                if self.add_batch_normalization:
                    temp_layer = layers.BatchNormalization(axis=-1)(temp_layer)
                # activation
                temp_layer = layers.Activation(self.activation)(temp_layer)
            downsampling_layers.append(temp_layer)
            temp_layer = max_pool_layer(pool_size=self.pool_size,
                                        strides=self.pool_strides,
                                        padding=self.padding)(temp_layer)
        for j in range(2):
            # Bottleneck
            temp_layer = conv_layer(self.n_initial_filters * pow(2, self.depth), kernel_size=self.kernel_size,
                                    strides=self.strides,
                                    padding=self.padding,
                                    activation='linear',
                                    kernel_regularizer=self.kernel_regularizer,
                                    bias_regularizer=self.bias_regularizer)(temp_layer)
            if self.add_batch_normalization:
                temp_layer = layers.BatchNormalization(axis=-1)(temp_layer)
            # activation
            temp_layer = layers.Activation(self.activation)(temp_layer)

        # Up sampling branch
        temp_layer_edge = temp_layer
        temp_layer_mask = temp_layer

        for i in range(self.depth):
            # EDGE PATH
            temp_layer_edge = conv_transpose_layer(self.n_initial_filters * pow(2, (self.depth - 1) - i),
                                                   kernel_size=self.deconv_kernel_size,
                                                   strides=self.deconv_strides,
                                                   activation='linear',
                                                   padding=self.padding,
                                                   kernel_regularizer=self.kernel_regularizer,
                                                   bias_regularizer=self.bias_regularizer)(temp_layer_edge)
            temp_layer_edge = layers.Activation(self.activation)(temp_layer_edge)

            # MASK PATH
            temp_layer_mask = conv_transpose_layer(self.n_initial_filters * pow(2, (self.depth - 1) - i),
                                                   kernel_size=self.deconv_kernel_size,
                                                   strides=self.deconv_strides,
                                                   activation='linear',
                                                   padding=self.padding,
                                                   kernel_regularizer=self.kernel_regularizer,
                                                   bias_regularizer=self.bias_regularizer)(temp_layer_mask)
            temp_layer_mask = layers.Activation(self.activation)(temp_layer_mask)

            # Concatenation
            temp_layer_edge = layers.Concatenate(axis=self.n_dim)([downsampling_layers[(self.depth - 1) - i], temp_layer_edge])
            temp_layer_mask = layers.Concatenate(axis=self.n_dim)([downsampling_layers[(self.depth - 1) - i], temp_layer_mask])

            for j in range(2):
                temp_layer_edge = conv_layer(self.n_initial_filters * pow(2, (self.depth - 1) - i),
                                             kernel_size=self.kernel_size,
                                             strides=self.strides,
                                             padding=self.padding,
                                             activation='linear',
                                             kernel_regularizer=self.kernel_regularizer,
                                             bias_regularizer=self.bias_regularizer)(temp_layer_edge)
                if self.add_batch_normalization:
                    temp_layer_edge = layers.BatchNormalization(axis=-1)(temp_layer_edge)
                temp_layer_edge = layers.Activation(self.activation)(temp_layer_edge)

                temp_layer_mask = conv_layer(self.n_initial_filters * pow(2, (self.depth - 1) - i),
                                             kernel_size=self.kernel_size,
                                             strides=self.strides,
                                             padding=self.padding,
                                             activation='linear',
                                             kernel_regularizer=self.kernel_regularizer,
                                             bias_regularizer=self.bias_regularizer)(temp_layer_mask)
                if self.add_batch_normalization:
                    temp_layer_mask = layers.BatchNormalization(axis=-1)(temp_layer_mask)
                temp_layer_mask = layers.Activation(self.activation)(temp_layer_mask)


        # Convolution 1 filter sigmoidal (to make size converge to final one)
        temp_layer_mask = conv_layer(self.n_classes, kernel_size=softmax_kernel_size,
                                     strides=self.strides,
                                     padding='same',
                                     activation='linear',
                                     kernel_regularizer=self.kernel_regularizer,
                                     bias_regularizer=self.bias_regularizer)(temp_layer_mask)

        temp_layer_edge = conv_layer(self.n_classes, kernel_size=softmax_kernel_size,
                                     strides=self.strides,
                                     padding='same',
                                     activation='linear',
                                     kernel_regularizer=self.kernel_regularizer,
                                     bias_regularizer=self.bias_regularizer)(temp_layer_edge)

        output_tensor_edge = layers.Softmax(axis=-1, dtype='float32', name='out_edge')(temp_layer_edge)
        output_tensor_mask = layers.Softmax(axis=-1, dtype='float32',  name='out_mask')(temp_layer_mask)
        self.model = Model(inputs=[input_tensor], outputs=[output_tensor_edge, output_tensor_mask])

    def set_initial_weights(self, weights):
        '''
        Set the initial weights of the U-Net, in case
        training was stopped and then resumed. An exception
        is raised in case the model currently configured
        has different properties than the one whose weights
        were stored.
        '''
        try:
            self.model.load_weights(weights)
        except:
            raise

    def get_n_parameters(self):
        '''
        Get the total number of parameters of the model
        '''
        return self.model.count_params()

    def summary(self):
        '''
        Print out summary of the model.
        '''
        print(self.model.summary())


class ERANet(object):
    """
    This class provides a simple interface to create
    a ERANet (Edge Reverse Attention) network with custom parameters.
    Args:
        input_size: input size for the network
        kernel_size: size of the kernel to be used in the convolutional layers of the U-Net
        strides: stride shape to be used in the convolutional layers of the U-Net
        deconv_strides: stride shape to be used in the deconvolutional layers of the U-Net
        deconv_kernel_size: kernel size shape to be used in the deconvolutional layers of the U-Net
        pool_size: size of the pool size to be used in MaxPooling layers
        pool_strides: size of the strides to be used in MaxPooling layers
        depth: depth of the U-Net model
        activation: activation function used in the U-Net layers
        padding: padding used for the input data in all the U-Net layers
        n_inital_filters: number of feature maps in the first layer of the U-Net
        add_batch_normalization: boolean flag to determine if batch normalization should be applied after convolutional layers
        kernel_regularizer: kernel regularizer to be applied to the convolutional layers of the U-Net
        bias_regularizer: bias regularizer to be applied to the convolutional layers of the U-Net
        n_classes: number of classes in labels
    """

    def __init__(self, input_size=(128, 128, 128, 1),
                 kernel_size=(3, 3, 3),
                 strides=(1, 1, 1),
                 deconv_kernel_size=(2, 2, 2),
                 deconv_strides=(2, 2, 2),
                 pool_size=(2, 2, 2),
                 pool_strides=(2, 2, 2),
                 depth=5,
                 activation='relu',
                 padding='same',
                 n_initial_filters=8,
                 add_batch_normalization=True,
                 kernel_regularizer=regularizers.l2(0.001),
                 bias_regularizer=regularizers.l2(0.001),
                 n_classes=3):

        self.input_size = input_size
        self.n_dim = len(input_size)  # Number of dimensions of the input data
        self.kernel_size = kernel_size
        self.strides = strides
        self.deconv_kernel_size = deconv_kernel_size
        self.deconv_strides = deconv_strides
        self.depth = depth
        self.activation = activation
        self.padding = padding
        self.n_initial_filters = n_initial_filters
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.add_batch_normalization = add_batch_normalization
        self.pool_size = pool_size
        self.pool_strides = pool_strides
        self.n_classes = n_classes

    def create_model(self):
        '''
        This function creates a U-Net network based
        on the configuration.
        '''
        # Check if 2D or 3D convolution must be used
        if (self.n_dim == 3):
            conv_layer = layers.Conv2D
            max_pool_layer = layers.MaxPooling2D
            conv_transpose_layer = layers.Conv2DTranspose
            softmax_kernel_size = (1, 1)
        elif (self.n_dim == 4):
            conv_layer = layers.Conv3D
            max_pool_layer = layers.MaxPooling3D
            conv_transpose_layer = layers.Conv3DTranspose
            softmax_kernel_size = (1, 1, 1)
        else:
            print("Could not handle input dimensions.")
            return

        # Input layer
        temp_layer = layers.Input(shape=self.input_size)
        input_tensor = temp_layer

        # Variables holding the layers so that they can be concatenated
        downsampling_layers = []
        upsampling_layers = []
        # Down sampling branch
        for i in range(self.depth):
            for j in range(2 + math.floor(i/2)):
                # Convolution
                if j == 0:
                    x = conv_layer(self.n_initial_filters * pow(2, i), kernel_size=(1,1,1),
                                   strides=self.strides,
                                   padding=self.padding,
                                   activation='linear',
                                   kernel_regularizer=self.kernel_regularizer,
                                   bias_regularizer=self.bias_regularizer)(temp_layer)
                temp_layer = conv_layer(self.n_initial_filters * pow(2, i), kernel_size=self.kernel_size,
                                        strides=self.strides,
                                        padding=self.padding,
                                        activation='linear',
                                        kernel_regularizer=self.kernel_regularizer,
                                        bias_regularizer=self.bias_regularizer)(temp_layer)
                # batch normalization
                if (self.add_batch_normalization):
                    temp_layer = layers.BatchNormalization(axis=-1)(temp_layer)
                # activation
                temp_layer = layers.Activation(self.activation)(temp_layer)
            temp_layer = layers.Add()([temp_layer, x])
            downsampling_layers.append(temp_layer)
            temp_layer = max_pool_layer(pool_size=self.pool_size,
                                        strides=self.pool_strides,
                                        padding=self.padding)(temp_layer)

        # Bottleneck
        temp_layer = ASPP(temp_layer, self.n_initial_filters * pow(2, self.depth))

        # Up sampling branch
        for i in range(self.depth):
            temp_layer = conv_transpose_layer(self.n_initial_filters * pow(2, (self.depth - 1) - i),
                                              kernel_size=self.deconv_kernel_size,
                                              strides=self.deconv_strides,
                                              activation='linear',
                                              padding=self.padding,
                                              kernel_regularizer=self.kernel_regularizer,
                                              bias_regularizer=self.bias_regularizer)(temp_layer)
            # activation
            temp_layer = layers.Activation(self.activation)(temp_layer)

            out_pee = PEE(downsampling_layers[(self.depth - 1) - i], self.n_initial_filters * pow(2, (self.depth - 1) - i))
            temp_layer = RA(temp_layer, out_pee, self.n_initial_filters * pow(2, (self.depth - 1) - i))

            # convolution
            for j in range(2 + math.floor(i/2)):
                # Convolution
                if j == 0:
                    x = conv_layer(self.n_initial_filters * pow(2, (self.depth - 1) - i),
                                   kernel_size=(1, 1, 1),
                                   strides=self.strides,
                                   padding=self.padding,
                                   activation='linear',
                                   kernel_regularizer=self.kernel_regularizer,
                                   bias_regularizer=self.bias_regularizer)(temp_layer)
                temp_layer = conv_layer(self.n_initial_filters * pow(2, (self.depth - 1) - i),
                                        kernel_size=self.kernel_size,
                                        strides=self.strides,
                                        padding=self.padding,
                                        activation='linear',
                                        kernel_regularizer=self.kernel_regularizer,
                                        bias_regularizer=self.bias_regularizer)(temp_layer)
                # activation
                temp_layer = layers.Activation(self.activation)(temp_layer)
            # addition
            temp_layer = layers.Add()([temp_layer, x])

        # Convolution 1 filter sigmoidal (to make size converge to final one)
        temp_layer = conv_layer(self.n_classes, kernel_size=softmax_kernel_size,
                                strides=self.strides,
                                padding='same',
                                activation='linear',
                                kernel_regularizer=self.kernel_regularizer,
                                bias_regularizer=self.bias_regularizer)(temp_layer)

        output_tensor = layers.Softmax(axis=-1)(temp_layer)
        self.model = Model(inputs=[input_tensor], outputs=[output_tensor])

    def set_initial_weights(self, weights):
        '''
        Set the initial weights of the U-Net, in case
        training was stopped and then resumed. An exception
        is raised in case the model currently configured
        has different properties than the one whose weights
        were stored.
        '''
        try:
            self.model.load_weights(weights)
        except:
            raise

    def get_n_parameters(self):
        '''
        Get the total number of parameters of the model
        '''
        return self.model.count_params()

    def summary(self):
        '''
        Print out summary of the model.
        '''
        print(self.model.summary())

