"""
The cnn submodule implements some classes to define CNN-based
models in a dynamic way.
"""
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import Model

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

class CELUNet(object):
    """
    This class provides a simple interface to create
    a CEL U-Net network with custom parameters.
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
                 n_initial_filters=4,
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

class ChenUNet(object):
    """
    This class provides a simple interface to create
    a UNet as described in Chen et al. with custom parameters.
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

class BAUNet(object):
    """
    This class provides a simple interface to create
    a BAUNet (Boundary Aware U) network with custom parameters.
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
        out_edge_list = []
        out_mask_list = []
        out_mtl_list = []
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

            if i > 0:
                out_pee = PEE(temp_layer, self.n_initial_filters * pow(2, i))
                out_mtl, out_edge, out_mask = MINI_MTL(out_pee,
                                                       self.n_initial_filters * pow(2, i),
                                                       self.n_classes,
                                                       i)
                out_edge_list.append(out_edge)
                out_mask_list.append(out_mask)
                out_mtl_list.append(out_mtl)
            else:
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
            # IF MINI_MTL is used
            if i != (self.depth - 1):  # don't do that in the shallowest layer
                out_cff = CFF(out_mtl_list,
                              self.input_size[1],
                              self.n_initial_filters * pow(2, (self.depth - 1) - i),
                              (self.depth - 1) - i)

                # Concatenation
                temp_layer = Concatenate()([temp_layer, out_cff])
            else:  # simple concatenation in the shallowest layer
                temp_layer = Concatenate()([temp_layer, downsampling_layers[(self.depth - i) - 1]])

            temp_layer = conv_layer(self.n_initial_filters * pow(2, (self.depth - 1) - i),
                                    kernel_size=(1, 1, 1),
                                    padding='same')(temp_layer)

        # Convolution 1 filter sigmoidal (to make size converge to final one)
        temp_layer = conv_layer(self.n_classes, kernel_size=softmax_kernel_size,
                                strides=self.strides,
                                padding='same',
                                activation='linear',
                                kernel_regularizer=self.kernel_regularizer,
                                bias_regularizer=self.bias_regularizer)(temp_layer)

        output_tensor = layers.Softmax(axis=-1, dtype='float32', name='out_final')(temp_layer)

        self.model = Model(inputs=[input_tensor], outputs=[output_tensor] + out_edge_list + out_mask_list)

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

class CDDUnet(object):
    """
    This class provides a simple interface to create
    a Contextual Deconvolutional Dense Net network with custom parameters.
    Args:
        input_size: input size for the network
        num_classes: number of classes in labels
        weight_decay: weight decay parameter
        growth_rate: number of feature maps produced by each layer in the dense blocks
        n_layers: number of layers in the first dense block. Layers in the subsequent blocks are defined
            according to this parameter.
        theta: fraction to reduce number of features in the transition blocks after each dense block.
        activation: activation function used in the U-Net layers
        kernel_size: size of the kernel to be used in the convolutional layers of the U-Net
        strides: stride shape to be used in the convolutional layers of the U-Net
        deconv_strides: stride shape to be used in the deconvolutional layers of the U-Net
        deconv_kernel_size: kernel size shape to be used in the deconvolutional layers of the U-Net
        pool_size: size of the pool size to be used in MaxPooling layers
        pool_strides: size of the strides to be used in MaxPooling layers
    """

    def __init__(self, input_size,
                 num_classes,
                 weight_decay=1E-4,
                 growth_rate=12,
                 n_layers=6,
                 theta=0.5,
                 activation = 'relu',
                 kernel_size=(3, 3, 3),
                 strides=(1, 1, 1),
                 deconv_kernel_size=(2, 2, 2),
                 deconv_strides=(2, 2, 2),
                 pool_size=(2, 2, 2),
                 pool_strides=(2, 2, 2),
                 ):

        self.input_size = input_size
        self.num_classes = num_classes
        self.weight_decay = weight_decay
        self.growth_rate = growth_rate
        self.n_layers = n_layers
        self.theta = theta
        self.activation = activation
        self.kernel_size = kernel_size
        self.strides = strides
        self.deconv_kernel_size = deconv_kernel_size
        self.deconv_strides = deconv_strides
        self.pool_size = pool_size
        self.pool_strides = pool_strides

        self.n_filters_0 = self.growth_rate/2
        self.n_filters_1 = self.growth_rate
        self.n_filters_2 = self.growth_rate * 2


    def create_model(self):

        input_layer = layers.Input(shape=self.input_size)

        # Convolutional layer 0
        x = Conv3D(self.n_filters_0, self.kernel_size, padding='same')(input_layer)
        x = BatchNormalization(gamma_regularizer=l2(self.weight_decay),
                               beta_regularizer=l2(self.weight_decay))(x)
        encoding_0 = Activation(self.activation, name='encoding_0')(x)  # 192x192x192x6

        # MaxPooling + Conv layer 1
        x = MaxPool3D(self.pool_size)(encoding_0)  # 96x96x96x6
        x = Conv3D(self.n_filters_1, self.kernel_size, padding='same')(x)
        x = BatchNormalization(gamma_regularizer=l2(self.weight_decay),
                               beta_regularizer=l2(self.weight_decay))(x)
        encoding_1 = Activation(self.activation, name='encoding_1')(x)  # 96x96x96x12

        # MaxPooling + Conv layer 2
        x = MaxPool3D(self.pool_size)(encoding_1)  # 48x48x48x12
        x = Conv3D(self.n_filters_2, self.kernel_size, padding='same')(x)
        x = BatchNormalization(gamma_regularizer=l2(self.weight_decay),
                               beta_regularizer=l2(self.weight_decay))(x)
        encoding_2 = Activation(self.activation, name='encoding_2')(x)  # 48x48x48x24

        # Dense Block 3
        x, n_block_3 = denseblock(encoding_2, concat_axis=-1, nb_layers=self.n_layers, nb_filter=self.n_filters_2,
                                  growth_rate=self.growth_rate)  # 48x48x48x96
        encoding_3, n_filters_3 = transition(x, concat_axis=-1, nb_filter=n_block_3,
                                             theta=self.theta)  # 24x24x24x48 (theta=0.5)

        # Dense Block 4
        x, n_block_4 = denseblock(encoding_3, concat_axis=-1, nb_layers=2 * self.n_layers, nb_filter=n_filters_3,
                                  growth_rate=self.growth_rate)  # 24x24x24x192
        encoding_4, n_filters_4 = transition(x, concat_axis=-1, nb_filter=n_block_4,
                                             theta=self.theta)  # 12x12x12x96 (theta=0.5)

        # Dense Block 5
        x, n_block_5 = denseblock(encoding_4, concat_axis=-1, nb_layers=4 * self.n_layers, nb_filter=n_filters_4,
                                  growth_rate=self.growth_rate)  # 12x12x12x348
        encoding_5, n_filters_5 = transition(x, concat_axis=-1, nb_filter=n_block_5, theta=self.theta)  # 6x6x6x192

        decoding_5 = compressionUnit(encoding_5, n_filters_4, kernel_size=self.kernel_size)  # 6x6x6x96

        # First concatenation
        x = upsamplingUnit(encoding_4, decoding_5, n_filters_4, n_filters_4, kernel_size=self.kernel_size,
                           deconv_kernel_size=self.deconv_kernel_size, deconv_strides=self.deconv_strides)
        x = denseUnit(x, kernel_size=self.kernel_size)
        x = denseUnit(x, kernel_size=self.kernel_size)
        decoding_4 = compressionUnit(x, n_filters_3, kernel_size=self.kernel_size)  # 12x12x12x48

        # Second concatenation
        x = upsamplingUnit(encoding_3, decoding_4, n_filters_3, n_filters_3, kernel_size=self.kernel_size,
                           deconv_kernel_size=self.deconv_kernel_size, deconv_strides=self.deconv_strides)  # 24x24x24x96
        x = denseUnit(x, kernel_size=self.kernel_size)
        x = denseUnit(x, kernel_size=self.kernel_size)
        decoding_3 = compressionUnit(x, self.n_filters_2, kernel_size=self.kernel_size)  # 24x24x24x24

        # Third concatenation
        x = upsamplingUnit(encoding_2, decoding_3, self.n_filters_2, self.n_filters_2, kernel_size=self.kernel_size,
                           deconv_kernel_size=self.deconv_kernel_size, deconv_strides=self.deconv_strides)  # 48x48x48x48
        x = denseUnit(x, kernel_size=self.kernel_size)
        x = denseUnit(x, kernel_size=self.kernel_size)
        decoding_2 = compressionUnit(x, self.n_filters_1)  # 48x48x48x12

        # Fourth concatenation
        x = upsamplingUnit(encoding_1, decoding_2, self.n_filters_1, self.n_filters_1, kernel_size=self.kernel_size,
                           deconv_kernel_size=self.deconv_kernel_size, deconv_strides=self.deconv_strides)  # 96x96x96x24
        x = denseUnit(x, kernel_size=self.kernel_size)
        x = denseUnit(x, kernel_size=self.kernel_size)
        decoding_1 = compressionUnit(x, self.n_filters_0)  # 96x96x96x6

        # Last concatenation
        x = upsamplingUnit(encoding_0, decoding_1, self.n_filters_0, self.n_filters_0, kernel_size=self.kernel_size,
                           deconv_kernel_size=self.deconv_kernel_size, deconv_strides=self.deconv_strides)  # 192x192x192x12

        x = BatchNormalization(gamma_regularizer=l2(self.weight_decay),
                               beta_regularizer=l2(self.weight_decay))(x)
        x = Activation(self.activation)(x)
        x = Conv3D(self.num_classes, (1, 1, 1), padding='same', kernel_initializer='he_uniform')(x)
        output_layer = Softmax(axis=-1)(x)

        self.model = Model(inputs=[input_layer], outputs=[output_layer])

    def set_initial_weights(self, weights):
        """
        Set the initial weights of the U-Net, in case
        training was stopped and then resumed. An exception
        is raised in case the model currently configured
        has different properties than the one whose weights
        were stored.
        """
        try:
            self.model.load_weights(weights)
        except:
            raise

    def get_n_parameters(self):
        """
        Get the total number of parameters of the model
        """
        return self.model.count_params()

    def summary(self):
        """
        Print out summary of the model.
        """
        print(self.model.summary())        

class VNet(object):
    '''
    This class provides a simple interface to create a 
    VNet network with custom parameters.
    '''
    def __init__(self, input_size=(128,128,128)):
        pass