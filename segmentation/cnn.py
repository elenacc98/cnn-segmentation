from tensorflow.keras import layers
from tensorflow.keras import regularizers

class UNet(object):
    '''
    This class provides a simple interface to create
    a U-Net network with custom parameters. 
    '''

    def __init__(self, input_size = (128,128,128,1), 
                        kernel_size = (3,3,3),
                        strides = (1,1,1),
                        pool_size = (2,2,2),
                        pool_strides = (2,2,2),
                        depth = 5,
                        activation = 'relu',
                        padding = 'same',
                        n_initial_filters = 8,
                        add_batch_normalization = True,
                        kernel_regularizer = regularizers.l2(0.001),
                        bias_regularizer = regularizers.l2(0.001)):

        '''
        Init function for the class, that allows to
        configure the U-Net model.
        '''

        self.input_size = input_size
        '''
        Input size for the network.
        '''
        
        self.n_dim = len(input_size)
        '''
        Number of dimensions of the input data
        '''
        
        self.kernel_size = kernel_size
        '''
        Size of the kernel to be used in the convolutional
        layers of the U-Net.
        '''

        self.strides = strides
        '''
        Stride shape to be used in the convolutional
        layers of the U-Net.
        '''

        self.depth = depth
        '''
        Depth of the U-Net model.
        '''

        self.activation = activation
        '''
        Activation function used in the U-Net layers.
        '''

        self.padding = padding
        '''
        Padding used for the input data in all the
        U-Net layers.
        '''

        self.n_initial_filters = n_initial_filters
        '''
        Number of feature maps in the first layer of the 
        U-Net.
        '''
        
        self.kernel_regularizer = kernel_regularizer
        '''
        Kernel regularizer to be applied to the convolutional
        layers of the U-Net.
        '''

        self.bias_regularizer = bias_regularizer
        '''
        Bias regularizer to be applied to the convolutional 
        layers of the U-Net.
        '''

        self.add_batch_normalization = add_batch_normalization
        '''
        Boolean flag to determine if batch normalization should
        be applied after convolutional layers.
        '''

        self.pool_size = pool_size
        '''
        Size of the pool size to be used in MaxPooling layers.
        '''

        self.pool_strides = pool_strides
        '''
        Size of the strides to be used in MaxPooling layers.
        '''
        # Create model
        self._create_model()

    def _create_model(self):
        '''
        This function creates a U-Net network based
        on the current configuration.
        '''
        # Check if 2D or 3D convolution must be used
        if (self.n_dim == 2):
            conv_layer = layers.Conv2D
            max_pool_layer = layers.MaxPooling2D
        elif (self.n_dim == 3):
            conv_layer = layers.Conv3D
            max_pool_layer = layers.MaxPooling3D
        else:
            print("Could not handle input dimensions.")
            return
        # Input layer
        temp_layer = layers.Input(shape=input_size)
        
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
            temp_layer = conv_layer(self.n_initial_filters*pow(2,depth), kernel_size = self.kernel_size, 
                                        strides = self.strides, 
                                        padding = self.padding,
                                        activation = self.activation,
                                        kernel_regularizer = self.kernel_regularizer,
                                        bias_regularizer = self.bias_regularizer)(temp_layer)
            if (self.add_batch_normalization):
                temp_layer = temp_layer = layers.BatchNormalization(axis = -1)(temp_layer)
            # activation
            temp_layer = layers.Activation(self.activation)(temp_layer)
        
        # Up sampling branch
        for i in range(self.depth):

            
              # Deconvolution 32 filters 
  cumulative_resulting_tensor = keras.layers.Conv3DTranspose(16, kernel_size = (2, 2, 2), strides = (2, 2, 2), padding = 'same', activation = 'linear', kernel_regularizer = keras.regularizers.l2(L2_REG_LAMBDA), bias_regularizer = keras.regularizers.l2(L2_REG_LAMBDA))(cumulative_resulting_tensor)
  # RELU
  cumulative_resulting_tensor = keras.layers.Activation('relu')(cumulative_resulting_tensor)
  # Concatenation
  cumulative_resulting_tensor = keras.layers.Concatenate(axis = CONCATENATION_DIRECTION_OF_FEATURES)([intermediate_tensor_2, cumulative_resulting_tensor])
  # Convolution 32 filters 
  cumulative_resulting_tensor = keras.layers.Conv3D(16, kernel_size = (3, 3, 3), strides = (1, 1, 1), padding = 'same', activation = 'linear', kernel_regularizer = keras.regularizers.l2(L2_REG_LAMBDA), bias_regularizer = keras.regularizers.l2(L2_REG_LAMBDA))(cumulative_resulting_tensor)
  # RELU
  cumulative_resulting_tensor = keras.layers.Activation('relu')(cumulative_resulting_tensor)
  # Convolution 32 filters 
  cumulative_resulting_tensor = keras.layers.Conv3D(16, kernel_size = (3, 3, 3), strides = (1, 1, 1), padding = 'same', activation = 'linear', kernel_regularizer = keras.regularizers.l2(L2_REG_LAMBDA), bias_regularizer = keras.regularizers.l2(L2_REG_LAMBDA))(cumulative_resulting_tensor)
  # RELU
  cumulative_resulting_tensor = keras.layers.Activation('relu')(cumulative_resulting_tensor)

        

    def set_initial_weights(self, weights):
        '''
        Set the initial weights of the U-Net, in case
        training was stopped and then resumed. An exception
        is raised in case the model currently configured
        has different properties than the one whose weights
        were stored.
        '''
        #assert(isinstance(self.model))
        try:
            self.model.load_weights(weights)
        except:
            raise
    
    def get_n_parameters(self):
        '''
        Get the total number of parameters of the model
        '''
        pass

    def summary(self):
        '''
        Print out summary of the model.
        '''
        pass

class VNet(object):
    '''
    This class provides a simple interface to create a 
    VNet network with custom parameters.
    '''
    def __init__(self, input_size=(128,128,128)):
        pass