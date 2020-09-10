import tf.keras.layers

class UNet(object):
    '''
    This class provides a simple interface to create
    a U-Net network with custom parameters. 
    '''

    def __init__(self, input_size = (128,128,128,1), 
                        depth = 5,
                        activation = 'relu',
                        padding = 'same',
                        n_filters = 8,
                        kernel_regularizer='',
                        bias_regularizer=''):

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

        self.n_filters = n_filters
        '''
        Number of feature maps in the first layer of the 
        U-Net.
        '''
        
        # Create model
        input_layer = tf.keras.layers.Input(shape=input_size)
        

    
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

class VNet(object):
    '''
    This class provides a simple interface to create a 
    VNet network with custom parameters.
    '''
    def __init__(self, input_size=(128,128,128)):
        pass