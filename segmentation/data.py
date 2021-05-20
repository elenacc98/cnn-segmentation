from tensorflow.keras.utils import Sequence
import numpy as np
import nibabel as nib

class DataGenerator(Sequence):
  """
  Class used for data generators. 
  """
  def __init__(self, id_list, batch_size=10, dim=(128,128,64), shuffle=True, n_classes=3):
    '''
    Function called when initializing the class.
    '''
    self.id_list = id_list
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.dim = dim
    self.on_epoch_end()
    self.n_classes = n_classes

  def on_epoch_end(self):
    '''
    Updates indexes after each epoch. If shuffle is set
    to True, the indexes are shuffled. Shuffling the order in which examples
    are fed to the classifier is helpful so that batches between
    epochs do not look alike. Doing so will eventually make our model more robust.
    '''
    self.indexes = np.arange(len(self.id_list))
    if self.shuffle == True:
      np.random.shuffle(self.indexes)
  
  def __data_generation(self, list_IDs_temp):
    '''
    Generates data containing batch_size samples
    X : (n_samples, *dim, n_channels)
    '''
    # Initialization
    X = np.empty((self.batch_size, *self.dim))
    Y = np.empty((self.batch_size, *self.dim))

    # Generate data
    for index, ID in enumerate(list_IDs_temp):
        # Store volume
        temp_volume = nib.load(ID[0]) 
        temp_volume = temp_volume.get_fdata()
        temp_volume = np.asarray(temp_volume)
        X[index, :, :, :] = temp_volume 
        # Store label
        temp_label = nib.load(ID[1])
        temp_label = temp_label.get_fdata()
        temp_label = np.asarray(temp_label)
        Y[index, :, :, :] = temp_label 
        
   
    X = X.reshape(X.shape + (1,)) # necessary to give it as input to model
    Y = self.remapLabels(Y)
    return X, Y
  
  def __len__(self):
    'Denotes the number of batches per epoch'
    return int(np.floor(len(self.id_list) / self.batch_size))
  
  def __getitem__(self, index):
    'Generate one batch of data'
    # Generate indexes of the batch
    indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

    # Find list of IDs
    id_list_temp = [self.id_list[k] for k in indexes]

    # Generate data
    X, y = self.__data_generation(id_list_temp)

    return X, y
  
  def remapLabels(self, labels_4D):
    labels_5D = np.zeros(labels_4D.shape + (self.n_classes, ))
    # Scan the classes 
    for c in range(self.n_classes):
      temp_indexes = np.where(labels_4D == c)
      labels_5D[temp_indexes + (np.ones(temp_indexes[0].shape, dtype = 'int') * c, )] = 1
    return labels_5D