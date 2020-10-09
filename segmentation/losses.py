"""
The losses submodule implements loss function to be used
in segmentation tasks.
"""
from segmentation.metrics import MeanDice

class MeanDiceLoss(MeanDice):

    def __init__(self, num_classes, name=None, dtype=None):
        super(MeanDice, self).__init__(self, name=name, dtype=dtype)
    
    
