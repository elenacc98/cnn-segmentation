from segmentation.metrics import MeanDice

class MeanDiceLoss(MeanDice):

    def __init__(self, num_classes, name=name, dtype=dtype):
        super(MeanDice, self).__init__(self, name=name, dtype=dtype)
    
    