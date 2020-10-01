from segmentation.cnn import UNet
from segmentation.metrics import PerClassIoU
m = PerClassIoU(num_classes=3, class_to_return=0)

unet = UNet(depth=3)
unet.create_model()
print(unet.get_n_parameters())
unet.summary()