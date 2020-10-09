from segmentation.cnn import UNet
from segmentation.metrics import PerClassIoU
from segmentation.callbacks import MetricsPlot

unet = UNet(input_size=(128,128,1),
            depth=5,
            n_initial_filters = 8,
            kernel_size=(3,3),
            strides=(1,1),
            deconv_kernel_size=(2,2),
            deconv_strides=(2,2),
            pool_size=(2,2),
            pool_strides=(2,2))
unet.create_model()

print(unet.summary())