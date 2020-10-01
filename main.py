from segmentation.cnn import UNet

unet = UNet(depth=3)
unet.create_model()
print(unet.get_n_parameters())
unet.summary()