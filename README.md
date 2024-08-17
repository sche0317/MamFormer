# MamFormer
 An input-aware multi-contrast MRI synthesis method

Here are some suggestions I would like to make (if you want to reproduce MamFormer).


During the training process, please use GAN loss with caution. Although it can improve the performance of the model, it is very prone to pattern collapse. So during the training process, please pay attention to the following points:
*    Normalize the inputs (very important!!!)
*    Please normalize the images between -1 and 1 and Tanh as the last layer of the generator output (very important!!!)
*    Do not use the std and mean values of original data, because this will causing data information loss when return to original image space.
*    the stability of the GAN game suffers if you have sparse gradients
*    LeakyReLU = good (in both G and D)
*    For Downsampling, use: Average Pooling, Conv2d + stride
*    For Upsampling, use: PixelShuffle, ConvTranspose2d + stride


When train the model, we highly recommend you to convert the source contrast data and target contrast data into tensor form in advance for storage, which can save a lot of data loading time.
